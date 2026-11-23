//! Byte-level BPE tokenizer (HuggingFace `tokenizer.json` format).
//!
//! Loads the vocab, merges, byte encoder, added (special) tokens, and the
//! pre-tokenizer regex from a `tokenizer.json` file, then provides
//! `encode`/`decode`. No external dependencies.

use std::collections::{HashMap, HashSet};

use crate::json::Json;
use crate::regex::Regex;

/// The `Split` pre-tokenizer pattern used by the tokenizer.
struct PreTokenizerSpec {
    pattern: String,
}

/// Extract the pre-tokenizer regex from a `tokenizer.json` value.
///
/// Handles a bare `Split` and a `Sequence` of sub-pre-tokenizers (taking the
/// first `Split` member). Newer `tokenizers` versions renamed the sequence key
/// from `pre_tokenizers` to `pretokenizers`. Unknown shapes are rejected: a
/// silent fallback to `\s+` would corrupt every token id without any warning.
fn extract_pre_tokenizer(json: &Json) -> Result<PreTokenizerSpec, String> {
    let pt = json
        .get("pre_tokenizer")
        .ok_or("tokenizer.json missing pre_tokenizer")?;
    let ptype = pt.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let split = if ptype == "Sequence" {
        let subs = pt
            .get("pre_tokenizers")
            .or_else(|| pt.get("pretokenizers"))
            .and_then(|v| v.as_arr())
            .ok_or("pre_tokenizer Sequence has no sub-pre-tokenizers")?;
        subs.iter()
            .find(|s| s.get("type").and_then(|v| v.as_str()) == Some("Split"))
            .ok_or("pre_tokenizer Sequence has no Split member")?
    } else if ptype == "Split" {
        pt
    } else {
        return Err(format!(
            "unsupported pre_tokenizer type '{}' (expected Split)",
            ptype
        ));
    };
    if let Some(behavior) = split.get("behavior").and_then(|v| v.as_str()) {
        // `tokenizers` serializes the SplitBehavior enum as "Isolated" in
        // recent versions and as "isolated" in older ones. Both mean the
        // same thing; our regex engine implements exactly that behavior.
        if !behavior.eq_ignore_ascii_case("isolated") {
            return Err(format!(
                "unsupported Split behavior '{}' (only isolated is implemented; \
                 accepted spellings: 'isolated' or 'Isolated')",
                behavior
            ));
        }
    }
    Ok(PreTokenizerSpec {
        pattern: pattern_field(split)?,
    })
}

/// Read a `pattern` field that may be `{"String": "..."}`, `{"Regex": "..."}`
/// (newer format, with the regex either as a string or as `{"pattern": ...}`),
/// or a bare string.
fn pattern_field(obj: &Json) -> Result<String, String> {
    let p = obj.get("pattern").ok_or("Split has no pattern")?;
    if let Some(s) = p.as_str() {
        return Ok(s.to_string());
    }
    for key in ["String", "Regex"] {
        if let Some(v) = p.get(key) {
            if let Some(s) = v.as_str() {
                return Ok(s.to_string());
            }
            if let Some(s) = v.get("pattern").and_then(|v| v.as_str()) {
                return Ok(s.to_string());
            }
        }
    }
    Err("unsupported Split pattern format".into())
}

/// Build the standard GPT-2 byte-level encoder/decoder map.
///
/// Printable bytes map to themselves; the remaining 68 bytes map to codepoints
/// U+0100..U+0143 in ascending byte order. Used when a `tokenizer.json` omits
/// the explicit `byte_encoder` field (common with newer `tokenizers` versions).
fn build_byte_level_map() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut printable: Vec<u8> = Vec::with_capacity(256);
    printable.extend(0x21..=0x7e); // '!' .. '~'
    printable.extend(0xa1..=0xac); // '¡' .. '¬'
    printable.extend(0xae..=0xff); // '®' .. 'ÿ'

    let mut encoder: HashMap<u8, char> = HashMap::new();
    let mut n = 0u32;
    for b in 0u16..=255 {
        let is_printable = printable.contains(&(b as u8));
        let code = if is_printable {
            b as u32
        } else {
            let c = 256 + n;
            n += 1;
            c
        };
        encoder.insert(b as u8, char::from_u32(code).unwrap_or('\u{fffd}'));
    }
    let decoder: HashMap<char, u8> = encoder.iter().map(|(&b, &c)| (c, b)).collect();
    (encoder, decoder)
}

pub struct Tokenizer {
    /// token string -> id (vocab + added tokens).
    vocab: HashMap<String, u32>,
    /// id -> token string.
    id_to_token: HashMap<u32, String>,
    /// (token_a, token_b) -> merge rank.
    merge_rank: HashMap<(String, String), usize>,
    /// byte value -> byte token char.
    byte_encoder: HashMap<u8, char>,
    /// byte token char -> byte value (inverse of byte_encoder).
    byte_decoder: HashMap<char, u8>,
    /// ids of special (added) tokens.
    special_ids: HashSet<u32>,
    /// All added-token strings (special and non-special), sorted longest-first
    /// so that overlapping tokens match greedily.
    added_tokens: Vec<(String, u32)>,
    unk_id: Option<u32>,
    eos_id: Option<u32>,
    pre_tokenizer: Regex,
    vocab_size: usize,
    chat_template: Option<String>,
}

impl Tokenizer {
    pub fn load(path: &str) -> Result<Tokenizer, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        let json = crate::json::parse(&raw).map_err(|e| format!("parse {}: {}", path, e))?;

        let model = json.get("model").ok_or("tokenizer.json missing 'model'")?;

        // Vocab.
        let mut vocab: HashMap<String, u32> = HashMap::new();
        if let Some(v) = model.get("vocab").and_then(|v| v.as_obj()) {
            for (k, val) in v {
                if let Some(id) = val.as_i64() {
                    vocab.insert(k.clone(), id as u32);
                }
            }
        }

        // Merges. Newer `tokenizers` versions store each merge as a single
        // space-separated string ("Ġ Ġ"); older ones use a two-element array
        // (["Ġ", "Ġ"]). Handle both. Byte-level tokens never contain a literal
        // ASCII space (byte 0x20 maps to U+0120), so split_once is unambiguous.
        let mut merge_rank: HashMap<(String, String), usize> = HashMap::new();
        if let Some(m) = model.get("merges").and_then(|v| v.as_arr()) {
            for (rank, pair) in m.iter().enumerate() {
                let parsed = if let Some(arr) = pair.as_arr() {
                    match (
                        arr.first().and_then(|v| v.as_str()),
                        arr.get(1).and_then(|v| v.as_str()),
                    ) {
                        (Some(a), Some(b)) => Some((a.to_string(), b.to_string())),
                        _ => None,
                    }
                } else if let Some(s) = pair.as_str() {
                    s.split_once(' ')
                        .map(|(a, b)| (a.to_string(), b.to_string()))
                } else {
                    None
                };
                if let Some((a, b)) = parsed {
                    merge_rank.insert((a, b), rank);
                }
            }
        }

        // Byte encoder / decoder. Newer `tokenizers` versions omit the
        // explicit `byte_encoder` field (it is derivable), so fall back to
        // the standard GPT-2 byte-level mapping.
        let (mut byte_encoder, mut byte_decoder) = build_byte_level_map();
        if let Some(be) = model.get("byte_encoder").and_then(|v| v.as_obj()) {
            let mut enc: HashMap<u8, char> = HashMap::new();
            for (k, val) in be {
                if let Ok(byte) = k.parse::<u8>() {
                    if let Some(s) = val.as_str() {
                        if let Some(ch) = s.chars().next() {
                            enc.insert(byte, ch);
                        }
                    }
                }
            }
            if !enc.is_empty() {
                byte_encoder = enc;
                byte_decoder = byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();
            }
        }

        // Added (special) tokens.
        let mut special_ids: HashSet<u32> = HashSet::new();
        let mut added_tokens: Vec<(String, u32)> = Vec::new();
        if let Some(arr) = json.get("added_tokens").and_then(|v| v.as_arr()) {
            for t in arr {
                let id = match t.get("id").and_then(|v| v.as_i64()) {
                    Some(i) => i as u32,
                    None => continue,
                };
                let content = match t.get("content").and_then(|v| v.as_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                vocab.insert(content.clone(), id);
                added_tokens.push((content, id));
                if t.get("special").and_then(|v| v.as_bool()).unwrap_or(false) {
                    special_ids.insert(id);
                }
            }
        }

        // Some added tokens (e.g. ids 248070+) exist only in the sibling
        // `tokenizer_config.json`. Load them too when that file is present.
        if let Some(parent) = std::path::Path::new(path).parent() {
            let cfg_path = parent.join("tokenizer_config.json");
            if let Ok(raw) = std::fs::read_to_string(&cfg_path) {
                if let Ok(cfg) = crate::json::parse(&raw) {
                    if let Some(decoder) = cfg.get("added_tokens_decoder").and_then(|v| v.as_obj())
                    {
                        for (id_str, entry) in decoder {
                            let Ok(id) = id_str.parse::<u32>() else {
                                continue;
                            };
                            let Some(content) = entry.get("content").and_then(|v| v.as_str())
                            else {
                                continue;
                            };
                            // tokenizer.json entries win on duplicate content.
                            if !vocab.contains_key(content) {
                                vocab.insert(content.to_string(), id);
                                added_tokens.push((content.to_string(), id));
                            }
                            if entry
                                .get("special")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                special_ids.insert(id);
                            }
                        }
                    }
                }
            }
        }

        // Longest-first so that e.g. `<|im_start|>` matches before any prefix.
        added_tokens.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        // Build id_to_token in sorted id order. Iterating a HashMap directly
        // made duplicate ids resolve differently between runs.
        let mut pairs: Vec<(u32, &String)> = vocab.iter().map(|(t, &id)| (id, t)).collect();
        pairs.sort();
        let mut id_to_token: HashMap<u32, String> = HashMap::new();
        for (id, tok) in pairs {
            id_to_token.entry(id).or_insert_with(|| tok.clone());
        }

        // unk / eos ids.
        let unk_id = model
            .get("unk_token")
            .and_then(|v| v.as_str())
            .and_then(|s| vocab.get(s).copied());
        let mut eos_id = model
            .get("eos_token")
            .and_then(|v| v.as_str())
            .and_then(|s| vocab.get(s).copied());
        // Qwen3.5 has no eos_token in tokenizer.json; its chat EOS is
        // `<|im_end|>`. Fall back to that (or `<|endoftext|>`).
        if eos_id.is_none() {
            eos_id = vocab
                .get("<|im_end|>")
                .copied()
                .or_else(|| vocab.get("<|endoftext|>").copied());
        }

        // Pre-tokenizer regex. The pre-tokenizer may be a bare `Split`
        // (with a `pattern` field) or a `Sequence` of sub-pre-tokenizers
        // (e.g. Qwen3.5: `Split` + `ByteLevel`). In the `Sequence` case we
        // take the pattern of the first `Split` member; the `ByteLevel`
        // member is handled by `bpe_encode_word` (byte-level conversion).
        let spec = extract_pre_tokenizer(&json)?;
        let pre_tokenizer = Regex::compile(&spec.pattern)
            .map_err(|e| format!("compile pre-tokenizer regex: {}", e))?;

        // Chat template (stored for reference / chat.rs).
        let chat_template = json
            .get("chat_template")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let vocab_size = *vocab.values().max().unwrap_or(&0) as usize + 1;

        Ok(Tokenizer {
            vocab,
            id_to_token,
            merge_rank,
            byte_encoder,
            byte_decoder,
            special_ids,
            added_tokens,
            unk_id,
            eos_id,
            pre_tokenizer,
            vocab_size,
            chat_template,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Override the EOS id from `config.json` (`eos_token_id`), which is more
    /// authoritative than the tokenizer's fallback for raw completions.
    pub fn set_eos_id(&mut self, id: u32) {
        self.eos_id = Some(id);
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// Encode text into token ids.
    ///
    /// Added tokens (e.g. `<|im_start|>`, `<tool_call>`, `<think>`) are
    /// protected: they are matched as plain substrings of the raw input and
    /// emitted directly, bypassing the pre-tokenizer and BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        let mut text_start = 0usize;
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // Added tokens are ASCII today, but a non-ASCII one must not make
            // `&text[text_start..i]` slice through a UTF-8 character.
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let mut matched: Option<(usize, u32)> = None;
            for (tok, id) in &self.added_tokens {
                let tb = tok.as_bytes();
                if i + tb.len() <= bytes.len() && &bytes[i..i + tb.len()] == tb {
                    matched = Some((tb.len(), *id));
                    break; // longest-first ordering
                }
            }
            if let Some((len, id)) = matched {
                // Flush pending text.
                if text_start < i {
                    self.encode_text(&text[text_start..i], &mut out);
                }
                out.push(id);
                i += len;
                text_start = i;
            } else {
                i += 1;
            }
        }
        if text_start < bytes.len() {
            self.encode_text(&text[text_start..], &mut out);
        }
        out
    }

    /// Pre-tokenize + BPE-encode a plain-text span.
    fn encode_text(&self, text: &str, out: &mut Vec<u32>) {
        let words = self.pre_tokenizer.pre_tokenize(text);
        for w in &words {
            out.extend(self.bpe_encode_word(w));
        }
    }

    /// Decode token ids into text.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if skip_special && self.special_ids.contains(&id) {
                continue;
            }
            bytes.extend_from_slice(&self.decode_token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Return the raw bytes produced by a single token id (empty for special
    /// tokens that should be skipped). Used by incremental decoders that need
    /// to buffer incomplete UTF-8 sequences across token boundaries.
    pub fn decode_token_bytes(&self, id: u32) -> Vec<u8> {
        let mut bytes: Vec<u8> = Vec::new();
        let tok = match self.id_to_token.get(&id) {
            Some(t) => t,
            None => return bytes,
        };
        for c in tok.chars() {
            if let Some(&b) = self.byte_decoder.get(&c) {
                bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
        bytes
    }

    /// BPE-encode a single pre-tokenized word into token ids.
    fn bpe_encode_word(&self, word: &str) -> Vec<u32> {
        // 1. Convert to byte tokens.
        let mut tokens: Vec<String> = Vec::new();
        for b in word.bytes() {
            if let Some(&ch) = self.byte_encoder.get(&b) {
                tokens.push(ch.to_string());
            } else {
                // Should not happen; fall back to the raw byte as a char.
                tokens.push(char::from(b).to_string());
            }
        }

        // 2. Apply BPE merges (find the lowest-rank adjacent pair, merge, repeat).
        loop {
            let mut best_i: Option<usize> = None;
            let mut best_rank = usize::MAX;
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].as_str(), tokens[i + 1].as_str());
                if let Some(&rank) = self
                    .merge_rank
                    .get(&(pair.0.to_string(), pair.1.to_string()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let i = match best_i {
                Some(i) => i,
                None => break,
            };
            let merged = tokens[i].clone() + &tokens[i + 1];
            tokens.remove(i + 1);
            tokens[i] = merged;
        }

        // 3. Map to ids.
        tokens
            .iter()
            .map(|t| {
                self.vocab
                    .get(t)
                    .copied()
                    .unwrap_or_else(|| self.unk_id.unwrap_or(0))
            })
            .collect()
    }
}

#[cfg(test)]
impl Tokenizer {
    /// Minimal in-memory tokenizer for HTTP/engine tests (no tokenizer.json).
    pub(crate) fn tiny() -> Tokenizer {
        let (byte_encoder, byte_decoder) = build_byte_level_map();
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: HashMap<u32, String> = HashMap::new();
        for b in 0u16..=255 {
            let ch = byte_encoder[&(b as u8)];
            let s = ch.to_string();
            vocab.insert(s.clone(), b as u32);
            id_to_token.insert(b as u32, s);
        }
        let specials = [
            ("<|im_start|>", 256u32),
            ("<|im_end|>", 257),
            ("<think>", 258),
            ("</think>", 259),
            ("<|endoftext|>", 260),
        ];
        let mut special_ids = HashSet::new();
        let mut added_tokens: Vec<(String, u32)> = Vec::new();
        for (text, id) in specials {
            vocab.insert(text.into(), id);
            id_to_token.insert(id, text.into());
            special_ids.insert(id);
            added_tokens.push((text.into(), id));
        }
        added_tokens.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
        Tokenizer {
            vocab,
            id_to_token,
            merge_rank: HashMap::new(),
            byte_encoder,
            byte_decoder,
            special_ids,
            added_tokens,
            unk_id: None,
            eos_id: Some(257),
            pre_tokenizer: Regex::compile(r"\s+|[^\s]+").unwrap(),
            vocab_size: 261,
            chat_template: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load() -> Option<Tokenizer> {
        Tokenizer::load("/tmp/qwen35/tokenizer.json").ok()
    }

    #[test]
    fn added_token_protection() {
        let t = match load() {
            Some(t) => t,
            None => return,
        };
        let cases: &[(&str, &[u32])] = &[
            ("<|im_start|>", &[248045]),
            ("<|im_end|>", &[248046]),
            ("<|endoftext|>", &[248044]),
            ("<tool_call>", &[248058]),
            ("<think>", &[248068]),
            ("</think>", &[248069]),
            ("a<|im_end|>b", &[64, 248046, 65]),
            ("x<tool_call>y", &[87, 248058, 88]),
            ("<think>hello</think>", &[248068, 14556, 248069]),
        ];
        for (text, expected) in cases {
            let got = t.encode(text);
            assert_eq!(&got, expected, "encode({:?})", text);
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("exllama-tok-{}-{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rejects_unknown_pre_tokenizer_instead_of_falling_back() {
        let dir = temp_dir("bad-pretok");
        let path = dir.join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"model":{"vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"WhitespaceSplit"}}"#,
        )
        .unwrap();
        let err = match Tokenizer::load(path.to_str().unwrap()) {
            Ok(_) => panic!("unknown pre-tokenizer must be rejected"),
            Err(e) => e,
        };
        assert!(err.contains("pre_tokenizer"), "{}", err);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn accepts_both_split_behavior_spellings() {
        let dir = temp_dir("split-behavior");
        let path = dir.join("tokenizer.json");
        // Newer `tokenizers` writes the enum variant as "Isolated"; older
        // versions wrote "isolated". Both must load (regression from v0.4.0).
        for behavior in ["Isolated", "isolated"] {
            let json = format!(
                r#"{{"model":{{"vocab":{{"a":0}},"merges":[]}},"pre_tokenizer":{{"type":"Split","behavior":"{}","pattern":{{"Regex":"\\s+"}}}}}}"#,
                behavior
            );
            std::fs::write(&path, json).unwrap();
            Tokenizer::load(path.to_str().unwrap())
                .unwrap_or_else(|e| panic!("behavior '{}' rejected: {}", behavior, e));
        }
        // Same for the Sequence shape used by the real Qwen3.5 tokenizer.
        std::fs::write(
            &path,
            r#"{"model":{"vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","behavior":"Isolated","pattern":{"Regex":"\\s+"}},{"type":"ByteLevel"}]}}"#,
        )
        .unwrap();
        Tokenizer::load(path.to_str().unwrap()).unwrap();

        // Other split behaviors genuinely change the tokenization semantics.
        std::fs::write(
            &path,
            r#"{"model":{"vocab":{"a":0},"merges":[]},"pre_tokenizer":{"type":"Split","behavior":"Inverted","pattern":{"Regex":"\\s+"}}}"#,
        )
        .unwrap();
        let err = match Tokenizer::load(path.to_str().unwrap()) {
            Ok(_) => panic!("Inverted split behavior must be rejected"),
            Err(e) => e,
        };
        assert!(err.contains("behavior"), "{}", err);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loads_added_tokens_from_tokenizer_config() {
        let dir = temp_dir("config-added");
        let path = dir.join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"model":{"vocab":{"a":0,"!":1},"merges":[]},"added_tokens":[],"pre_tokenizer":{"type":"Split","pattern":{"Regex":"\\s+"}}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"100":{"content":"<ext>","special":true}}}"#,
        )
        .unwrap();
        let t = Tokenizer::load(path.to_str().unwrap()).unwrap();
        assert_eq!(t.encode("<ext>"), vec![100]);
        assert!(t.is_special(100));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn byte_level_map_without_explicit_field() {
        let (enc, dec) = build_byte_level_map();
        // Printable bytes map to themselves.
        assert_eq!(enc[&b'!'], '!');
        assert_eq!(enc[&b'~'], '~');
        // Non-printable bytes map to U+0100.. in ascending order.
        assert_eq!(enc[&0], '\u{0100}');
        assert_eq!(enc[&32], '\u{0120}'); // space -> Ġ
        assert_eq!(enc[&127], '\u{0121}');
        assert_eq!(enc[&173], '\u{0143}');
        assert_eq!(enc[&255], '\u{00ff}');
        // Decoder is the exact inverse.
        for (&b, &c) in &enc {
            assert_eq!(dec[&c], b);
        }
        assert_eq!(enc.len(), 256);
    }

    #[test]
    fn load_tokenizer_without_byte_encoder() {
        // The UnstableLlama EXL3 repo's tokenizer.json omits byte_encoder.
        let t = match Tokenizer::load("/tmp/unstable_tok.json") {
            Ok(t) => t,
            Err(_) => return,
        };
        assert_eq!(t.encode("Hello, world!"), vec![9419, 11, 1814, 0]);
        assert_eq!(t.encode("<|im_start|>"), vec![248045]);
        assert_eq!(t.encode("a<|im_end|>b"), vec![64, 248046, 65]);
    }

    #[test]
    fn encode_reference() {
        let t = match load() {
            Some(t) => t,
            None => return,
        };
        let cases: &[(&str, &[u32])] = &[
            ("Hello, world!", &[9419, 11, 1814, 0]),
            ("a b", &[64, 292]),
            ("multi\nline", &[25429, 198, 1021]),
            ("café", &[895, 56868]),
            ("don't", &[14572, 914]),
            ("it's", &[275, 579]),
        ];
        for (text, expected) in cases {
            let got = t.encode(text);
            assert_eq!(&got, expected, "encode({:?})", text);
        }
    }
}
