//! OpenAI-compatible API handlers and the inference engine.

use crate::chat::{apply_chat_template, Message};
use crate::http::{respond, Request, Streamer};
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::{json::Json, rng::Rng};
use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub struct Params {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub stop_tokens: Vec<u32>,
    /// Emit a final SSE chunk with usage (`stream_options.include_usage`).
    pub include_usage: bool,
    pub enable_thinking: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 0, // 0 = disabled
            max_tokens: 512,
            stop: Vec::new(),
            seed: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
            stop_tokens: Vec::new(),
            include_usage: false,
            enable_thinking: true, // Qwen3.5-9B / 4B default
        }
    }
}

pub struct GenResult {
    pub text: String,
    /// Thinking trace emitted between `<think>` and `</think>` (Qwen3.5).
    pub reasoning_content: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

/// Qwen3.5 added-token ids used as reasoning delimiters.
const TOK_THINK_START: u32 = 248068;
const TOK_THINK_END: u32 = 248069;

/// Emit the longest valid UTF-8 prefix of `pending`, buffering the rest.
fn flush_utf8(pending: &mut Vec<u8>, target: &mut String) -> String {
    let valid = match std::str::from_utf8(pending) {
        Ok(_) => pending.len(),
        Err(e) => e.valid_up_to(),
    };
    if valid == 0 {
        return String::new();
    }
    let chunk = String::from_utf8_lossy(&pending[..valid]).into_owned();
    pending.drain(..valid);
    target.push_str(&chunk);
    chunk
}

/// Monotonic counter used for request ids and for seeding unseeded requests.
/// Wall-clock time alone collides for concurrent requests in the same second.
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Routes generated text into the answer and reasoning streams.
///
/// Three jobs, all of which matter for streaming:
///   1. Assemble UTF-8 characters that are split across token boundaries.
///   2. Hold back a short tail of *answer* text so a `stop` string can be
///      removed before any chunk is sent to the client.
///   3. Track `<think>` / `</think>` and flush buffered text to the stream
///      that was active when that text was generated.
struct OutputRouter {
    split_reasoning: bool,
    in_reasoning: bool,
    stops: Vec<String>,
    max_stop_len: usize,
    /// Bytes of an incomplete UTF-8 character at the token boundary.
    bytes: Vec<u8>,
    /// Complete answer characters withheld for stop-string detection.
    held: String,
    content: String,
    reasoning: String,
}

impl OutputRouter {
    fn new(split_reasoning: bool, start_in_reasoning: bool, stops: Vec<String>) -> Self {
        let max_stop_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        OutputRouter {
            split_reasoning,
            in_reasoning: split_reasoning && start_in_reasoning,
            stops,
            max_stop_len,
            bytes: Vec::new(),
            held: String::new(),
            content: String::new(),
            reasoning: String::new(),
        }
    }

    fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    fn is_delimiter(&self, tok: u32) -> bool {
        self.split_reasoning && (tok == TOK_THINK_START || tok == TOK_THINK_END)
    }

    /// Switch stream after a reasoning delimiter. Text buffered for the old
    /// stream (including the withheld stop-tail) is emitted before toggling.
    /// Returns true if a stop string matched or the client disconnected.
    fn toggle(&mut self, tok: u32, emit: &mut dyn FnMut(&str, bool) -> bool) -> bool {
        let was = self.in_reasoning;
        let mut stopped = self.flush_complete(emit);
        // Incomplete bytes belong to the stream that was active when they
        // were generated; a delimiter can never complete them, so emit them
        // lossily now instead of letting them leak into the other stream.
        if !self.bytes.is_empty() {
            let chunk = String::from_utf8_lossy(&self.bytes).into_owned();
            self.bytes.clear();
            if was {
                self.reasoning.push_str(&chunk);
            } else {
                self.content.push_str(&chunk);
            }
            stopped |= self.emit_chunk(&chunk, was, emit);
        }
        if !self.held.is_empty() {
            let rest = std::mem::take(&mut self.held);
            if !emit(&rest, false) {
                stopped = true;
            }
        }
        self.in_reasoning = tok == TOK_THINK_START;
        stopped
    }

    /// Feed the decoded bytes of a normal token. Returns true when a stop
    /// string matched or the client disconnected.
    fn push(&mut self, text: &[u8], emit: &mut dyn FnMut(&str, bool) -> bool) -> bool {
        self.bytes.extend_from_slice(text);
        self.flush_complete(emit)
    }

    /// Flush every complete character currently buffered.
    fn flush_complete(&mut self, emit: &mut dyn FnMut(&str, bool) -> bool) -> bool {
        let target = if self.in_reasoning {
            &mut self.reasoning
        } else {
            &mut self.content
        };
        let chunk = flush_utf8(&mut self.bytes, target);
        if chunk.is_empty() {
            return false;
        }
        self.emit_chunk(&chunk, self.in_reasoning, emit)
    }

    /// Append a complete chunk to the active stream and stream it, withholding
    /// a short tail from the answer stream for stop-string detection.
    fn emit_chunk(
        &mut self,
        chunk: &str,
        in_reasoning: bool,
        emit: &mut dyn FnMut(&str, bool) -> bool,
    ) -> bool {
        if in_reasoning {
            // `flush_utf8` already appended the chunk to `reasoning`.
            return !emit(chunk, true);
        }

        self.held.push_str(chunk);

        // Stop strings are matched against the full answer text, in order,
        // exactly as before. The withheld tail is guaranteed to contain the
        // suffix because it is at least as long as the longest stop string.
        let before = self.content.len();
        let mut matched = false;
        for s in &self.stops {
            if !s.is_empty() && self.content.ends_with(s.as_str()) {
                self.content.truncate(self.content.len() - s.len());
                matched = true;
            }
        }
        if matched {
            let removed = before - self.content.len();
            if self.held.len() >= removed {
                self.held.truncate(self.held.len() - removed);
            } else {
                self.held.clear();
            }
            let rest = std::mem::take(&mut self.held);
            if !rest.is_empty() {
                let _ = emit(&rest, false);
            }
            return true;
        }

        // Emit everything except a tail that may still grow into a stop
        // string, extending the cut to a UTF-8 character boundary.
        let keep = self.max_stop_len.min(self.held.len());
        let mut cut = self.held.len() - keep;
        while cut > 0 && !self.held.is_char_boundary(cut) {
            cut -= 1;
        }
        if cut > 0 {
            let out = self.held[..cut].to_string();
            self.held = self.held[cut..].to_string();
            if !emit(&out, false) {
                return true;
            }
        }
        false
    }

    /// Called once generation ends: emit the held tail and any leftover bytes.
    fn finish(&mut self, emit: &mut dyn FnMut(&str, bool) -> bool) {
        let _ = self.flush_complete(emit);
        if !self.bytes.is_empty() {
            let chunk = String::from_utf8_lossy(&self.bytes).into_owned();
            self.bytes.clear();
            if self.in_reasoning {
                self.reasoning.push_str(&chunk);
            } else {
                self.content.push_str(&chunk);
            }
            let _ = self.emit_chunk(&chunk, self.in_reasoning, emit);
        }
        if !self.held.is_empty() {
            let rest = std::mem::take(&mut self.held);
            let _ = emit(&rest, false);
        }
    }
}

pub struct Engine {
    model: Mutex<Model>,
    tokenizer: Tokenizer,
    pub model_name: String,
    pub default_max_tokens: usize,
    pub threads: usize,
}

impl Engine {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        model_name: String,
        default_max_tokens: usize,
        threads: usize,
    ) -> Self {
        Engine {
            model: Mutex::new(model),
            tokenizer,
            model_name,
            default_max_tokens,
            threads,
        }
    }

    /// Shared generation core. `prompt_ids` is the already-tokenized prompt.
    /// `split_reasoning` enables Qwen3.5 `<think>`/`</think>` splitting; when
    /// false all decoded text is treated as content (raw completions).
    /// `on_token` receives (chunk, is_reasoning) and returns false when the
    /// client has gone away, which aborts generation promptly.
    fn generate<F>(
        &self,
        prompt_ids: Vec<u32>,
        params: &Params,
        split_reasoning: bool,
        start_in_reasoning: bool,
        mut on_token: F,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool) -> bool,
    {
        let prompt_tokens = prompt_ids.len();
        if prompt_tokens == 0 {
            return Err("empty prompt: no tokens to generate from".into());
        }
        let requested_tokens = if params.max_tokens == 0 {
            self.default_max_tokens
        } else {
            params.max_tokens
        };

        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        model.reset();
        // Clamp the generation budget to the room left in the KV cache.
        // Otherwise the final `forward(&[tok])` can cross context_size and
        // fail with a 500 after the client has already received many tokens.
        let available = model.context_size().saturating_sub(prompt_tokens);
        let max_tokens = requested_tokens.min(available);
        if max_tokens == 0 {
            eprintln!(
                "[done] 0 tokens (prompt fills context: {}/{}) finish_reason=length",
                prompt_tokens,
                model.context_size()
            );
            return Ok(GenResult {
                text: String::new(),
                reasoning_content: String::new(),
                prompt_tokens,
                completion_tokens: 0,
                finish_reason: "length".into(),
            });
        }

        // Prompt prefill in batches so we can report progress + tok/s.
        const PROMPT_BATCH: usize = 32;
        let prompt_start = std::time::Instant::now();
        let mut logits = Vec::new();
        let mut processed = 0usize;
        for chunk in prompt_ids.chunks(PROMPT_BATCH) {
            logits = model.forward(chunk)?;
            processed += chunk.len();
            let secs = prompt_start.elapsed().as_secs_f64().max(1e-6);
            eprintln!(
                "[prompt] {}/{} tokens  ({:.1} tok/s)",
                processed,
                prompt_tokens,
                processed as f64 / secs
            );
        }

        let mut rng = Rng::new(params.seed.unwrap_or_else(default_seed));
        let mut completion: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut router =
            OutputRouter::new(split_reasoning, start_in_reasoning, params.stop.clone());
        let mut finish = String::from("length");

        let debug = std::env::var("EXLLAMA_DEBUG").is_ok();
        const DECODE_BATCH: usize = 16;
        let decode_start = std::time::Instant::now();
        for (step, _) in (0..max_tokens).enumerate() {
            if debug && step < 5 {
                let mut stats = logits.iter().fold(
                    (f32::INFINITY, f32::NEG_INFINITY, 0usize, 0usize),
                    |(mn, mx, nan, inf), &v| {
                        (
                            mn.min(v),
                            mx.max(v),
                            nan + v.is_nan() as usize,
                            inf + v.is_infinite() as usize,
                        )
                    },
                );
                stats.0 = logits.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!(
                    "[dbg] step {} logits min={} max={} nan={} inf={}",
                    step, stats.0, stats.1, stats.2, stats.3
                );
                let mut top: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (id, v) in top.iter().take(6) {
                    eprintln!(
                        "[dbg]   tok {} logit {} text={:?}",
                        id,
                        v,
                        self.tokenizer.decode(&[*id as u32], false)
                    );
                }
            }
            let tok = sample(&logits, params, &mut rng, &completion);
            if is_stop_token(tok, &self.tokenizer) || params.stop_tokens.contains(&tok) {
                finish = "stop".into();
                break;
            }

            // Reasoning delimiters toggle the output stream but are never
            // emitted. They must still be fed through the model: skipping
            // forward() would desync the KV cache / recurrent state / position
            // from the token sequence the model itself generated.
            if router.is_delimiter(tok) {
                logits = model.forward(&[tok])?;
                if router.toggle(tok, &mut on_token) {
                    finish = "stop".into();
                    break;
                }
                continue;
            }
            completion.push(tok);

            // Decode bytes are assembled into complete UTF-8 characters, then
            // routed through the stop-string tail buffer before streaming.
            let stopped = if !self.tokenizer.is_special(tok) {
                router.push(&self.tokenizer.decode_token_bytes(tok), &mut on_token)
            } else {
                false
            };
            if stopped {
                finish = "stop".into();
                break;
            }
            logits = model.forward(&[tok])?;

            // Report decode progress every DECODE_BATCH generated tokens.
            let generated = step + 1;
            if generated.is_multiple_of(DECODE_BATCH) {
                let secs = decode_start.elapsed().as_secs_f64().max(1e-6);
                eprintln!(
                    "[decode] {} tokens generated  ({:.2} tok/s)",
                    generated,
                    generated as f64 / secs
                );
            }
        }
        let decode_secs = decode_start.elapsed().as_secs_f64().max(1e-6);
        eprintln!(
            "[done] {} tokens in {:.2}s  ({:.2} tok/s)  finish_reason={}",
            completion.len(),
            decode_secs,
            completion.len() as f64 / decode_secs,
            finish
        );

        // Emit the withheld tail and any trailing UTF-8 bytes.
        router.finish(&mut on_token);

        Ok(GenResult {
            text: router.content,
            reasoning_content: router.reasoning,
            prompt_tokens,
            completion_tokens: completion.len(),
            finish_reason: finish,
        })
    }

    pub fn run_chat<F>(
        &self,
        messages: &[Message],
        params: &Params,
        on_token: F,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool) -> bool,
    {
        let prompt = apply_chat_template(messages, true, params.enable_thinking);
        let prompt_ids = self.tokenizer.encode(&prompt);
        self.generate(prompt_ids, params, true, params.enable_thinking, on_token)
    }

    pub fn run_completion<F>(
        &self,
        prompt: &str,
        params: &Params,
        on_token: F,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool) -> bool,
    {
        let prompt_ids = self.tokenizer.encode(prompt);
        self.generate(prompt_ids, params, false, false, on_token)
    }
}

fn is_stop_token(tok: u32, tokz: &Tokenizer) -> bool {
    tokz.eos_id().map(|e| tok == e).unwrap_or(false)
}

/// Repetition penalty + temperature + top-k + top-p sampling.
/// `history` is the tokens generated so far; each distinct token in it gets
/// its logit divided by `repetition_penalty` (positive) or multiplied (negative),
/// which breaks greedy repetition loops when the penalty is > 1.
fn sample(logits: &[f32], params: &Params, rng: &mut Rng, history: &[u32]) -> u32 {
    let rep = params.repetition_penalty;
    let need_copy = (rep > 0.0 && rep != 1.0)
        || params.presence_penalty != 0.0
        || params.frequency_penalty != 0.0
        || !params.logit_bias.is_empty();
    // Only copy the logits when some logit adjustment actually applies.
    let mut penalized;
    let logits = if need_copy {
        penalized = logits.to_vec();
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for &t in history {
            let i = t as usize;
            if i < penalized.len() && rep > 0.0 && rep != 1.0 && seen.insert(t) {
                if penalized[i] > 0.0 {
                    penalized[i] /= rep;
                } else {
                    penalized[i] *= rep;
                }
            }
        }
        if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for &t in history {
                *counts.entry(t).or_insert(0) += 1;
            }
            for (&t, &count) in &counts {
                let i = t as usize;
                if i < penalized.len() {
                    penalized[i] -= params.presence_penalty;
                    penalized[i] -= params.frequency_penalty * count as f32;
                }
            }
        }
        for &(id, bias) in &params.logit_bias {
            let i = id as usize;
            if i < penalized.len() {
                penalized[i] += bias;
            }
        }
        &penalized[..]
    } else {
        logits
    };

    let temp = params.temperature;
    if temp <= 0.0 {
        return argmax(logits);
    }

    let n = logits.len();
    let top_k = if (1..n).contains(&params.top_k) {
        params.top_k
    } else {
        0
    };

    // Plain temperature sampling needs neither a sort nor an allocation.
    if top_k == 0 && params.top_p >= 1.0 {
        return sample_softmax(logits, temp, rng);
    }

    // One candidate list; top-k uses an O(n) partition followed by a sort of
    // only the retained candidates instead of sorting the whole vocabulary.
    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temp))
        .collect();
    if top_k > 0 {
        scores.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_k);
    }
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if scores.is_empty() {
        return 0;
    }

    // `exp` is monotonic, so logit order equals probability order. Two passes
    // over the candidates avoid a separate probability vector.
    let max = scores[0].1;
    let sum: f32 = scores.iter().map(|&(_, s)| (s - max).exp()).sum();
    let mass = sum * params.top_p.clamp(0.0, 1.0);
    let target = rng.next_f64() as f32 * mass;
    let mut cum = 0.0f32;
    for &(idx, s) in &scores {
        cum += (s - max).exp();
        if cum >= target {
            return idx as u32;
        }
    }
    scores.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
}

/// Weighted sampling over every logit with no helper vector: one pass for the
/// softmax denominator, one to draw from the CDF.
fn sample_softmax(logits: &[f32], temp: f32, rng: &mut Rng) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / temp;
    let sum: f32 = logits.iter().map(|&l| (l / temp - max).exp()).sum();
    let target = rng.next_f64() as f32 * sum;
    let mut cum = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        cum += (l / temp - max).exp();
        if cum >= target {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best as u32
}

/// Parse a `stop` field that may be a string, array of strings, or null.
pub fn parse_stop(v: Option<&Json>) -> Vec<String> {
    let mut out = Vec::new();
    match v {
        Some(Json::Str(s)) => out.push(s.clone()),
        Some(Json::Arr(a)) => {
            for x in a {
                if let Some(s) = x.as_str() {
                    out.push(s.to_string());
                }
            }
        }
        _ => {}
    }
    out
}

pub fn json_error(status: u16, message: &str, etype: &str) -> (u16, String) {
    let body = Json::Obj(vec![(
        "error".to_string(),
        Json::Obj(vec![
            ("message".to_string(), Json::Str(message.to_string())),
            ("type".to_string(), Json::Str(etype.to_string())),
            ("code".to_string(), Json::Num(status as f64)),
        ]),
    )]);
    (status, body.to_string())
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as f64)
        .unwrap_or(0.0)
}

/// Generate a seed for unseeded requests. Wall-clock time alone repeats for
/// concurrent requests, so mix in the pid and a monotonic counter.
fn default_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let c = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    t ^ (std::process::id() as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ c.wrapping_mul(0xBF58476D1CE4E5B9)
}

fn gen_id(prefix: &str) -> String {
    let c = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "{}-{:016x}",
        prefix,
        now() as u64 ^ ((std::process::id() as u64) << 32) ^ c
    )
}

/// Hard cap on `max_tokens` so a malformed (but positive) value cannot keep a
/// request generating forever.
const MAX_TOKENS_LIMIT: i64 = 1_000_000;

fn json_int(body: &Json, key: &str) -> Result<Option<i64>, (u16, String)> {
    match body.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Num(n))
            if n.is_finite()
                && n.fract() == 0.0
                && *n >= i64::MIN as f64
                && *n <= i64::MAX as f64 =>
        {
            Ok(Some(*n as i64))
        }
        Some(_) => Err(json_error(
            400,
            &format!("{} must be an integer", key),
            "invalid_request_error",
        )),
    }
}

fn json_f32(body: &Json, key: &str) -> Result<Option<f32>, (u16, String)> {
    match body.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Num(n)) if n.is_finite() => Ok(Some(*n as f32)),
        Some(_) => Err(json_error(
            400,
            &format!("{} must be a finite number", key),
            "invalid_request_error",
        )),
    }
}

fn parse_stop_tokens(v: Option<&Json>) -> Vec<u32> {
    let mut out = Vec::new();
    if let Some(Json::Arr(a)) = v {
        for x in a {
            if let Json::Num(n) = x {
                if n.is_finite() && n.fract() == 0.0 && *n >= 0.0 && *n <= u32::MAX as f64 {
                    out.push(*n as u32);
                }
            }
        }
    }
    out
}

fn parse_logit_bias(body: &Json) -> Result<Vec<(u32, f32)>, (u16, String)> {
    let mut out = Vec::new();
    let Some(obj) = body.get("logit_bias").and_then(|v| v.as_obj()) else {
        return Ok(out);
    };
    let bad = |msg: String| json_error(400, &msg, "invalid_request_error");
    for (k, v) in obj {
        let id: u32 = k
            .parse()
            .map_err(|_| bad(format!("logit_bias key '{}' must be a token id", k)))?;
        let bias = v
            .as_f64()
            .ok_or_else(|| bad(format!("logit_bias[{}] must be a number", k)))?;
        if !bias.is_finite() || bias.abs() > 100.0 {
            return Err(bad("logit_bias values must be in [-100, 100]".into()));
        }
        out.push((id, bias as f32));
    }
    Ok(out)
}

fn want_usage(body: &Json) -> bool {
    body.get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn check_model(body: &Json, engine: &Engine) -> Result<(), (u16, String)> {
    if let Some(name) = body.get("model").and_then(|v| v.as_str()) {
        if !name.is_empty() && name != engine.model_name {
            return Err(json_error(
                400,
                &format!(
                    "unknown model '{}'; this server serves '{}'",
                    name, engine.model_name
                ),
                "invalid_request_error",
            ));
        }
    }
    Ok(())
}

fn parse_params(body: &Json) -> Result<Params, (u16, String)> {
    let bad = |msg: &str| json_error(400, msg, "invalid_request_error");
    let mut p = Params::default();

    if let Some(v) = json_f32(body, "temperature")? {
        if v < 0.0 {
            return Err(bad("temperature must be >= 0"));
        }
        p.temperature = v;
    }
    if let Some(v) = json_f32(body, "top_p")? {
        if !(0.0..=1.0).contains(&v) {
            return Err(bad("top_p must be in (0, 1]"));
        }
        p.top_p = v;
    }
    if let Some(v) = json_int(body, "top_k")? {
        if v < 0 || (v as u64) > usize::MAX as u64 {
            return Err(bad("top_k must be a non-negative integer"));
        }
        p.top_k = v as usize;
    }
    // `max_completion_tokens` wins when both are present (OpenAI behavior).
    let mut limit = json_int(body, "max_tokens")?;
    if let Some(v) = json_int(body, "max_completion_tokens")? {
        limit = Some(v);
    }
    if let Some(v) = limit {
        if !(0..=MAX_TOKENS_LIMIT).contains(&v) {
            return Err(bad(&format!(
                "max_tokens must be in [0, {}]",
                MAX_TOKENS_LIMIT
            )));
        }
        p.max_tokens = v as usize;
    }
    if let Some(v) = json_int(body, "seed")? {
        p.seed = Some(v as u64);
    }
    if let Some(v) = json_f32(body, "repetition_penalty")? {
        if v < 0.0 {
            return Err(bad("repetition_penalty must be >= 0"));
        }
        p.repetition_penalty = v;
    }
    for (key, field) in [("presence_penalty", 0usize), ("frequency_penalty", 1usize)] {
        if let Some(v) = json_f32(body, key)? {
            if !(-2.0..=2.0).contains(&v) {
                return Err(bad(&format!("{} must be in [-2, 2]", key)));
            }
            if field == 0 {
                p.presence_penalty = v;
            } else {
                p.frequency_penalty = v;
            }
        }
    }
    p.logit_bias = parse_logit_bias(body)?;
    p.stop = parse_stop(body.get("stop"));
    p.stop_tokens = parse_stop_tokens(body.get("stop"));
    if let Some(Json::Arr(a)) = body.get("stop_token_ids") {
        p.stop_tokens
            .extend(parse_stop_tokens(Some(&Json::Arr(a.clone()))));
    }
    p.include_usage = want_usage(body);
    if let Some(Json::Bool(v)) = body.get("enable_thinking") {
        p.enable_thinking = *v;
    }
    Ok(p)
}

fn want_stream(body: &Json) -> bool {
    body.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn parse_messages(body: &Json) -> Result<Vec<Message>, (u16, String)> {
    let arr = body
        .get("messages")
        .and_then(|v| v.as_arr())
        .ok_or_else(|| json_error(400, "messages is required", "invalid_request_error"))?;
    if arr.is_empty() {
        return Err(json_error(
            400,
            "messages must not be empty",
            "invalid_request_error",
        ));
    }
    let mut out = Vec::new();
    for m in arr {
        let role = m
            .get("role")
            .and_then(|v| v.as_str())
            .ok_or_else(|| json_error(400, "message.role is required", "invalid_request_error"))?
            .to_string();
        let content = m
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        out.push(Message { role, content });
    }
    Ok(out)
}

fn usage_json(r: &GenResult) -> Json {
    Json::Obj(vec![
        ("prompt_tokens".into(), Json::Num(r.prompt_tokens as f64)),
        (
            "completion_tokens".into(),
            Json::Num(r.completion_tokens as f64),
        ),
        (
            "total_tokens".into(),
            Json::Num((r.prompt_tokens + r.completion_tokens) as f64),
        ),
    ])
}

#[allow(clippy::too_many_arguments)]
fn chat_chunk_json(
    id: &str,
    model: &str,
    delta_role: Option<&str>,
    content: Option<&str>,
    reasoning: Option<&str>,
    finish: Option<&str>,
    usage: Option<Json>,
) -> Json {
    let mut delta: Vec<(String, Json)> = Vec::new();
    if let Some(r) = delta_role {
        delta.push(("role".into(), Json::Str(r.into())));
    }
    if let Some(c) = content {
        delta.push(("content".into(), Json::Str(c.into())));
    }
    if let Some(r) = reasoning {
        delta.push(("reasoning_content".into(), Json::Str(r.into())));
    }
    let choice = Json::Obj(vec![
        ("index".into(), Json::Num(0.0)),
        ("delta".into(), Json::Obj(delta)),
        (
            "finish_reason".into(),
            match finish {
                Some(f) => Json::Str(f.into()),
                None => Json::Null,
            },
        ),
    ]);
    let mut obj = vec![
        ("id".into(), Json::Str(id.into())),
        ("object".into(), Json::Str("chat.completion.chunk".into())),
        ("created".into(), Json::Num(now())),
        ("model".into(), Json::Str(model.into())),
        ("choices".into(), Json::Arr(vec![choice])),
    ];
    if let Some(u) = usage {
        obj.push(("usage".into(), u));
    }
    Json::Obj(obj)
}

fn chat_chunk(
    id: &str,
    model: &str,
    delta_role: Option<&str>,
    content: Option<&str>,
    reasoning: Option<&str>,
    finish: Option<&str>,
) -> String {
    chat_chunk_json(id, model, delta_role, content, reasoning, finish, None).to_string()
}

pub fn handle_chat(mut stream: TcpStream, req: Request, engine: &Arc<Engine>) {
    let body: Json = match parse_body(&req) {
        Ok(b) => b,
        Err(e) => {
            let (s, b) = e;
            respond(&mut stream, s, "application/json", &b);
            return;
        }
    };
    if let Err(e) = check_model(&body, engine) {
        let (s, b) = e;
        respond(&mut stream, s, "application/json", &b);
        return;
    }
    let messages = match parse_messages(&body) {
        Ok(m) => m,
        Err(e) => {
            let (s, b) = e;
            respond(&mut stream, s, "application/json", &b);
            return;
        }
    };
    let params = match parse_params(&body) {
        Ok(p) => p,
        Err(e) => {
            let (s, b) = e;
            respond(&mut stream, s, "application/json", &b);
            return;
        }
    };
    let stream_mode = want_stream(&body);
    let id = gen_id("chatcmpl");
    let model_name = engine.model_name.clone();

    if stream_mode {
        let mut sse = Streamer::new(stream, 200, "text/event-stream");
        // First chunk carries the role.
        sse.write_chunk(&format!(
            "data: {}\n\n",
            chat_chunk(&id, &model_name, Some("assistant"), None, None, None)
        ));
        let res = engine.run_chat(&messages, &params, |chunk, is_reasoning| {
            if chunk.is_empty() {
                return true;
            }
            let (content, reasoning) = if is_reasoning {
                (None, Some(chunk))
            } else {
                (Some(chunk), None)
            };
            sse.write_chunk(&format!(
                "data: {}\n\n",
                chat_chunk(&id, &model_name, None, content, reasoning, None)
            ))
        });
        match res {
            Ok(r) => {
                let usage = params.include_usage.then(|| usage_json(&r));
                sse.write_chunk(&format!(
                    "data: {}\n\n",
                    chat_chunk_json(
                        &id,
                        &model_name,
                        None,
                        None,
                        None,
                        Some(r.finish_reason.as_str()),
                        usage
                    )
                ));
                sse.write_chunk("data: [DONE]\n\n");
                sse.end();
            }
            Err(e) => {
                sse.write_chunk(&format!(
                    "data: {}\n\n",
                    json_error(500, &e, "server_error").1
                ));
                sse.end();
            }
        }
    } else {
        let res = engine.run_chat(&messages, &params, |_, _| true);
        match res {
            Ok(r) => {
                let usage = usage_json(&r);
                let mut message = vec![
                    ("role".into(), Json::Str("assistant".into())),
                    ("content".into(), Json::Str(r.text)),
                ];
                // Only include reasoning_content when there is a thinking trace,
                // matching vLLM/SGLang behavior.
                if !r.reasoning_content.is_empty() {
                    message.push(("reasoning_content".into(), Json::Str(r.reasoning_content)));
                }
                let resp = Json::Obj(vec![
                    ("id".into(), Json::Str(id)),
                    ("object".into(), Json::Str("chat.completion".into())),
                    ("created".into(), Json::Num(now())),
                    ("model".into(), Json::Str(model_name)),
                    (
                        "choices".into(),
                        Json::Arr(vec![Json::Obj(vec![
                            ("index".into(), Json::Num(0.0)),
                            ("message".into(), Json::Obj(message)),
                            ("finish_reason".into(), Json::Str(r.finish_reason)),
                            ("logprobs".into(), Json::Null),
                        ])]),
                    ),
                    ("usage".into(), usage),
                ]);
                respond(&mut stream, 200, "application/json", &resp.to_string());
            }
            Err(e) => {
                let (s, b) = json_error(500, &e, "server_error");
                respond(&mut stream, s, "application/json", &b);
            }
        }
    }
}

pub fn handle_completions(mut stream: TcpStream, req: Request, engine: &Arc<Engine>) {
    let body: Json = match parse_body(&req) {
        Ok(b) => b,
        Err(e) => {
            let (s, b) = e;
            respond(&mut stream, s, "application/json", &b);
            return;
        }
    };
    if let Err(e) = check_model(&body, engine) {
        let (s, b) = e;
        respond(&mut stream, s, "application/json", &b);
        return;
    }
    // prompt may be a string or an array of strings.
    let prompt = match body.get("prompt") {
        Some(Json::Str(s)) => s.clone(),
        Some(Json::Arr(a)) => a
            .iter()
            .filter_map(|x| x.as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => {
            let (s, b) = json_error(400, "prompt is required", "invalid_request_error");
            respond(&mut stream, s, "application/json", &b);
            return;
        }
    };
    if prompt.is_empty() {
        let (s, b) = json_error(400, "prompt must not be empty", "invalid_request_error");
        respond(&mut stream, s, "application/json", &b);
        return;
    }
    let params = match parse_params(&body) {
        Ok(p) => p,
        Err(e) => {
            let (s, b) = e;
            respond(&mut stream, s, "application/json", &b);
            return;
        }
    };
    let stream_mode = want_stream(&body);
    let id = gen_id("cmpl");
    let model_name = engine.model_name.clone();

    if stream_mode {
        let mut sse = Streamer::new(stream, 200, "text/event-stream");
        let res = engine.run_completion(&prompt, &params, |chunk, _| {
            if chunk.is_empty() {
                return true;
            }
            let c = Json::Obj(vec![
                ("text".into(), Json::Str(chunk.to_string())),
                ("index".into(), Json::Num(0.0)),
                ("logprobs".into(), Json::Null),
            ]);
            sse.write_chunk(&format!(
                "data: {}\n\n",
                Json::Obj(vec![
                    ("id".into(), Json::Str(id.clone())),
                    ("object".into(), Json::Str("text_completion".into())),
                    ("created".into(), Json::Num(now())),
                    ("model".into(), Json::Str(model_name.clone())),
                    ("choices".into(), Json::Arr(vec![c])),
                ])
            ))
        });
        match res {
            Ok(r) => {
                let usage = params.include_usage.then(|| usage_json(&r));
                let c = Json::Obj(vec![
                    ("text".into(), Json::Str(String::new())),
                    ("index".into(), Json::Num(0.0)),
                    ("logprobs".into(), Json::Null),
                    ("finish_reason".into(), Json::Str(r.finish_reason)),
                ]);
                let mut final_chunk = vec![
                    ("id".into(), Json::Str(id)),
                    ("object".into(), Json::Str("text_completion".into())),
                    ("created".into(), Json::Num(now())),
                    ("model".into(), Json::Str(model_name)),
                    ("choices".into(), Json::Arr(vec![c])),
                ];
                if let Some(u) = usage {
                    final_chunk.push(("usage".into(), u));
                }
                sse.write_chunk(&format!("data: {}\n\n", Json::Obj(final_chunk)));
                sse.write_chunk("data: [DONE]\n\n");
                sse.end();
            }
            Err(e) => {
                sse.write_chunk(&format!(
                    "data: {}\n\n",
                    json_error(500, &e, "server_error").1
                ));
                sse.end();
            }
        }
    } else {
        let res = engine.run_completion(&prompt, &params, |_, _| true);
        match res {
            Ok(r) => {
                let usage = usage_json(&r);
                let resp = Json::Obj(vec![
                    ("id".into(), Json::Str(id)),
                    ("object".into(), Json::Str("text_completion".into())),
                    ("created".into(), Json::Num(now())),
                    ("model".into(), Json::Str(model_name)),
                    (
                        "choices".into(),
                        Json::Arr(vec![Json::Obj(vec![
                            ("text".into(), Json::Str(r.text)),
                            ("index".into(), Json::Num(0.0)),
                            ("logprobs".into(), Json::Null),
                            ("finish_reason".into(), Json::Str(r.finish_reason)),
                        ])]),
                    ),
                    ("usage".into(), usage),
                ]);
                respond(&mut stream, 200, "application/json", &resp.to_string());
            }
            Err(e) => {
                let (s, b) = json_error(500, &e, "server_error");
                respond(&mut stream, s, "application/json", &b);
            }
        }
    }
}

fn parse_body(req: &Request) -> Result<Json, (u16, String)> {
    if req.body.is_empty() {
        return Err(json_error(
            400,
            "empty request body",
            "invalid_request_error",
        ));
    }
    let s = std::str::from_utf8(&req.body)
        .map_err(|_| json_error(400, "body must be utf-8", "invalid_request_error"))?;
    crate::json::parse(s).map_err(|e| {
        json_error(
            400,
            &format!("invalid JSON: {}", e),
            "invalid_request_error",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener};

    fn start_server() -> SocketAddr {
        let engine = Arc::new(Engine::new(
            crate::model::Model::tiny(),
            Tokenizer::tiny(),
            "tiny-model".into(),
            64,
            1,
        ));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            crate::http::serve(listener, move |s, r| crate::route(s, r, &engine));
        });
        addr
    }

    fn raw(addr: SocketAddr, request: &str) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut resp = String::new();
        stream.read_to_string(&mut resp).unwrap();
        resp
    }

    fn post(addr: SocketAddr, path: &str, payload: &str) -> String {
        raw(
            addr,
            &format!(
                "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                path,
                payload.len(),
                payload
            ),
        )
    }

    fn get(addr: SocketAddr, path: &str) -> String {
        raw(
            addr,
            &format!(
                "GET {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
                path
            ),
        )
    }

    fn status(resp: &str) -> u16 {
        resp[9..12].parse().unwrap()
    }

    fn body(resp: &str) -> &str {
        resp.split_once("\r\n\r\n").unwrap().1
    }

    #[test]
    fn repetition_penalty_breaks_greedy_loop() {
        let mut params = Params {
            temperature: 0.0, // greedy
            ..Default::default()
        };
        let mut rng = Rng::new(1);
        // Token 5 has the highest logit; with a history containing it, a
        // repetition penalty > 1 should make a different token win.
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 10.0];
        assert_eq!(sample(&logits, &params, &mut rng, &[]), 5);
        params.repetition_penalty = 2.0;
        assert_eq!(sample(&logits, &params, &mut rng, &[5]), 4);
        // Repeated history entries must not stack (HashSet dedup).
        assert_eq!(sample(&logits, &params, &mut rng, &[5, 5, 5]), 4);
    }

    #[test]
    fn generation_is_clamped_to_remaining_context() {
        let engine = Engine::new(
            crate::model::Model::tiny(),
            Tokenizer::tiny(),
            "tiny-model".into(),
            64,
            1,
        );
        let params = Params {
            temperature: 0.0,
            max_tokens: 1000,
            enable_thinking: false,
            ..Default::default()
        };
        let messages = vec![Message {
            role: "user".into(),
            content: "hi".into(),
        }];
        let res = engine.run_chat(&messages, &params, |_, _| true).unwrap();
        assert!(res.completion_tokens > 0);
        assert!(res.completion_tokens <= 64 - res.prompt_tokens);
        assert_eq!(res.finish_reason, "length");

        // A prompt that already fills the context returns cleanly instead of
        // failing in `forward` on the first decode step.
        let full = engine
            .generate(vec![0u32; 64], &params, false, false, |_, _| true)
            .unwrap();
        assert_eq!(full.completion_tokens, 0);
        assert_eq!(full.finish_reason, "length");
        assert!(full.text.is_empty());

        // An empty prompt has no distribution to sample from.
        assert!(engine
            .generate(vec![], &params, false, false, |_, _| true)
            .is_err());
    }

    #[test]
    fn flush_utf8_buffers_incomplete_chars() {
        let mut buf = Vec::new();
        let mut target = String::new();
        // Emoji 😊 = 4 bytes F0 9F 98 8A. Feed first 3 bytes: nothing emitted.
        buf.extend_from_slice(&[0xF0, 0x9F, 0x98]);
        assert_eq!(flush_utf8(&mut buf, &mut target), "");
        assert_eq!(buf.len(), 3);
        assert!(target.is_empty());
        // Complete the char: emit it.
        buf.push(0x8A);
        assert_eq!(flush_utf8(&mut buf, &mut target), "😊");
        assert_eq!(target, "😊");
        assert!(buf.is_empty());
        // ASCII after multibyte works too.
        buf.extend_from_slice(b"ok!");
        assert_eq!(flush_utf8(&mut buf, &mut target), "ok!");
        assert_eq!(target, "😊ok!");
    }

    #[test]
    fn chat_chunk_reasoning_delta() {
        let s = chat_chunk("id", "model", None, None, Some("thinking..."), None);
        assert!(s.contains("reasoning_content"));
        assert!(s.contains("thinking..."));
        let s2 = chat_chunk("id", "model", None, Some("answer"), None, None);
        assert!(s2.contains("\"content\":\"answer\""));
        assert!(!s2.contains("reasoning_content"));
    }

    #[test]
    fn router_delimiters_toggle_streams_and_flush_before_switch() {
        let mut r = OutputRouter::new(true, true, vec![]);
        let mut emitted: Vec<(String, bool)> = Vec::new();
        assert!(r.in_reasoning());
        assert!(r.is_delimiter(TOK_THINK_START));
        assert!(r.is_delimiter(TOK_THINK_END));
        assert!(!r.is_delimiter(248_000));

        // Text generated as reasoning is flushed to that stream before </think>.
        {
            let mut emit = |s: &str, reasoning: bool| {
                emitted.push((s.into(), reasoning));
                true
            };
            assert!(!r.push(b"think", &mut emit));
            assert!(!r.toggle(TOK_THINK_END, &mut emit));
        }
        assert!(!r.in_reasoning());
        assert_eq!(r.reasoning, "think");
        assert!(emitted.iter().any(|(s, b)| s == "think" && *b));

        // Subsequent text goes to the answer stream.
        {
            let mut emit = |s: &str, reasoning: bool| {
                emitted.push((s.into(), reasoning));
                true
            };
            assert!(!r.push(b"answer", &mut emit));
            r.finish(&mut emit);
        }
        assert_eq!(r.content, "answer");
        assert!(emitted.iter().any(|(s, b)| s == "answer" && !*b));
    }

    #[test]
    fn router_multibyte_char_does_not_cross_stream_switch() {
        let mut r = OutputRouter::new(true, false, vec![]);
        let mut emitted: Vec<(String, bool)> = Vec::new();
        let mut emit = |s: &str, reasoning: bool| {
            emitted.push((s.into(), reasoning));
            true
        };
        r.toggle(TOK_THINK_START, &mut emit);
        assert!(r.in_reasoning());
        // First three bytes of an emoji, then a delimiter before the last byte.
        r.push(&[0xF0, 0x9F, 0x98], &mut emit);
        r.toggle(TOK_THINK_END, &mut emit);
        r.push(&[0x8A], &mut emit);
        r.finish(&mut emit);
        // The partial bytes were flushed to the reasoning stream before the
        // switch; the leftover continuation byte cannot form an emoji in the
        // answer stream.
        assert!(r.reasoning.contains('\u{fffd}'));
        assert!(emitted.iter().any(|(s, b)| s.contains('\u{fffd}') && *b));
        assert!(!r.content.contains('😊'));
        assert!(!emitted.iter().any(|(s, b)| !*b && s.contains('😊')));
    }

    #[test]
    fn router_strips_stop_string_before_streaming() {
        let mut r = OutputRouter::new(false, false, vec!["STOP".into()]);
        let mut emitted: Vec<String> = Vec::new();
        let mut emit = |s: &str, _: bool| {
            emitted.push(s.to_string());
            true
        };
        assert!(!r.push(b"Hello ", &mut emit));
        assert!(!r.push(b"world", &mut emit));
        assert!(!r.push(b"ST", &mut emit));
        assert!(r.push(b"OP", &mut emit));
        r.finish(&mut emit);
        assert_eq!(r.content, "Hello world");
        let streamed: String = emitted.concat();
        assert!(!streamed.contains("STOP"), "streamed={:?}", streamed);
        assert_eq!(streamed, "Hello world");
    }

    #[test]
    fn router_strips_overlapping_stop_strings() {
        let mut r = OutputRouter::new(false, false, vec!["STOP".into(), "OP".into()]);
        let mut emitted = String::new();
        let mut emit = |s: &str, _: bool| {
            emitted.push_str(s);
            true
        };
        assert!(!r.push(b"answer ", &mut emit));
        assert!(r.push(b"STOP", &mut emit));
        assert_eq!(r.content, "answer ");
        assert_eq!(emitted, "answer ");
    }

    #[test]
    fn parse_params_rejects_invalid_values() {
        for body in [
            r#"{"max_tokens": -1}"#,
            r#"{"max_tokens": 2000000}"#,
            r#"{"max_tokens": 1.5}"#,
            r#"{"top_k": -1}"#,
            r#"{"top_k": 1.5}"#,
            r#"{"temperature": -0.5}"#,
            r#"{"temperature": "hot"}"#,
            r#"{"top_p": 1.5}"#,
            r#"{"repetition_penalty": -1}"#,
        ] {
            let j = crate::json::parse(body).unwrap();
            assert!(parse_params(&j).is_err(), "body={}", body);
        }
        let j = crate::json::parse(r#"{"temperature":0.7,"top_p":0.9,"max_tokens":128}"#).unwrap();
        let p = parse_params(&j).unwrap();
        assert_eq!(p.temperature, 0.7);
        assert_eq!(p.top_p, 0.9);
        assert_eq!(p.max_tokens, 128);

        // max_completion_tokens overrides max_tokens.
        let j = crate::json::parse(r#"{"max_tokens":0,"max_completion_tokens":64}"#).unwrap();
        assert_eq!(parse_params(&j).unwrap().max_tokens, 64);
    }

    /// The pre-Phase-1 sampling algorithm, kept as a reference for equivalence.
    fn reference_sample(logits: &[f32], p: &Params, rng: &mut Rng, history: &[u32]) -> u32 {
        let mut logits = logits.to_vec();
        if p.repetition_penalty > 0.0 && p.repetition_penalty != 1.0 {
            let mut seen = std::collections::HashSet::new();
            for &t in history {
                let i = t as usize;
                if i < logits.len() && seen.insert(t) {
                    if logits[i] > 0.0 {
                        logits[i] /= p.repetition_penalty;
                    } else {
                        logits[i] *= p.repetition_penalty;
                    }
                }
            }
        }
        if p.temperature <= 0.0 {
            return argmax(&logits);
        }
        let mut scores: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l / p.temperature))
            .collect();
        if (1..=logits.len()).contains(&p.top_k) {
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scores.truncate(p.top_k);
        }
        let max = scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = scores.iter().map(|(_, s)| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for e in exps.iter_mut() {
            *e /= sum;
        }
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            exps[b]
                .partial_cmp(&exps[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut keep = Vec::new();
        for &i in &order {
            keep.push(i);
            cum += exps[i];
            if cum >= p.top_p {
                break;
            }
        }
        let ksum: f32 = keep.iter().map(|&i| exps[i]).sum();
        let r = rng.next_f64() as f32 * ksum;
        let mut cdf = 0.0f32;
        for &i in &keep {
            cdf += exps[i];
            if r <= cdf {
                return i as u32;
            }
        }
        keep.last().copied().unwrap_or(0) as u32
    }

    #[test]
    fn greedy_sample_matches_reference() {
        let logits: Vec<f32> = (0..256)
            .map(|i| ((i * 7919) % 10007) as f32 * 0.01 - 3.0)
            .collect();
        let p = Params {
            temperature: 0.0,
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let history = [3u32, 3, 17];
        let mut a = Rng::new(1);
        let mut b = Rng::new(1);
        assert_eq!(
            sample(&logits, &p, &mut a, &history),
            reference_sample(&logits, &p, &mut b, &history)
        );
    }

    #[test]
    fn sample_fast_path_distribution_matches_reference() {
        let logits = [3.0f32, 2.0, 1.0, 0.0];
        let p = Params {
            temperature: 1.0,
            ..Default::default()
        };
        let n = 20_000u64;
        let mut new_counts = [0usize; 4];
        let mut ref_counts = [0usize; 4];
        for seed in 1..=n {
            new_counts[sample(&logits, &p, &mut Rng::new(seed), &[]) as usize] += 1;
            ref_counts[reference_sample(&logits, &p, &mut Rng::new(seed), &[]) as usize] += 1;
        }
        for i in 0..4 {
            let a = new_counts[i] as f64 / n as f64;
            let b = ref_counts[i] as f64 / n as f64;
            assert!((a - b).abs() < 0.02, "token {}: new={} ref={}", i, a, b);
        }
    }

    #[test]
    fn sample_respects_top_p() {
        // The dominant logit carries > 1 - 1e-4 of the mass, so a 0.5 nucleus
        // must always keep only that token.
        let logits = [10.0f32, 1.0, 0.0, -1.0];
        let p = Params {
            temperature: 1.0,
            top_p: 0.5,
            ..Default::default()
        };
        for seed in 0..128u64 {
            assert_eq!(sample(&logits, &p, &mut Rng::new(seed), &[]), 0);
        }
    }

    #[test]
    fn sample_respects_top_k() {
        let mut logits = vec![-5.0f32; 64];
        logits[10] = 4.0;
        logits[42] = 3.0;
        let p = Params {
            temperature: 0.7,
            top_k: 2,
            ..Default::default()
        };
        for seed in 0..64u64 {
            let tok = sample(&logits, &p, &mut Rng::new(seed), &[]);
            assert!(tok == 10 || tok == 42, "unexpected token {}", tok);
        }
    }

    #[test]
    fn request_ids_and_default_seeds_are_unique() {
        assert_ne!(gen_id("chatcmpl"), gen_id("chatcmpl"));
        assert_ne!(default_seed(), default_seed());
    }

    /// Opt-in end-to-end throughput benchmark. Requires a real EXL3 model:
    ///   EXLLAMA_BENCH_MODEL=/path/to/model cargo test --release \
    ///       bench_generation_throughput -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_generation_throughput() {
        let dir = match std::env::var("EXLLAMA_BENCH_MODEL") {
            Ok(p) => p,
            Err(_) => return,
        };
        let model = crate::model::Model::load(&dir, 4096, 4, "f32").unwrap();
        let tok_path = std::path::Path::new(&dir).join("tokenizer.json");
        let tokenizer = Tokenizer::load(&tok_path.to_string_lossy()).unwrap();
        let engine = Engine::new(model, tokenizer, "bench".into(), 32, 4);
        let params = Params {
            temperature: 0.0,
            max_tokens: 32,
            enable_thinking: false,
            ..Default::default()
        };
        let messages = vec![Message {
            role: "user".into(),
            content: "Count: 1, 2, 3, 4,".into(),
        }];

        let t0 = std::time::Instant::now();
        let res = engine.run_chat(&messages, &params, |_, _| true).unwrap();
        let secs = t0.elapsed().as_secs_f64().max(1e-6);
        let total = res.prompt_tokens + res.completion_tokens;
        eprintln!(
            "BENCH generation: {} prompt + {} completion tokens in {:.2}s \
             ({:.2} tok/s overall, {:.2} decode tok/s, finish={})",
            res.prompt_tokens,
            res.completion_tokens,
            secs,
            total as f64 / secs,
            res.completion_tokens as f64 / secs,
            res.finish_reason
        );
    }

    // --- std-only HTTP integration tests against the tiny synthetic model ---

    #[test]
    fn http_health_models_cors_and_404() {
        let addr = start_server();
        let r = get(addr, "/health");
        assert_eq!(status(&r), 200);
        assert!(body(&r).contains("\"status\":\"ok\""));
        assert!(r.contains("Access-Control-Allow-Origin: *"));

        let r = get(addr, "/v1/models");
        assert_eq!(status(&r), 200);
        assert!(body(&r).contains("\"object\":\"list\""));

        let r = raw(
            addr,
            "OPTIONS /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert_eq!(status(&r), 204);
        assert!(r.contains("Access-Control-Allow-Origin: *"));

        assert_eq!(status(&get(addr, "/nope")), 404);
    }

    #[test]
    fn http_rejects_bad_requests() {
        let addr = start_server();
        let cases = [
            ("/v1/chat/completions", "{}"),
            ("/v1/chat/completions", "not json"),
            (
                "/v1/chat/completions",
                r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}"#,
            ),
            (
                "/v1/chat/completions",
                r#"{"model":"wrong","messages":[{"role":"user","content":"hi"}]}"#,
            ),
            ("/v1/completions", "{}"),
            ("/v1/chat/completions", r#"{"messages":[]}"#),
            ("/v1/completions", r#"{"prompt":""}"#),
        ];
        for (path, payload) in cases {
            let r = post(addr, path, payload);
            assert_eq!(status(&r), 400, "payload={} response={}", payload, r);
            assert!(body(&r).contains("\"error\""));
        }
    }

    #[test]
    fn http_chat_json_response_and_usage() {
        let addr = start_server();
        let payload = r#"{"model":"tiny-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"enable_thinking":false}"#;
        let r = post(addr, "/v1/chat/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        let b = body(&r);
        for needle in [
            "\"chat.completion\"",
            "\"usage\"",
            "\"prompt_tokens\"",
            "\"completion_tokens\"",
            "\"finish_reason\"",
            "\"content\"",
        ] {
            assert!(b.contains(needle), "missing {} in {}", needle, b);
        }
    }

    #[test]
    fn http_chat_stream_framing_done_and_usage() {
        let addr = start_server();
        let payload = r#"{"model":"tiny-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"enable_thinking":false,"stream":true,"stream_options":{"include_usage":true}}"#;
        let r = post(addr, "/v1/chat/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        assert!(r.contains("Transfer-Encoding: chunked"));
        assert!(r.contains("Cache-Control: no-cache"));
        assert!(r.contains("text/event-stream"));
        let b = body(&r);
        assert!(b.contains("data: [DONE]"), "{}", b);
        assert!(b.contains("\"usage\""), "{}", b);
        assert!(
            b.contains("\"finish_reason\":\"stop\"") || b.contains("\"finish_reason\":\"length\"")
        );
        let first = b.lines().next().unwrap_or("");
        assert!(
            first.chars().all(|c| c.is_ascii_hexdigit()) && !first.is_empty(),
            "bad chunk header: {:?}",
            first
        );
    }

    #[test]
    fn http_completions_stream() {
        let addr = start_server();
        let payload =
            r#"{"model":"tiny-model","prompt":"hi","max_tokens":8,"temperature":0,"stream":true}"#;
        let r = post(addr, "/v1/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        assert!(body(&r).contains("data: [DONE]"));
        assert!(body(&r).contains("text_completion"));
    }

    #[test]
    fn http_rejects_oversized_body() {
        let addr = start_server();
        let req = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100000000\r\n\r\n{}";
        let r = raw(addr, req);
        assert!(!r.contains("200 OK"), "{}", r);
    }
}
