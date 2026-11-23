//! Deterministic, dependency-free fuzz smoke tests.
//!
//! These are not a replacement for `cargo fuzz`, but they exercise the same
//! untrusted-input parsers (JSON, HTTP, regex, tokenizer, chat, API params)
//! with thousands of seeded random inputs on every `cargo test` run. Any panic
//! is a bug; parse errors are expected and ignored. No new crate is needed.

use crate::chat::{apply_chat_template, Message};
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use std::io::Write;
use std::net::{Shutdown, TcpListener, TcpStream};

/// Small xorshift-style LCG. Deterministic so failures are reproducible.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 1
    }

    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

const JSON_ATOMS: &[u8] = b"{}[],:\"\\0123456789.eE+-truefalsnul \n\t\rxyz";

#[test]
fn fuzz_json_parse_serialize_roundtrip() {
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    for _ in 0..2000 {
        let len = rng.pick(80);
        let s: String = (0..len)
            .map(|_| JSON_ATOMS[rng.pick(JSON_ATOMS.len())] as char)
            .collect();
        if let Ok(v) = crate::json::parse(&s) {
            let out = v.to_string();
            let reparsed = crate::json::parse(&out);
            assert!(reparsed.is_ok(), "roundtrip failed: {:?} -> {:?}", s, out);
        }
    }
    // Arbitrary (often invalid) UTF-8 fed through the lossy wrapper used by
    // the HTTP body parser must not panic.
    for seed in 0..200u64 {
        let mut r = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let n = r.pick(64);
        let bytes = r.bytes(n);
        let s = String::from_utf8_lossy(&bytes);
        let _ = crate::json::parse(&s);
    }
    // The depth cap rejects deeply nested input instead of overflowing the stack.
    let deep = format!("{}{}", "[".repeat(512), "]".repeat(512));
    assert!(crate::json::parse(&deep).is_err());
}

const REGEX_ATOMS: &[&str] = &[
    "a", "b", "[a-z]", "\\s", "\\d", "\\w", "(?i:x)", "(?:a|b)", "(?=x)", "(?!x)", "*", "+", "?",
    "{2,3}", "|", "(", ")", "[", "]", "\\", "^", "$", "\\p{L}",
];

#[test]
fn fuzz_regex_compile_and_match() {
    let mut rng = Lcg(0xfeed_face_cafe_babe);
    for _ in 0..600 {
        let pattern: String = (0..rng.pick(8))
            .map(|_| REGEX_ATOMS[rng.pick(REGEX_ATOMS.len())])
            .collect();
        if let Ok(re) = crate::regex::Regex::compile(&pattern) {
            let text: String = (0..rng.pick(40))
                .map(|_| (b'a' + rng.pick(8) as u8) as char)
                .collect();
            let chars: Vec<char> = text.chars().collect();
            for pos in 0..=chars.len() {
                let _ = re.match_at(&chars, pos);
            }
            let _ = re.pre_tokenize(&text);
        }
    }
}

#[test]
fn fuzz_api_params_chat_and_tokenizer() {
    let engine =
        crate::server::Engine::new(Model::tiny(), Tokenizer::tiny(), "tiny".into(), 8, 1, true);
    let mut rng = Lcg(0x0123_4567_89ab_cdef);
    for _ in 0..800 {
        let len = rng.pick(120);
        let s: String = (0..len)
            .map(|_| JSON_ATOMS[rng.pick(JSON_ATOMS.len())] as char)
            .collect();
        let Ok(body) = crate::json::parse(&s) else {
            continue;
        };
        let _ = crate::server::parse_params(&body, &engine);
        let _ = crate::server::parse_completion_params(&body, &engine);
        if let Ok(messages) = crate::server::parse_messages(&body) {
            for thinking in [true, false] {
                let _ = apply_chat_template(&messages, true, thinking);
            }
        }
    }
    // Random Unicode-ish text through the byte-level tokenizer.
    let tok = Tokenizer::tiny();
    for _ in 0..300 {
        let s: String = (0..rng.pick(40))
            .map(|_| char::from_u32(rng.pick(0x2fff) as u32).unwrap_or('\u{fffd}'))
            .collect();
        let ids = tok.encode(&s);
        let _ = tok.decode(&ids, false);
        for id in ids {
            let _ = tok.decode(&[id], false);
        }
    }
    // Random message shapes (including missing fields) must not panic.
    for _ in 0..200 {
        let role = if rng.pick(2) == 0 {
            "user"
        } else {
            "assistant"
        };
        let content: String = (0..rng.pick(20))
            .map(|_| JSON_ATOMS[rng.pick(JSON_ATOMS.len())] as char)
            .collect();
        let messages = vec![Message {
            role: role.into(),
            content,
            ..Default::default()
        }];
        let _ = apply_chat_template(&messages, true, rng.pick(2) == 0);
    }
}

#[test]
fn fuzz_http_request_parser() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut rng = Lcg(0xdead_beef_1234_5678);
    for _ in 0..120 {
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let n = rng.pick(160);
        let mut bytes = Vec::with_capacity(n + 1);
        for _ in 0..n {
            // Bias toward HTTP syntax so we hit header/body code paths.
            bytes.push(if rng.pick(3) == 0 {
                b"\r\n:0123456789XYZ /"[rng.pick(17)]
            } else {
                rng.next() as u8
            });
        }
        bytes.push(b'\n');
        client.write_all(&bytes).unwrap();
        client.shutdown(Shutdown::Write).unwrap();
        // EOF terminates header/body reads; malformed inputs simply return Err.
        let _ = crate::http::read_request(&mut server);
    }
}
