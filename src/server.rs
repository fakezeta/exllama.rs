//! OpenAI-compatible API handlers and the inference engine.

use crate::chat::{apply_chat_template, Message};
use crate::http::{respond, Request, Streamer};
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::{json::Json, rng::Rng};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

pub struct Params {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
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
    /// `on_token` receives (chunk, is_reasoning).
    fn generate<F>(
        &self,
        prompt_ids: Vec<u32>,
        params: &Params,
        split_reasoning: bool,
        start_in_reasoning: bool,
        mut on_token: F,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool),
    {
        let prompt_tokens = prompt_ids.len();
        let max_tokens = if params.max_tokens == 0 {
            self.default_max_tokens
        } else {
            params.max_tokens
        };

        let mut model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        model.reset();
        let mut logits = model.forward(&prompt_ids);

        let mut rng = Rng::new(params.seed.unwrap_or(0));
        let mut completion: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut out_text = String::new();
        let mut reasoning_text = String::new();
        // Accumulate raw output bytes and only emit complete UTF-8 characters.
        // A multibyte char (e.g. an emoji) can span several tokens, so slicing
        // a decoded String by byte offsets would panic on a char boundary.
        let mut pending: Vec<u8> = Vec::new();
        let mut in_reasoning = split_reasoning && start_in_reasoning;
        let mut finish = String::from("length");

        let debug = std::env::var("EXLLAMA_DEBUG").is_ok();
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
            let tok = sample(&logits, params, &mut rng);
            if is_stop_token(tok, &self.tokenizer) {
                finish = "stop".into();
                break;
            }
            // Reasoning delimiters toggle which stream subsequent text goes to.
            // The markers themselves are never emitted in either stream.
            if split_reasoning && tok == TOK_THINK_START {
                in_reasoning = true;
                continue;
            }
            if split_reasoning && tok == TOK_THINK_END {
                in_reasoning = false;
                continue;
            }
            completion.push(tok);

            let target = if in_reasoning {
                &mut reasoning_text
            } else {
                &mut out_text
            };
            if !self.tokenizer.is_special(tok) {
                pending.extend_from_slice(&self.tokenizer.decode_token_bytes(tok));
            }
            let chunk = flush_utf8(&mut pending, target);

            if !in_reasoning
                && params
                    .stop
                    .iter()
                    .any(|s| !s.is_empty() && out_text.ends_with(s.as_str()))
            {
                for s in &params.stop {
                    if !s.is_empty() && out_text.ends_with(s.as_str()) {
                        out_text.truncate(out_text.len() - s.len());
                    }
                }
                finish = "stop".into();
                break;
            }
            if !chunk.is_empty() {
                on_token(&chunk, in_reasoning);
            }
            logits = model.forward(&[tok]);
        }

        // Flush any trailing complete characters left in the buffer.
        if !pending.is_empty() {
            flush_utf8(
                &mut pending,
                if in_reasoning {
                    &mut reasoning_text
                } else {
                    &mut out_text
                },
            );
        }

        Ok(GenResult {
            text: out_text,
            reasoning_content: reasoning_text,
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
        F: FnMut(&str, bool),
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
        F: FnMut(&str, bool),
    {
        let prompt_ids = self.tokenizer.encode(prompt);
        self.generate(prompt_ids, params, false, false, on_token)
    }
}

fn is_stop_token(tok: u32, tokz: &Tokenizer) -> bool {
    tokz.eos_id().map(|e| tok == e).unwrap_or(false)
}

/// Temperature + top-k + top-p sampling.
fn sample(logits: &[f32], params: &Params, rng: &mut Rng) -> u32 {
    let temp = params.temperature;
    if temp <= 0.0 {
        return argmax(logits);
    }

    let mut scores: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temp))
        .collect();

    // Top-k.
    if (1..=logits.len()).contains(&params.top_k) {
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(params.top_k);
    }

    // Softmax over the remaining.
    let max = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = scores.iter().map(|(_, s)| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in exps.iter_mut() {
        *e /= sum;
    }

    // Top-p (nucleus): keep the smallest prefix of the prob-sorted list whose
    // cumulative mass reaches top_p.
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        exps[b]
            .partial_cmp(&exps[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0f32;
    let mut keep: Vec<usize> = Vec::new();
    for &i in &order {
        keep.push(i);
        cum += exps[i];
        if cum >= params.top_p {
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

fn gen_id(prefix: &str) -> String {
    format!(
        "{}-{:013x}",
        prefix,
        now() as u64 ^ (std::process::id() as u64)
    )
}

fn parse_params(body: &Json) -> Params {
    let mut p = Params::default();
    if let Some(v) = body.get("temperature").and_then(|v| v.as_f64()) {
        p.temperature = v as f32;
    }
    if let Some(v) = body.get("top_p").and_then(|v| v.as_f64()) {
        p.top_p = v as f32;
    }
    if let Some(v) = body.get("top_k").and_then(|v| v.as_usize()) {
        p.top_k = v;
    }
    if let Some(v) = body.get("max_tokens").and_then(|v| v.as_usize()) {
        p.max_tokens = v;
    }
    if let Some(v) = body.get("max_completion_tokens").and_then(|v| v.as_usize()) {
        p.max_tokens = v;
    }
    if let Some(v) = body.get("seed").and_then(|v| v.as_i64()) {
        p.seed = Some(v as u64);
    }
    if let Some(v) = body.get("repetition_penalty").and_then(|v| v.as_f64()) {
        p.repetition_penalty = v as f32;
    }
    if let Some(v) = body.get("enable_thinking").and_then(|v| v.as_bool()) {
        p.enable_thinking = v;
    }
    p.stop = parse_stop(body.get("stop"));
    p
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

fn chat_chunk(
    id: &str,
    model: &str,
    delta_role: Option<&str>,
    content: Option<&str>,
    reasoning: Option<&str>,
    finish: Option<&str>,
) -> String {
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
    Json::Obj(vec![
        ("id".into(), Json::Str(id.into())),
        ("object".into(), Json::Str("chat.completion.chunk".into())),
        ("created".into(), Json::Num(now())),
        ("model".into(), Json::Str(model.into())),
        ("choices".into(), Json::Arr(vec![choice])),
    ])
    .to_string()
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
    let messages = match parse_messages(&body) {
        Ok(m) => m,
        Err(e) => {
            let (s, b) = e;
            respond(&mut stream, s, "application/json", &b);
            return;
        }
    };
    let params = parse_params(&body);
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
            if !chunk.is_empty() {
                let (content, reasoning) = if is_reasoning {
                    (None, Some(chunk))
                } else {
                    (Some(chunk), None)
                };
                sse.write_chunk(&format!(
                    "data: {}\n\n",
                    chat_chunk(&id, &model_name, None, content, reasoning, None)
                ));
            }
        });
        match res {
            Ok(r) => {
                sse.write_chunk(&format!(
                    "data: {}\n\n",
                    chat_chunk(
                        &id,
                        &model_name,
                        None,
                        None,
                        None,
                        Some(r.finish_reason.as_str())
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
        let res = engine.run_chat(&messages, &params, |_, _| {});
        match res {
            Ok(r) => {
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
                    (
                        "usage".into(),
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
                        ]),
                    ),
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
    let params = parse_params(&body);
    let stream_mode = want_stream(&body);
    let id = gen_id("cmpl");
    let model_name = engine.model_name.clone();

    if stream_mode {
        let mut sse = Streamer::new(stream, 200, "text/event-stream");
        let res = engine.run_completion(&prompt, &params, |chunk, _| {
            if !chunk.is_empty() {
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
                ));
            }
        });
        match res {
            Ok(r) => {
                let c = Json::Obj(vec![
                    ("text".into(), Json::Str(String::new())),
                    ("index".into(), Json::Num(0.0)),
                    ("logprobs".into(), Json::Null),
                    ("finish_reason".into(), Json::Str(r.finish_reason)),
                ]);
                sse.write_chunk(&format!(
                    "data: {}\n\n",
                    Json::Obj(vec![
                        ("id".into(), Json::Str(id)),
                        ("object".into(), Json::Str("text_completion".into())),
                        ("created".into(), Json::Num(now())),
                        ("model".into(), Json::Str(model_name)),
                        ("choices".into(), Json::Arr(vec![c])),
                    ])
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
        let res = engine.run_completion(&prompt, &params, |_, _| {});
        match res {
            Ok(r) => {
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
                    (
                        "usage".into(),
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
                        ]),
                    ),
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
}
