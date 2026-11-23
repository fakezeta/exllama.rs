//! OpenAI-compatible API handlers and the inference engine.

use crate::chat::{apply_chat_template, parse_tool_calls, tools_system_prompt, Message, ToolCall};
use crate::http::{respond, Request, Streamer};
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::{json::Json, rng::Rng};
use std::cell::RefCell;
use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
}

#[derive(Clone)]
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
    /// OpenAI `tools` array, rendered into the system prompt by `run_chat`.
    pub tools: Option<Json>,
    /// Number of completions per prompt (`n`). Generated sequentially.
    pub n: usize,
    /// OpenAI abuse-monitoring field. Accepted and validated, otherwise unused.
    pub user: Option<String>,
    pub response_format: ResponseFormat,
    /// Include per-token log probabilities in the response.
    pub logprobs: bool,
    /// Number of alternative tokens reported per position (top_logprobs for
    /// chat, `logprobs: N` for completions). Capped at MAX_TOP_LOGPROBS.
    pub top_logprobs: usize,
    /// Internal wall-clock deadline override. Never set from the API; tests
    /// use it to exercise the deadline path without waiting minutes.
    pub deadline: Option<Instant>,
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
            tools: None,
            n: 1,
            user: None,
            response_format: ResponseFormat::Text,
            logprobs: false,
            top_logprobs: 0,
            deadline: None,
        }
    }
}

/// A single generation holds the model mutex, so bound its wall-clock time
/// regardless of the requested token count.
const MAX_GENERATION_SECONDS: u64 = 300;

/// A single token's log probability (used for top_logprobs entries).
#[derive(Clone, Debug)]
pub struct Logprob {
    pub token: String,
    pub token_id: u32,
    pub logprob: f32,
}

/// Log probability of a generated token plus its most likely alternatives.
#[derive(Clone, Debug)]
pub struct TokenLogprob {
    pub token: String,
    pub token_id: u32,
    pub logprob: f32,
    pub top: Vec<Logprob>,
}

pub struct GenResult {
    pub text: String,
    /// Thinking trace emitted between `<think>` and `</think>` (Qwen3.5).
    pub reasoning_content: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    /// One entry per generated completion token (empty unless `logprobs` is on).
    /// Delimiters and stop tokens are excluded, matching `completion_tokens`.
    pub logprobs: Vec<TokenLogprob>,
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
    /// Model-size-aware default for chat requests that omit `enable_thinking`.
    pub default_enable_thinking: bool,
}

impl Engine {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        model_name: String,
        default_max_tokens: usize,
        threads: usize,
        default_enable_thinking: bool,
    ) -> Self {
        Engine {
            model: Mutex::new(model),
            tokenizer,
            model_name,
            default_max_tokens,
            threads,
            default_enable_thinking,
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
        on_token: F,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool) -> bool,
    {
        self.generate_inner(
            prompt_ids,
            params,
            split_reasoning,
            start_in_reasoning,
            on_token,
            |_| {},
        )
    }

    fn generate_inner<F, G>(
        &self,
        prompt_ids: Vec<u32>,
        params: &Params,
        split_reasoning: bool,
        start_in_reasoning: bool,
        mut on_token: F,
        mut on_logprob: G,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool) -> bool,
        G: FnMut(&TokenLogprob),
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
                logprobs: Vec::new(),
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
        let mut logprobs: Vec<TokenLogprob> = Vec::new();
        let mut router =
            OutputRouter::new(split_reasoning, start_in_reasoning, params.stop.clone());
        let mut finish = String::from("length");

        let debug = std::env::var("EXLLAMA_DEBUG").is_ok();
        const DECODE_BATCH: usize = 16;
        let decode_start = std::time::Instant::now();
        let deadline = params
            .deadline
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(MAX_GENERATION_SECONDS));
        for (step, _) in (0..max_tokens).enumerate() {
            if Instant::now() >= deadline {
                finish = "length".into();
                break;
            }
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
            // Compute logprobs from the same (penalty-adjusted) distribution
            // that `sample` used, before sampling transforms (temperature,
            // top-k, top-p) are applied — the vLLM/SGLang convention.
            let lp_info = params.logprobs.then(|| {
                logprob_info(
                    &logits,
                    params,
                    &completion,
                    tok,
                    params.top_logprobs,
                    &self.tokenizer,
                )
            });
            completion.push(tok);
            if let Some(info) = lp_info {
                logprobs.push(info);
            }

            // Decode bytes are assembled into complete UTF-8 characters, then
            // routed through the stop-string tail buffer before streaming.
            let stopped = if !self.tokenizer.is_special(tok) {
                router.push(&self.tokenizer.decode_token_bytes(tok), &mut on_token)
            } else {
                false
            };
            // Emit the logprob delta *after* this token's content delta. The
            // old order put logprobs before content, which is not how OpenAI
            // chunks a token's output position.
            if params.logprobs {
                if let Some(info) = logprobs.last() {
                    on_logprob(info);
                }
            }
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
            logprobs,
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
        self.run_chat_inner(messages, params, on_token, |_| {})
    }

    pub fn run_chat_inner<F, G>(
        &self,
        messages: &[Message],
        params: &Params,
        on_token: F,
        on_logprob: G,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool) -> bool,
        G: FnMut(&TokenLogprob),
    {
        // With `tools`, Qwen3.5 injects a tool-definition preamble into the
        // system turn (merged with any user-supplied system message).
        let mut rendered: Vec<Message> =
            apply_response_format_chat(messages, params.response_format);
        if let Some(tools) = &params.tools {
            let mut sys = tools_system_prompt(tools);
            if rendered.first().is_some_and(|m| m.role == "system") {
                let user_sys = rendered[0].content.trim();
                if !user_sys.is_empty() {
                    sys.push_str("\n\n");
                    sys.push_str(user_sys);
                }
                rendered.remove(0);
            }
            rendered.insert(
                0,
                Message {
                    role: "system".into(),
                    content: sys,
                    ..Default::default()
                },
            );
        }
        let prompt = apply_chat_template(&rendered, true, params.enable_thinking);
        let prompt_ids = self.tokenizer.encode(&prompt);
        self.generate_inner(
            prompt_ids,
            params,
            true,
            params.enable_thinking,
            on_token,
            on_logprob,
        )
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
        self.run_completion_inner(prompt, params, on_token, |_| {})
    }

    pub fn run_completion_inner<F, G>(
        &self,
        prompt: &str,
        params: &Params,
        on_token: F,
        on_logprob: G,
    ) -> Result<GenResult, String>
    where
        F: FnMut(&str, bool) -> bool,
        G: FnMut(&TokenLogprob),
    {
        let prompt = apply_response_format_text(prompt, params.response_format);
        let prompt_ids = self.tokenizer.encode(&prompt);
        self.generate_inner(prompt_ids, params, false, false, on_token, on_logprob)
    }
}

fn is_stop_token(tok: u32, tokz: &Tokenizer) -> bool {
    tokz.eos_id().map(|e| tok == e).unwrap_or(false)
}

/// Repetition penalty + temperature + top-k + top-p sampling.
/// `history` is the tokens generated so far; each distinct token in it gets
/// its logit divided by `repetition_penalty` (positive) or multiplied (negative),
/// which breaks greedy repetition loops when the penalty is > 1.
fn prepare_logits<'a>(
    logits: &'a [f32],
    params: &Params,
    history: &[u32],
    buf: &'a mut Vec<f32>,
) -> &'a [f32] {
    let rep = params.repetition_penalty;
    let need_copy = (rep > 0.0 && rep != 1.0)
        || params.presence_penalty != 0.0
        || params.frequency_penalty != 0.0
        || !params.logit_bias.is_empty();
    // Only copy the logits when some logit adjustment actually applies.
    if !need_copy {
        return logits;
    }
    buf.clear();
    buf.extend_from_slice(logits);
    {
        let penalized = &mut buf[..];
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
    }
    buf
}

/// Sample after applying logit processors (penalties and bias), sharing that
/// work with `logprob_info` so reported probabilities match the distribution
/// the token was drawn from.
fn sample(logits: &[f32], params: &Params, rng: &mut Rng, history: &[u32]) -> u32 {
    let mut buf = Vec::new();
    let logits = prepare_logits(logits, params, history, &mut buf);

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
    // Draw in f64: casting a near-1.0 f64 to f32 can round it up to exactly
    // 1.0, making `target == mass` and biasing the last candidate.
    let sum: f64 = scores.iter().map(|&(_, s)| (s - max).exp() as f64).sum();
    let mass = sum * params.top_p.clamp(0.0, 1.0) as f64;
    let target = rng.next_f64() * mass;
    let mut cum = 0.0f64;
    for &(idx, s) in &scores {
        cum += (s - max).exp() as f64;
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
    let sum: f64 = logits.iter().map(|&l| (l / temp - max).exp() as f64).sum();
    let target = rng.next_f64() * sum;
    let mut cum = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        cum += (l / temp - max).exp() as f64;
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

/// Hard cap aligned with OpenAI (`top_logprobs` <= 20, `n` <= 128).
const MAX_TOP_LOGPROBS: usize = 20;
const MAX_N: usize = 128;
const JSON_HINT: &str =
    "\n\nRespond with a single valid JSON object and nothing else. Do not use markdown code fences.";

/// JSON mode is best-effort: we strengthen the prompt but cannot constrain the
/// decoder without a grammar engine. No output validation/retry is attempted.
fn apply_response_format_text<'a>(
    prompt: &'a str,
    fmt: ResponseFormat,
) -> std::borrow::Cow<'a, str> {
    if fmt != ResponseFormat::JsonObject {
        return std::borrow::Cow::Borrowed(prompt);
    }
    let mut out = String::with_capacity(prompt.len() + JSON_HINT.len());
    out.push_str(prompt);
    out.push_str(JSON_HINT);
    std::borrow::Cow::Owned(out)
}

fn apply_response_format_chat(messages: &[Message], fmt: ResponseFormat) -> Vec<Message> {
    let mut out = messages.to_vec();
    if fmt != ResponseFormat::JsonObject {
        return out;
    }
    if let Some(last) = out.last_mut() {
        if last.role == "user" {
            last.content.push_str(JSON_HINT);
            return out;
        }
    }
    out.push(Message {
        role: "user".into(),
        content: JSON_HINT.trim_start().to_string(),
        ..Default::default()
    });
    out
}

/// Each choice gets a distinct RNG stream; with a supplied seed choice 0 keeps
/// that exact seed and later choices use increasing seeds. Greedy (temp 0)
/// requests still produce identical choices, as with OpenAI.
fn choice_seeds(params: &Params, n: usize) -> Vec<Option<u64>> {
    let base = params.seed.unwrap_or_else(default_seed);
    (0..n).map(|i| Some(base.wrapping_add(i as u64))).collect()
}

/// Log probability of `tok` plus the `top_n` most likely alternatives, taken
/// from the same processed logits used for sampling (before temperature/top-p).
fn logprob_info(
    logits: &[f32],
    params: &Params,
    history: &[u32],
    tok: u32,
    top_n: usize,
    tokz: &Tokenizer,
) -> TokenLogprob {
    let mut buf = Vec::new();
    let p = prepare_logits(logits, params, history, &mut buf);
    let (max, sum) = if p.is_empty() {
        (0.0f64, 1.0f64)
    } else {
        let max = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let max = if max.is_finite() { max } else { 0.0 };
        let sum: f64 = p.iter().map(|&v| (v as f64 - max).exp()).sum();
        (max, sum)
    };
    let lp = |v: f32| (v as f64 - max - sum.ln()) as f32;

    let mut top: Vec<(usize, f32)> = p.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let k = top_n.min(top.len());
    if k < top.len() {
        top.select_nth_unstable_by(k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    top.truncate(k);
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top = top
        .into_iter()
        .map(|(id, v)| Logprob {
            token: tokz.decode(&[id as u32], false),
            token_id: id as u32,
            logprob: lp(v),
        })
        .collect();

    let id = (tok as usize).min(p.len().saturating_sub(1));
    TokenLogprob {
        token: tokz.decode(&[tok], false),
        token_id: tok,
        logprob: p.get(id).map(|&v| lp(v)).unwrap_or(f32::NEG_INFINITY),
        top,
    }
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

pub(crate) fn parse_params(body: &Json, engine: &Engine) -> Result<Params, (u16, String)> {
    parse_params_inner(body, engine, true)
}

pub(crate) fn parse_completion_params(
    body: &Json,
    engine: &Engine,
) -> Result<Params, (u16, String)> {
    parse_params_inner(body, engine, false)
}

fn parse_params_inner(body: &Json, engine: &Engine, chat: bool) -> Result<Params, (u16, String)> {
    let bad = |msg: &str| json_error(400, msg, "invalid_request_error");
    let mut p = Params {
        enable_thinking: engine.default_enable_thinking,
        ..Default::default()
    };

    if let Some(v) = json_f32(body, "temperature")? {
        if v < 0.0 {
            return Err(bad("temperature must be >= 0"));
        }
        p.temperature = v;
    }
    if let Some(v) = json_f32(body, "top_p")? {
        if !(v > 0.0 && v <= 1.0) {
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
        if v < 0 {
            return Err(bad("seed must be a non-negative integer"));
        }
        p.seed = Some(v as u64);
    }
    if let Some(v) = json_f32(body, "repetition_penalty")? {
        // Zero used to silently disable the penalty; reject it so a typo
        // cannot quietly change decoding behavior.
        if !v.is_finite() || v <= 0.0 {
            return Err(bad("repetition_penalty must be a finite number > 0"));
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
    p.tools = body
        .get("tools")
        .and_then(|v| v.as_arr())
        .map(|a| Json::Arr(a.to_vec()));

    if let Some(v) = json_int(body, "n")? {
        if !(1..=MAX_N as i64).contains(&v) {
            return Err(bad(&format!("n must be in [1, {}]", MAX_N)));
        }
        p.n = v as usize;
    }
    match body.get("user") {
        None | Some(Json::Null) => {}
        Some(Json::Str(s)) => p.user = Some(s.clone()),
        Some(_) => return Err(bad("user must be a string")),
    }
    if let Some(v) = body.get("response_format") {
        if !v.is_null() {
            let ty = v
                .get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| bad("response_format.type must be a string"))?;
            p.response_format = match ty {
                "text" => ResponseFormat::Text,
                "json_object" => ResponseFormat::JsonObject,
                other => {
                    return Err(bad(&format!(
                        "unsupported response_format type '{}' (supported: text, json_object)",
                        other
                    )))
                }
            };
        }
    }
    if chat {
        // Chat uses `logprobs: bool` + `top_logprobs: 0..=20`.
        match body.get("logprobs") {
            None | Some(Json::Null) => {}
            Some(Json::Bool(v)) => p.logprobs = *v,
            Some(_) => return Err(bad("logprobs must be a boolean for chat completions")),
        }
        if let Some(v) = json_int(body, "top_logprobs")? {
            if !(0..=MAX_TOP_LOGPROBS as i64).contains(&v) {
                return Err(bad(&format!(
                    "top_logprobs must be in [0, {}]",
                    MAX_TOP_LOGPROBS
                )));
            }
            p.top_logprobs = v as usize;
        }
        if p.top_logprobs > 0 && !p.logprobs {
            return Err(bad("top_logprobs requires logprobs: true"));
        }
    } else {
        // Completions use `logprobs: N` (N alternatives per token).
        if let Some(v) = json_int(body, "logprobs")? {
            if !(0..=MAX_TOP_LOGPROBS as i64).contains(&v) {
                return Err(bad(&format!(
                    "logprobs must be in [0, {}]",
                    MAX_TOP_LOGPROBS
                )));
            }
            p.logprobs = true;
            p.top_logprobs = v as usize;
        }
    }
    Ok(p)
}

fn want_stream(body: &Json) -> bool {
    body.get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

pub(crate) fn parse_messages(body: &Json) -> Result<Vec<Message>, (u16, String)> {
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
        let content = match m.get("content") {
            Some(Json::Str(s)) => s.clone(),
            // Text-only content parts are accepted; vision parts are ignored.
            Some(Json::Arr(a)) => a
                .iter()
                .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        let reasoning_content = m
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let name = m.get("name").and_then(|v| v.as_str()).map(str::to_string);
        let tool_calls = m.get("tool_calls").and_then(|v| v.as_arr()).map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    let func = tc.get("function").unwrap_or(tc);
                    let name = func.get("name")?.as_str()?.to_string();
                    let arguments = match func.get("arguments") {
                        Some(Json::Str(s)) => s.clone(),
                        Some(v @ Json::Obj(_)) => v.to_string(),
                        _ => "{}".into(),
                    };
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    Some(ToolCall {
                        id,
                        name,
                        arguments,
                    })
                })
                .collect::<Vec<_>>()
        });
        out.push(Message {
            role,
            content,
            reasoning_content,
            tool_calls,
            name,
        });
    }
    Ok(out)
}

fn usage_json(r: &GenResult) -> Json {
    usage_from(std::slice::from_ref(r))
}

/// For `n > 1` OpenAI usage is the total across all choices.
fn usage_from(results: &[GenResult]) -> Json {
    let prompt: usize = results.iter().map(|r| r.prompt_tokens).sum();
    let completion: usize = results.iter().map(|r| r.completion_tokens).sum();
    Json::Obj(vec![
        ("prompt_tokens".into(), Json::Num(prompt as f64)),
        ("completion_tokens".into(), Json::Num(completion as f64)),
        (
            "total_tokens".into(),
            Json::Num((prompt + completion) as f64),
        ),
    ])
}

fn logprob_json(l: &Logprob) -> Json {
    Json::Obj(vec![
        ("token".into(), Json::Str(l.token.clone())),
        ("token_id".into(), Json::Num(l.token_id as f64)),
        (
            "logprob".into(),
            if l.logprob.is_finite() {
                Json::Num(l.logprob as f64)
            } else {
                Json::Null
            },
        ),
    ])
}

fn token_logprob_json(t: &TokenLogprob) -> Json {
    Json::Obj(vec![
        ("token".into(), Json::Str(t.token.clone())),
        ("token_id".into(), Json::Num(t.token_id as f64)),
        (
            "logprob".into(),
            if t.logprob.is_finite() {
                Json::Num(t.logprob as f64)
            } else {
                Json::Null
            },
        ),
        (
            "top_logprobs".into(),
            Json::Arr(t.top.iter().map(logprob_json).collect()),
        ),
    ])
}

/// Chat choices carry `logprobs.content`, an array aligned with the output tokens.
fn chat_logprobs_json(tokens: &[TokenLogprob]) -> Json {
    Json::Obj(vec![(
        "content".into(),
        Json::Arr(tokens.iter().map(token_logprob_json).collect()),
    )])
}

/// Completions use the parallel-array shape with byte `text_offset`s.
fn completion_logprobs_json(tokens: &[TokenLogprob]) -> Json {
    let mut texts = Vec::new();
    let mut logps = Vec::new();
    let mut tops = Vec::new();
    let mut offsets = Vec::new();
    let mut offset = 0usize;
    for t in tokens {
        texts.push(Json::Str(t.token.clone()));
        logps.push(if t.logprob.is_finite() {
            Json::Num(t.logprob as f64)
        } else {
            Json::Null
        });
        tops.push(Json::Arr(
            t.top
                .iter()
                .map(|x| {
                    if x.logprob.is_finite() {
                        Json::Num(x.logprob as f64)
                    } else {
                        Json::Null
                    }
                })
                .collect(),
        ));
        offsets.push(Json::Num(offset as f64));
        offset += t.token.len();
    }
    Json::Obj(vec![
        ("tokens".into(), Json::Arr(texts)),
        ("token_logprobs".into(), Json::Arr(logps)),
        ("top_logprobs".into(), Json::Arr(tops)),
        ("text_offset".into(), Json::Arr(offsets)),
    ])
}

/// Incrementally extracts Qwen3.5 `<tool_call>` blocks from streamed answer
/// text. Content before a block streams normally; text inside a block is
/// withheld until it closes, then reported as a parsed call so SSE can emit
/// OpenAI `tool_calls` deltas rather than raw XML.
struct ToolCallStreamer {
    pending: String,
    next_index: usize,
}

impl ToolCallStreamer {
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn new() -> Self {
        Self {
            pending: String::new(),
            next_index: 0,
        }
    }

    /// Returns (plain content to stream, completed calls as (index, name,
    /// arguments JSON text)).
    fn push(&mut self, chunk: &str) -> (String, Vec<(usize, String, String)>) {
        self.pending.push_str(chunk);
        let mut content = String::new();
        let mut calls = Vec::new();
        loop {
            let Some(open) = self.pending.find(Self::OPEN) else {
                // Hold back the longest suffix that is also a prefix of the
                // opening tag; it may complete in a later chunk. A fixed
                // hold-back would leak earlier text into the content stream.
                let max = self.pending.len().min(Self::OPEN.len() - 1);
                let mut hold = 0;
                for n in (1..=max).rev() {
                    let start = self.pending.len() - n;
                    if self.pending.is_char_boundary(start)
                        && Self::OPEN.starts_with(&self.pending[start..])
                    {
                        hold = n;
                        break;
                    }
                }
                let safe = self.pending.len() - hold;
                if safe > 0 {
                    content.push_str(&self.pending[..safe]);
                    self.pending.drain(..safe);
                }
                return (content, calls);
            };
            content.push_str(&self.pending[..open]);
            let after_open = open + Self::OPEN.len();
            let Some(close_rel) = self.pending[after_open..].find(Self::CLOSE) else {
                // Keep the opening marker and everything after it buffered.
                self.pending.drain(..open);
                return (content, calls);
            };
            let close = after_open + close_rel + Self::CLOSE.len();
            let block = self.pending[open..close].to_string();
            self.pending.drain(..close);
            if let Some(call) = crate::chat::parse_one_tool_call(&block, self.next_index) {
                calls.push((self.next_index, call.name, call.arguments));
                self.next_index += 1;
            }
        }
    }

    /// Flush any trailing text that did not form a tool call.
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn any_calls(&self) -> bool {
        self.next_index > 0
    }
}

/// One OpenAI `tool_calls` delta entry. `id`/`name` appear only in the first
/// fragment for a call; later fragments carry just `arguments`.
fn tool_call_delta(
    index: usize,
    id: Option<&str>,
    name: Option<&str>,
    arguments: Option<&str>,
) -> Json {
    let mut function: Vec<(String, Json)> = Vec::new();
    if let Some(n) = name {
        function.push(("name".into(), Json::Str(n.into())));
    }
    if let Some(a) = arguments {
        function.push(("arguments".into(), Json::Str(a.into())));
    }
    let mut call: Vec<(String, Json)> = vec![("index".into(), Json::Num(index as f64))];
    if let Some(id) = id {
        call.push(("id".into(), Json::Str(id.into())));
    }
    call.push(("type".into(), Json::Str("function".into())));
    call.push(("function".into(), Json::Obj(function)));
    Json::Arr(vec![Json::Obj(call)])
}

#[allow(clippy::too_many_arguments)]
fn chat_chunk_json(
    id: &str,
    model: &str,
    index: usize,
    delta_role: Option<&str>,
    content: Option<&str>,
    reasoning: Option<&str>,
    finish: Option<&str>,
    logprobs: Option<Json>,
    usage: Option<Json>,
    tool_calls: Option<Json>,
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
    if let Some(tc) = tool_calls {
        delta.push(("tool_calls".into(), tc));
    }
    let mut choice: Vec<(String, Json)> = vec![
        ("index".into(), Json::Num(index as f64)),
        ("delta".into(), Json::Obj(delta)),
        (
            "finish_reason".into(),
            match finish {
                Some(f) => Json::Str(f.into()),
                None => Json::Null,
            },
        ),
    ];
    if let Some(lp) = logprobs {
        choice.push(("logprobs".into(), lp));
    }
    let mut obj = vec![
        ("id".into(), Json::Str(id.into())),
        ("object".into(), Json::Str("chat.completion.chunk".into())),
        ("created".into(), Json::Num(now())),
        ("model".into(), Json::Str(model.into())),
        ("choices".into(), Json::Arr(vec![Json::Obj(choice)])),
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
    chat_chunk_json(
        id, model, 0, delta_role, content, reasoning, finish, None, None, None,
    )
    .to_string()
}

/// Non-streaming chat choice, including optional tool calls and logprobs.
fn chat_choice(r: &GenResult, index: usize, logprobs: bool) -> Json {
    let (clean_text, calls) = parse_tool_calls(&r.text);
    let finish = if calls.is_empty() {
        r.finish_reason.clone()
    } else {
        "tool_calls".to_string()
    };
    let mut message: Vec<(String, Json)> = vec![
        ("role".into(), Json::Str("assistant".into())),
        ("content".into(), Json::Str(clean_text)),
    ];
    if !r.reasoning_content.is_empty() {
        message.push((
            "reasoning_content".into(),
            Json::Str(r.reasoning_content.clone()),
        ));
    }
    if !calls.is_empty() {
        let arr: Vec<Json> = calls
            .iter()
            .map(|c| {
                Json::Obj(vec![
                    ("id".into(), Json::Str(c.id.clone())),
                    ("type".into(), Json::Str("function".into())),
                    (
                        "function".into(),
                        Json::Obj(vec![
                            ("name".into(), Json::Str(c.name.clone())),
                            ("arguments".into(), Json::Str(c.arguments.clone())),
                        ]),
                    ),
                ])
            })
            .collect();
        message.push(("tool_calls".into(), Json::Arr(arr)));
    }
    Json::Obj(vec![
        ("index".into(), Json::Num(index as f64)),
        ("message".into(), Json::Obj(message)),
        ("finish_reason".into(), Json::Str(finish)),
        (
            "logprobs".into(),
            if logprobs {
                chat_logprobs_json(&r.logprobs)
            } else {
                Json::Null
            },
        ),
    ])
}

fn completion_choice(r: &GenResult, index: usize, logprobs: bool) -> Json {
    Json::Obj(vec![
        ("text".into(), Json::Str(r.text.clone())),
        ("index".into(), Json::Num(index as f64)),
        (
            "logprobs".into(),
            if logprobs {
                completion_logprobs_json(&r.logprobs)
            } else {
                Json::Null
            },
        ),
        ("finish_reason".into(), Json::Str(r.finish_reason.clone())),
    ])
}

fn completion_chunk_json(
    id: &str,
    model: &str,
    index: usize,
    text: &str,
    logprobs: Option<Json>,
    finish: Option<&str>,
    usage: Option<Json>,
) -> Json {
    let mut choice: Vec<(String, Json)> = vec![
        ("text".into(), Json::Str(text.into())),
        ("index".into(), Json::Num(index as f64)),
        ("logprobs".into(), logprobs.unwrap_or(Json::Null)),
    ];
    if let Some(f) = finish {
        choice.push(("finish_reason".into(), Json::Str(f.into())));
    }
    let mut obj = vec![
        ("id".into(), Json::Str(id.into())),
        ("object".into(), Json::Str("text_completion".into())),
        ("created".into(), Json::Num(now())),
        ("model".into(), Json::Str(model.into())),
        ("choices".into(), Json::Arr(vec![Json::Obj(choice)])),
    ];
    if let Some(u) = usage {
        obj.push(("usage".into(), u));
    }
    Json::Obj(obj)
}

/// Final SSE chunk with `choices: []` and aggregate usage, matching the
/// OpenAI convention also used by the chat endpoint.
fn completion_usage_chunk(id: &str, model: &str, usage: Json) -> Json {
    Json::Obj(vec![
        ("id".into(), Json::Str(id.into())),
        ("object".into(), Json::Str("text_completion".into())),
        ("created".into(), Json::Num(now())),
        ("model".into(), Json::Str(model.into())),
        ("choices".into(), Json::Arr(Vec::new())),
        ("usage".into(), usage),
    ])
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
    let params = match parse_params(&body, engine) {
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
    let seeds = choice_seeds(&params, params.n);

    if stream_mode {
        let mut sse = Streamer::new(stream, 200, "text/event-stream");
        let mut results: Vec<GenResult> = Vec::with_capacity(params.n);
        // Both callbacks need to write chunks; a RefCell shares the borrow.
        let sse_cell = RefCell::new(&mut sse);
        let emit = |json: Json| {
            sse_cell
                .borrow_mut()
                .write_chunk(&format!("data: {}\n\n", json));
        };
        let mut failed = false;
        for (i, seed) in seeds.iter().enumerate() {
            let mut p = params.clone();
            p.seed = *seed;
            let mut tool_stream = ToolCallStreamer::new();
            // OpenAI sends a separate role chunk for every choice.
            sse_cell.borrow_mut().write_chunk(&format!(
                "data: {}\n\n",
                chat_chunk_json(
                    &id,
                    &model_name,
                    i,
                    Some("assistant"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None
                )
            ));
            let res = engine.run_chat_inner(
                &messages,
                &p,
                |chunk, is_reasoning| {
                    if chunk.is_empty() {
                        return true;
                    }
                    if is_reasoning {
                        emit(chat_chunk_json(
                            &id,
                            &model_name,
                            i,
                            None,
                            None,
                            Some(chunk),
                            None,
                            None,
                            None,
                            None,
                        ));
                        return true;
                    }
                    // Strip `<tool_call>` XML from the content stream and
                    // report the parsed calls as OpenAI tool-call deltas.
                    let (content, calls) = tool_stream.push(chunk);
                    if !content.is_empty() {
                        emit(chat_chunk_json(
                            &id,
                            &model_name,
                            i,
                            None,
                            Some(&content),
                            None,
                            None,
                            None,
                            None,
                            None,
                        ));
                    }
                    for (tc_index, name, arguments) in calls {
                        let call_id = format!("call_{}", tc_index);
                        emit(chat_chunk_json(
                            &id,
                            &model_name,
                            i,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            Some(tool_call_delta(tc_index, Some(&call_id), Some(&name), None)),
                        ));
                        // OpenAI streams arguments in fragments.
                        let mut rest = arguments.as_str();
                        while !rest.is_empty() {
                            let mut end = rest.len().min(64);
                            while end > 0 && !rest.is_char_boundary(end) {
                                end -= 1;
                            }
                            emit(chat_chunk_json(
                                &id,
                                &model_name,
                                i,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                Some(tool_call_delta(tc_index, None, None, Some(&rest[..end]))),
                            ));
                            rest = &rest[end..];
                        }
                    }
                    true
                },
                |lp| {
                    if p.logprobs {
                        sse_cell.borrow_mut().write_chunk(&format!(
                            "data: {}\n\n",
                            chat_chunk_json(
                                &id,
                                &model_name,
                                i,
                                None,
                                None,
                                None,
                                None,
                                Some(chat_logprobs_json(std::slice::from_ref(lp))),
                                None,
                                None
                            )
                        ));
                    }
                },
            );
            match res {
                Ok(r) => {
                    // Flush any trailing plain text the streamer held back
                    // while waiting for a possible tool-call marker.
                    let tail = tool_stream.finish();
                    if !tail.is_empty() {
                        emit(chat_chunk_json(
                            &id,
                            &model_name,
                            i,
                            None,
                            Some(&tail),
                            None,
                            None,
                            None,
                            None,
                            None,
                        ));
                    }
                    let finish =
                        if tool_stream.any_calls() || !parse_tool_calls(&r.text).1.is_empty() {
                            "tool_calls"
                        } else {
                            r.finish_reason.as_str()
                        };
                    emit(chat_chunk_json(
                        &id,
                        &model_name,
                        i,
                        None,
                        None,
                        None,
                        Some(finish),
                        None,
                        None,
                        None,
                    ));
                    results.push(r);
                }
                Err(e) => {
                    sse_cell.borrow_mut().write_chunk(&format!(
                        "data: {}\n\n",
                        json_error(500, &e, "server_error").1
                    ));
                    failed = true;
                    break;
                }
            }
        }
        let sse = sse_cell.into_inner();
        if failed {
            sse.write_chunk("data: [DONE]\n\n");
            sse.end();
            return;
        }
        // Final chunk carries aggregate usage (OpenAI uses choices: []).
        let mut final_chunk: Vec<(String, Json)> = vec![
            ("id".into(), Json::Str(id.clone())),
            ("object".into(), Json::Str("chat.completion.chunk".into())),
            ("created".into(), Json::Num(now())),
            ("model".into(), Json::Str(model_name.clone())),
            ("choices".into(), Json::Arr(Vec::new())),
        ];
        if params.include_usage {
            final_chunk.push(("usage".into(), usage_from(&results)));
        }
        sse.write_chunk(&format!("data: {}\n\n", Json::Obj(final_chunk)));
        sse.write_chunk("data: [DONE]\n\n");
        sse.end();
    } else {
        let mut results: Vec<GenResult> = Vec::with_capacity(params.n);
        for seed in &seeds {
            let mut p = params.clone();
            p.seed = *seed;
            match engine.run_chat(&messages, &p, |_, _| true) {
                Ok(r) => results.push(r),
                Err(e) => {
                    let (s, b) = json_error(500, &e, "server_error");
                    respond(&mut stream, s, "application/json", &b);
                    return;
                }
            }
        }
        let choices: Vec<Json> = results
            .iter()
            .enumerate()
            .map(|(i, r)| chat_choice(r, i, params.logprobs))
            .collect();
        let resp = Json::Obj(vec![
            ("id".into(), Json::Str(id)),
            ("object".into(), Json::Str("chat.completion".into())),
            ("created".into(), Json::Num(now())),
            ("model".into(), Json::Str(model_name)),
            ("choices".into(), Json::Arr(choices)),
            ("usage".into(), usage_from(&results)),
        ]);
        respond(&mut stream, 200, "application/json", &resp.to_string());
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
    let params = match parse_completion_params(&body, engine) {
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
    let seeds = choice_seeds(&params, params.n);

    if stream_mode {
        let mut sse = Streamer::new(stream, 200, "text/event-stream");
        let mut results: Vec<GenResult> = Vec::with_capacity(params.n);
        let sse_cell = RefCell::new(&mut sse);
        let mut failed = false;
        for (i, seed) in seeds.iter().enumerate() {
            let mut p = params.clone();
            p.seed = *seed;
            let res = engine.run_completion_inner(
                &prompt,
                &p,
                |chunk, _| {
                    if chunk.is_empty() {
                        return true;
                    }
                    sse_cell.borrow_mut().write_chunk(&format!(
                        "data: {}\n\n",
                        completion_chunk_json(&id, &model_name, i, chunk, None, None, None)
                    ))
                },
                |lp| {
                    if p.logprobs {
                        sse_cell.borrow_mut().write_chunk(&format!(
                            "data: {}\n\n",
                            completion_chunk_json(
                                &id,
                                &model_name,
                                i,
                                "",
                                Some(completion_logprobs_json(std::slice::from_ref(lp))),
                                None,
                                None
                            )
                        ));
                    }
                },
            );
            match res {
                Ok(r) => {
                    let finish = r.finish_reason.clone();
                    results.push(r);
                    sse_cell.borrow_mut().write_chunk(&format!(
                        "data: {}\n\n",
                        completion_chunk_json(&id, &model_name, i, "", None, Some(&finish), None)
                    ));
                }
                Err(e) => {
                    sse_cell.borrow_mut().write_chunk(&format!(
                        "data: {}\n\n",
                        json_error(500, &e, "server_error").1
                    ));
                    failed = true;
                    break;
                }
            }
        }
        let sse = sse_cell.into_inner();
        if failed {
            sse.write_chunk("data: [DONE]\n\n");
            sse.end();
            return;
        }
        // Like chat, usage goes in its own final chunk with no choices.
        if params.include_usage {
            sse.write_chunk(&format!(
                "data: {}\n\n",
                completion_usage_chunk(&id, &model_name, usage_from(&results))
            ));
        }
        sse.write_chunk("data: [DONE]\n\n");
        sse.end();
    } else {
        let mut results: Vec<GenResult> = Vec::with_capacity(params.n);
        for seed in &seeds {
            let mut p = params.clone();
            p.seed = *seed;
            match engine.run_completion(&prompt, &p, |_, _| true) {
                Ok(r) => results.push(r),
                Err(e) => {
                    let (s, b) = json_error(500, &e, "server_error");
                    respond(&mut stream, s, "application/json", &b);
                    return;
                }
            }
        }
        let choices: Vec<Json> = results
            .iter()
            .enumerate()
            .map(|(i, r)| completion_choice(r, i, params.logprobs))
            .collect();
        let resp = Json::Obj(vec![
            ("id".into(), Json::Str(id)),
            ("object".into(), Json::Str("text_completion".into())),
            ("created".into(), Json::Num(now())),
            ("model".into(), Json::Str(model_name)),
            ("choices".into(), Json::Arr(choices)),
            ("usage".into(), usage_from(&results)),
        ]);
        respond(&mut stream, 200, "application/json", &resp.to_string());
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

    fn tiny_engine() -> Engine {
        Engine::new(
            crate::model::Model::tiny(),
            Tokenizer::tiny(),
            "tiny-model".into(),
            64,
            1,
            true,
        )
    }

    fn start_server() -> SocketAddr {
        let engine = Arc::new(tiny_engine());
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

    /// Extract the JSON payloads from a chunked SSE response body, in order.
    fn sse_chunks(b: &str) -> Vec<Json> {
        let mut out = Vec::new();
        let mut rest = b;
        while let Some(pos) = rest.find("data: ") {
            rest = &rest[pos + "data: ".len()..];
            if rest.starts_with("[DONE]") {
                break;
            }
            let Some(end) = rest.find("\n\n") else {
                break;
            };
            if let Ok(j) = crate::json::parse(&rest[..end]) {
                out.push(j);
            }
            rest = &rest[end + 2..];
        }
        out
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
            true,
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
            ..Default::default()
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
    fn generation_deadline_stops_before_first_token() {
        let engine = tiny_engine();
        let params = Params {
            temperature: 0.0,
            max_tokens: 64,
            enable_thinking: false,
            deadline: Some(Instant::now()), // already expired
            ..Default::default()
        };
        let messages = vec![Message {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        let res = engine.run_chat(&messages, &params, |_, _| true).unwrap();
        assert_eq!(res.completion_tokens, 0);
        assert_eq!(res.finish_reason, "length");
        assert!(res.logprobs.is_empty());
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
            r#"{"top_p": 0}"#,
            r#"{"seed": -1}"#,
            r#"{"repetition_penalty": -1}"#,
            r#"{"repetition_penalty": 0}"#,
        ] {
            let j = crate::json::parse(body).unwrap();
            assert!(parse_params(&j, &tiny_engine()).is_err(), "body={}", body);
        }
        let j = crate::json::parse(r#"{"temperature":0.7,"top_p":0.9,"max_tokens":128}"#).unwrap();
        let p = parse_params(&j, &tiny_engine()).unwrap();
        assert_eq!(p.temperature, 0.7);
        assert_eq!(p.top_p, 0.9);
        assert_eq!(p.max_tokens, 128);

        // max_completion_tokens overrides max_tokens.
        let j = crate::json::parse(r#"{"max_tokens":0,"max_completion_tokens":64}"#).unwrap();
        assert_eq!(parse_params(&j, &tiny_engine()).unwrap().max_tokens, 64);
    }

    #[test]
    fn parse_new_api_fields() {
        let engine = tiny_engine();
        let j = crate::json::parse(
            r#"{"n":3,"user":"u1","response_format":{"type":"json_object"},"logprobs":true,"top_logprobs":4}"#,
        )
        .unwrap();
        let p = parse_params(&j, &engine).unwrap();
        assert_eq!(p.n, 3);
        assert_eq!(p.user.as_deref(), Some("u1"));
        assert_eq!(p.response_format, ResponseFormat::JsonObject);
        assert!(p.logprobs);
        assert_eq!(p.top_logprobs, 4);

        // Completions use `logprobs: N` rather than a boolean.
        let j = crate::json::parse(r#"{"logprobs":2,"n":2}"#).unwrap();
        let p = parse_completion_params(&j, &engine).unwrap();
        assert!(p.logprobs);
        assert_eq!(p.top_logprobs, 2);
        assert_eq!(p.n, 2);

        for body in [
            r#"{"n":0}"#,
            r#"{"n":129}"#,
            r#"{"user":5}"#,
            r#"{"response_format":{"type":"json_schema"}}"#,
            r#"{"response_format":{}}"#,
            r#"{"logprobs":"yes"}"#,
            r#"{"top_logprobs":21,"logprobs":true}"#,
            r#"{"top_logprobs":1}"#, // requires logprobs: true
        ] {
            let j = crate::json::parse(body).unwrap();
            assert!(parse_params(&j, &engine).is_err(), "body={}", body);
        }
        let j = crate::json::parse(r#"{"logprobs":21}"#).unwrap();
        assert!(parse_completion_params(&j, &engine).is_err());
        // `null` is treated as "not provided" everywhere.
        let j =
            crate::json::parse(r#"{"n":null,"user":null,"logprobs":null,"response_format":null}"#)
                .unwrap();
        let p = parse_params(&j, &engine).unwrap();
        assert_eq!(p.n, 1);
        assert!(p.user.is_none());
        assert!(!p.logprobs);
    }

    #[test]
    fn logprob_info_matches_log_softmax_and_applies_penalties() {
        let tokz = Tokenizer::tiny();
        let p = Params::default();
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let info = logprob_info(&logits, &p, &[], 3, 2, &tokz);
        let max = 4.0f64;
        let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
        let expected = (4.0f64 - max) - sum.ln();
        assert!((info.logprob as f64 - expected).abs() < 1e-5);
        assert_eq!(info.token_id, 3);
        assert_eq!(info.top.len(), 2);
        assert_eq!(info.top[0].token_id, 3); // highest logit first
        assert!(info.top[0].logprob >= info.top[1].logprob);

        // top_n is capped by the vocabulary size.
        assert_eq!(logprob_info(&logits, &p, &[], 0, 99, &tokz).top.len(), 4);

        // Repetition penalties are applied before logprobs are computed.
        let mut penalized = Params {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        penalized.temperature = 0.0;
        let before = logprob_info(&logits, &p, &[], 3, 0, &tokz).logprob;
        let after = logprob_info(&logits, &penalized, &[3], 3, 0, &tokz).logprob;
        assert!(after < before, "before={} after={}", before, after);
    }

    #[test]
    fn choice_seeds_and_response_format_hints() {
        let p = Params {
            seed: Some(123),
            ..Default::default()
        };
        assert_eq!(choice_seeds(&p, 3), vec![Some(123), Some(124), Some(125)]);

        assert_eq!(apply_response_format_text("hi", ResponseFormat::Text), "hi");
        let hinted = apply_response_format_text("hi", ResponseFormat::JsonObject);
        assert!(hinted.starts_with("hi"));
        assert!(hinted.contains("valid JSON object"));

        let msgs = vec![Message {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        let out = apply_response_format_chat(&msgs, ResponseFormat::JsonObject);
        assert_eq!(out.len(), 1);
        assert!(out[0].content.contains("valid JSON object"));
        // A trailing assistant turn gets a fresh user turn with the hint.
        let msgs = vec![Message {
            role: "assistant".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        let out = apply_response_format_chat(&msgs, ResponseFormat::JsonObject);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn http_chat_n_choices_and_logprobs() {
        let addr = start_server();
        let payload = r#"{"model":"tiny-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"enable_thinking":false,"n":2,"logprobs":true,"top_logprobs":1,"seed":7}"#;
        let r = post(addr, "/v1/chat/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        let j = crate::json::parse(body(&r)).unwrap();
        let choices = j.get("choices").unwrap().as_arr().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].get("index").unwrap().as_f64(), Some(0.0));
        assert_eq!(choices[1].get("index").unwrap().as_f64(), Some(1.0));
        let usage = j.get("usage").unwrap();
        let total = usage.get("total_tokens").unwrap().as_f64().unwrap() as usize;
        let mut reported = 0;
        for c in choices {
            let lp = c
                .get("logprobs")
                .unwrap()
                .get("content")
                .unwrap()
                .as_arr()
                .unwrap();
            assert!(!lp.is_empty());
            assert_eq!(
                lp[0].get("top_logprobs").unwrap().as_arr().unwrap().len(),
                1
            );
            reported += lp.len();
        }
        // Every emitted completion token has one logprob entry per choice.
        assert_eq!(
            total,
            usage.get("prompt_tokens").unwrap().as_f64().unwrap() as usize + reported
        );
    }

    #[test]
    fn http_completions_logprobs_array_shape() {
        let addr = start_server();
        let payload = r#"{"model":"tiny-model","prompt":"hi","max_tokens":4,"temperature":0,"logprobs":2,"n":2}"#;
        let r = post(addr, "/v1/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        let j = crate::json::parse(body(&r)).unwrap();
        let choices = j.get("choices").unwrap().as_arr().unwrap();
        assert_eq!(choices.len(), 2);
        let lp = choices[0].get("logprobs").unwrap();
        let texts = lp.get("tokens").unwrap().as_arr().unwrap();
        let values = lp.get("token_logprobs").unwrap().as_arr().unwrap();
        let offsets = lp.get("text_offset").unwrap().as_arr().unwrap();
        assert_eq!(texts.len(), values.len());
        assert_eq!(texts.len(), offsets.len());
        assert_eq!(
            lp.get("top_logprobs").unwrap().as_arr().unwrap()[0]
                .as_arr()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn http_stream_emits_logprob_deltas() {
        let addr = start_server();
        let payload = r#"{"model":"tiny-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0,"enable_thinking":false,"stream":true,"logprobs":true,"top_logprobs":1}"#;
        let r = post(addr, "/v1/chat/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        let b = body(&r);
        assert!(b.contains("\"top_logprobs\""));
        assert!(b.contains("\"logprob\""));
        assert!(b.contains("data: [DONE]"));
    }

    #[test]
    fn http_stream_logprob_delta_follows_content_delta() {
        let addr = start_server();
        let payload = r#"{"model":"tiny-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0,"enable_thinking":false,"stream":true,"logprobs":true,"top_logprobs":0}"#;
        let r = post(addr, "/v1/chat/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        let mut content_at = None;
        let mut logprob_at = None;
        for (i, chunk) in sse_chunks(body(&r)).iter().enumerate() {
            let Some(choice) = chunk
                .get("choices")
                .and_then(|v| v.as_arr())
                .and_then(|a| a.first())
            else {
                continue;
            };
            if content_at.is_none() && choice.get("delta").and_then(|d| d.get("content")).is_some()
            {
                content_at = Some(i);
            }
            if logprob_at.is_none() && choice.get("logprobs").is_some() {
                logprob_at = Some(i);
            }
        }
        let content_at = content_at.expect("no content delta in stream");
        let logprob_at = logprob_at.expect("no logprob delta in stream");
        assert!(
            content_at < logprob_at,
            "content delta {} should precede logprob delta {}",
            content_at,
            logprob_at
        );
    }

    #[test]
    fn http_completions_stream_usage_is_final_empty_choices_chunk() {
        let addr = start_server();
        let payload = r#"{"model":"tiny-model","prompt":"hi","max_tokens":4,"temperature":0,"stream":true,"stream_options":{"include_usage":true},"logprobs":1}"#;
        let r = post(addr, "/v1/completions", payload);
        assert_eq!(status(&r), 200, "{}", r);
        let chunks = sse_chunks(body(&r));
        assert!(chunks.len() > 1);
        // No earlier chunk carries usage.
        for c in &chunks[..chunks.len() - 1] {
            assert!(c.get("usage").is_none(), "early usage chunk: {}", c);
        }
        let last = chunks.last().unwrap();
        assert!(last
            .get("choices")
            .and_then(|v| v.as_arr())
            .is_some_and(|a| a.is_empty()));
        assert!(last
            .get("usage")
            .and_then(|u| u.get("total_tokens"))
            .is_some());
    }

    #[test]
    fn tool_call_streamer_extracts_blocks_across_chunks() {
        let mut s = ToolCallStreamer::new();
        let (c1, calls1) = s.push("Before <tool_");
        assert_eq!(c1, "Before "); // partial marker held back
        assert!(calls1.is_empty());

        let (c2, calls2) = s.push(
            "call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\nAfter",
        );
        assert_eq!(c2, "\nAfter");
        assert_eq!(calls2.len(), 1);
        assert_eq!(calls2[0].0, 0);
        assert_eq!(calls2[0].1, "get_weather");
        assert_eq!(calls2[0].2, r#"{"city":"Paris"}"#);
        assert!(s.any_calls());
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn tool_call_delta_has_openai_shape() {
        let first = tool_call_delta(2, Some("call_2"), Some("f"), None);
        let arr = first.as_arr().unwrap();
        assert_eq!(arr.len(), 1);
        let call = &arr[0];
        assert_eq!(call.get("index").unwrap().as_f64(), Some(2.0));
        assert_eq!(call.get("id").unwrap().as_str(), Some("call_2"));
        assert_eq!(call.get("type").unwrap().as_str(), Some("function"));
        let f = call.get("function").unwrap();
        assert_eq!(f.get("name").unwrap().as_str(), Some("f"));
        assert!(f.get("arguments").is_none());

        let second = tool_call_delta(2, None, None, Some("{\"x\":1}"));
        let call = &second.as_arr().unwrap()[0];
        assert!(call.get("id").is_none());
        assert_eq!(
            call.get("function")
                .unwrap()
                .get("arguments")
                .unwrap()
                .as_str(),
            Some("{\"x\":1}")
        );

        // The delta is placed inside the choice's `delta` object.
        let chunk = chat_chunk_json(
            "id",
            "model",
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(first),
        );
        assert!(chunk
            .get("choices")
            .and_then(|v| v.as_arr())
            .and_then(|a| a.first())
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("tool_calls"))
            .is_some());
    }

    #[test]
    fn parse_messages_keeps_reasoning_names_and_tool_calls() {
        let body = crate::json::parse(
            r#"{"messages":[{"role":"assistant","content":"answer","reasoning_content":"why","name":"a","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":{"x":1}}}]}]}"#,
        )
        .unwrap();
        let msgs = parse_messages(&body).unwrap();
        assert_eq!(msgs[0].reasoning_content.as_deref(), Some("why"));
        assert_eq!(msgs[0].name.as_deref(), Some("a"));
        let calls = msgs[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, r#"{"x":1}"#);

        let parts = crate::json::parse(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
        )
        .unwrap();
        assert_eq!(parse_messages(&parts).unwrap()[0].content, "hi");

        let with_tools = crate::json::parse(r#"{"tools":[{"type":"function"}]}"#).unwrap();
        assert!(parse_params(&with_tools, &tiny_engine())
            .unwrap()
            .tools
            .is_some());
    }

    #[test]
    fn parse_params_respects_model_thinking_default() {
        let small = Engine::new(
            crate::model::Model::tiny(),
            Tokenizer::tiny(),
            "small".into(),
            64,
            1,
            false,
        );
        let empty = crate::json::parse("{}").unwrap();
        assert!(!parse_params(&empty, &small).unwrap().enable_thinking);
        let explicit = crate::json::parse(r#"{"enable_thinking":true}"#).unwrap();
        assert!(parse_params(&explicit, &small).unwrap().enable_thinking);
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
        let engine = Engine::new(model, tokenizer, "bench".into(), 32, 4, true);
        let params = Params {
            temperature: 0.0,
            max_tokens: 32,
            enable_thinking: false,
            ..Default::default()
        };
        let messages = vec![Message {
            role: "user".into(),
            content: "Count: 1, 2, 3, 4,".into(),
            ..Default::default()
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
