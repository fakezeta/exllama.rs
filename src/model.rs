//! Qwen3.5 text decoder: config, weights, KV cache, and forward pass.

use std::collections::HashMap;
use std::sync::Arc;

use crate::exl3::{f32_to_half, half_f32, matmul_f16, Exl3Model, QuantLinear};
use crate::json::Json;
use crate::tensor::{dot, matmul_t, F16Matrix, Matrix};

pub struct ModelConfig {
    pub architecture: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub full_attention_interval: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub tie_word_embeddings: bool,
    /// From config.json / text_config (`eos_token_id`), when present.
    pub eos_token_id: Option<u32>,
    /// Explicit `enable_thinking` from config, when present.
    pub enable_thinking: Option<bool>,
}

impl ModelConfig {
    /// Qwen3.5 0.8B / 2B default to thinking off; 4B / 9B default to on.
    /// Hidden size is a reliable proxy: 2048 for 2B, 4096 for 9B.
    pub fn default_enable_thinking(&self) -> bool {
        self.enable_thinking.unwrap_or(self.hidden_size >= 2560)
    }
}

/// State for one full-attention layer: fp16 KV cache. Storing the code bits
/// directly halves context RAM; each step decodes the used prefix into f32
/// through a lookup table before computing attention.
#[derive(Clone)]
struct FullAttnState {
    k: Vec<u16>, // (max_ctx, num_kv_heads, head_dim), fp16 bits
    v: Vec<u16>,
}

/// State for one linear-attention layer: convolution buffer + recurrent state.
#[derive(Clone)]
struct LinearAttnState {
    /// Circular buffer of the last `kernel` qkv vectors (oldest at index 0).
    conv: Vec<f32>, // (kernel, conv_dim)
    conv_pos: usize,
    /// Recurrent state S, one (value_dim x key_dim) matrix per value head.
    state: Vec<f32>, // (num_value_heads, value_dim, key_dim)
}

/// Snapshot of the per-sequence state after prefill. Only the *used* prefix
/// of each KV cache is copied, so generating `n` choices from one prefill
/// costs O(1) snapshot memory instead of O(n) full context-sized caches.
pub(crate) struct ModelState {
    position: usize,
    full: Vec<(Vec<u16>, Vec<u16>)>,
    linear: Vec<(Vec<f32>, usize, Vec<f32>)>,
}

/// Shape of one GQA attention call, grouped to keep the helper signature small.
#[derive(Clone, Copy)]
struct AttnShape {
    nh: usize,
    nkv: usize,
    hd: usize,
    tlen: usize,
}

/// Scratch buffers reused across heads and tokens by the DeltaNet step, so
/// the recurrent path stays allocation-free per head.
struct DeltaNetScratch {
    kd: usize,
    vd: usize,
    qh: Vec<f32>,
    kh: Vec<f32>,
    mem: Vec<f32>,
    delta: Vec<f32>,
    o: Vec<f32>,
}

/// Immutable model data shared by every inference instance. Weights, config
/// and precomputed tables are loaded once; only the runtime state below is
/// per-request, so extra instances cost KV/recurrent state, not another copy
/// of a multi-GB model.
struct ModelCore {
    config: ModelConfig,
    prefix: String,
    weights: HashMap<String, Matrix>,
    /// fp16 embedding tables; one row decoded per token (also used by lm_head
    /// for tied checkpoints).
    embeddings_f16: HashMap<String, F16Matrix>,
    linears: HashMap<String, QuantLinear>,
    vectors: HashMap<String, Vec<f32>>,
    threads: usize,
    context_size: usize,
    /// Precomputed RoPE phases: `rope_cos[position * rope_half + d]`.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    rope_half: usize,
    /// Zeroed templates used to allocate fresh instance state.
    full_template: Vec<FullAttnState>,
    linear_template: Vec<LinearAttnState>,
}

/// One inference instance: shared immutable core + mutable per-sequence state.
pub struct Model {
    core: Arc<ModelCore>,
    position: usize,
    full_states: Vec<FullAttnState>,
    linear_states: Vec<LinearAttnState>,
}

fn cfg_num(j: &Json, key: &str, default: usize) -> usize {
    j.get(key).and_then(|v| v.as_usize()).unwrap_or(default)
}
fn cfg_f32(j: &Json, key: &str, default: f32) -> f32 {
    j.get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

impl Model {
    pub fn load(
        path: &str,
        context_size: usize,
        threads: usize,
        weight_mode: &str,
    ) -> Result<Model, String> {
        if context_size == 0 {
            return Err("context_size must be > 0".into());
        }
        let ex = Exl3Model::load(path, weight_mode)?;
        let cfg = &ex.config;

        let arch = cfg
            .get("architectures")
            .and_then(|v| v.as_arr())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("Qwen3_5ForConditionalGeneration")
            .to_string();

        let tc = cfg.get("text_config").unwrap_or(cfg);

        let layer_types: Vec<String> = tc
            .get("layer_types")
            .and_then(|v| v.as_arr())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let num_layers = cfg_num(tc, "num_hidden_layers", 32);
        let rope = tc.get("rope_parameters").unwrap_or(tc);

        // Use the prefix detected from the shards by `Exl3Model::load`.
        // Re-detecting it from `weights` used to break untied multimodal
        // models: their embedding lives in `embeddings_f16`, so the
        // `model.language_model.embed_tokens.weight` key was absent and the
        // prefix silently fell back to `model.`, causing a "missing embedding
        // tensor" error at the first token of every request (v0.4.1 bug).
        let prefix = ex.prefix.clone();

        let config = ModelConfig {
            architecture: arch,
            num_layers,
            hidden_size: cfg_num(tc, "hidden_size", 4096),
            vocab_size: cfg_num(tc, "vocab_size", 248320),
            num_attention_heads: cfg_num(tc, "num_attention_heads", 16),
            num_key_value_heads: cfg_num(tc, "num_key_value_heads", 4),
            head_dim: cfg_num(tc, "head_dim", 256),
            intermediate_size: cfg_num(tc, "intermediate_size", 12288),
            layer_types,
            full_attention_interval: cfg_num(tc, "full_attention_interval", 4),
            linear_conv_kernel_dim: cfg_num(tc, "linear_conv_kernel_dim", 4),
            linear_key_head_dim: cfg_num(tc, "linear_key_head_dim", 128),
            linear_num_key_heads: cfg_num(tc, "linear_num_key_heads", 16),
            linear_num_value_heads: cfg_num(tc, "linear_num_value_heads", 32),
            linear_value_head_dim: cfg_num(tc, "linear_value_head_dim", 128),
            rms_norm_eps: cfg_f32(tc, "rms_norm_eps", 1e-6),
            rope_theta: cfg_f32(rope, "rope_theta", 1e7),
            partial_rotary_factor: cfg_f32(rope, "partial_rotary_factor", 0.25),
            tie_word_embeddings: cfg
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            eos_token_id: tc
                .get("eos_token_id")
                .or_else(|| cfg.get("eos_token_id"))
                .and_then(|v| {
                    v.as_usize().map(|x| x as u32).or_else(|| {
                        v.as_arr()
                            .and_then(|a| a.first())
                            .and_then(|x| x.as_usize())
                            .map(|x| x as u32)
                    })
                }),
            enable_thinking: tc
                .get("enable_thinking")
                .or_else(|| cfg.get("enable_thinking"))
                .and_then(|v| v.as_bool()),
        };
        if config.linear_conv_kernel_dim < 2 {
            return Err(format!(
                "invalid linear_conv_kernel_dim={} (must be >= 2)",
                config.linear_conv_kernel_dim
            ));
        }
        if config.num_key_value_heads == 0 || config.num_attention_heads == 0 {
            return Err("attention head counts must be > 0".into());
        }

        // RoPE phases depend only on (position, rotary dimension), so compute
        // them once instead of doing powf/sin_cos per head per token.
        let rotary = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
        let rope_half = rotary / 2;
        let mut rope_cos = vec![0.0f32; context_size * rope_half];
        let mut rope_sin = vec![0.0f32; context_size * rope_half];
        for pos in 0..context_size {
            for d in 0..rope_half {
                let freq = 1.0 / config.rope_theta.powf(2.0 * d as f32 / rotary as f32);
                let (s, c) = (pos as f32 * freq).sin_cos();
                rope_cos[pos * rope_half + d] = c;
                rope_sin[pos * rope_half + d] = s;
            }
        }

        // Allocate per-sequence state.
        let mut full_states = Vec::new();
        let mut linear_states = Vec::new();
        for i in 0..config.num_layers {
            let is_full = if i < config.layer_types.len() {
                config.layer_types[i] == "full_attention"
            } else {
                (i + 1) % config.full_attention_interval == 0
            };
            if is_full {
                let kv = config.num_key_value_heads * config.head_dim;
                full_states.push(FullAttnState {
                    k: vec![0u16; context_size * kv],
                    v: vec![0u16; context_size * kv],
                });
            } else {
                let conv_dim = config.linear_num_value_heads * config.linear_value_head_dim
                    + 2 * config.linear_num_key_heads * config.linear_key_head_dim;
                linear_states.push(LinearAttnState {
                    // The conv state stores the last (kernel - 1) qkv vectors;
                    // the current token supplies the kernel-th value at runtime.
                    conv: vec![0.0; (config.linear_conv_kernel_dim - 1) * conv_dim],
                    conv_pos: 0,
                    state: vec![
                        0.0;
                        config.linear_num_value_heads
                            * config.linear_value_head_dim
                            * config.linear_key_head_dim
                    ],
                });
            }
        }

        let core = Arc::new(ModelCore {
            config,
            prefix,
            weights: ex.weights,
            embeddings_f16: ex.embeddings_f16,
            linears: ex.linears,
            vectors: ex.vectors,
            threads,
            context_size,
            rope_cos,
            rope_sin,
            rope_half,
            full_template: full_states.clone(),
            linear_template: linear_states.clone(),
        });
        Ok(Model {
            core,
            position: 0,
            full_states,
            linear_states,
        })
    }

    /// Create another inference instance sharing the loaded weights/config but
    /// with fresh, zeroed KV and recurrent state.
    pub fn spawn(&self) -> Model {
        Model {
            core: Arc::clone(&self.core),
            position: 0,
            full_states: self.core.full_template.clone(),
            linear_states: self.core.linear_template.clone(),
        }
    }

    #[cfg(test)]
    fn core_mut(&mut self) -> &mut ModelCore {
        Arc::get_mut(&mut self.core).expect("test model core must be uniquely owned")
    }

    pub fn config(&self) -> &ModelConfig {
        &self.core.config
    }

    pub fn context_size(&self) -> usize {
        self.core.context_size
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for s in &mut self.full_states {
            s.k.fill(0);
            s.v.fill(0);
        }
        for s in &mut self.linear_states {
            s.conv.fill(0.0);
            s.conv_pos = 0;
            s.state.fill(0.0);
        }
    }

    /// Copy the state produced by a prefill. Only KV rows up to `position`
    /// are retained; the unused tail of the cache is irrelevant to decoding.
    pub(crate) fn snapshot(&self) -> ModelState {
        let kv = self.core.config.num_key_value_heads * self.core.config.head_dim;
        let used = self.position * kv;
        ModelState {
            position: self.position,
            full: self
                .full_states
                .iter()
                .map(|s| (s.k[..used].to_vec(), s.v[..used].to_vec()))
                .collect(),
            linear: self
                .linear_states
                .iter()
                .map(|s| (s.conv.clone(), s.conv_pos, s.state.clone()))
                .collect(),
        }
    }

    /// Restore a post-prefill snapshot into this model, so a new choice can
    /// decode from the same state without re-running the prompt.
    pub(crate) fn restore(&mut self, state: &ModelState) {
        debug_assert_eq!(self.full_states.len(), state.full.len());
        debug_assert_eq!(self.linear_states.len(), state.linear.len());
        self.position = state.position;
        let kv = self.core.config.num_key_value_heads * self.core.config.head_dim;
        let used = state.position * kv;
        for (dst, (k, v)) in self.full_states.iter_mut().zip(&state.full) {
            dst.k[..used].copy_from_slice(k);
            dst.v[..used].copy_from_slice(v);
        }
        for (dst, (conv, conv_pos, recurrent)) in self.linear_states.iter_mut().zip(&state.linear) {
            dst.conv.copy_from_slice(conv);
            dst.conv_pos = *conv_pos;
            dst.state.copy_from_slice(recurrent);
        }
    }

    /// Return a weight matrix by unprefixed name, trying the language-model
    /// prefix first and then the bare name (e.g. top-level `lm_head.weight` in
    /// multimodal checkpoints).
    fn w(&self, name: &str) -> Option<&Matrix> {
        let prefixed = format!("{}{}", self.core.prefix, name);
        let with_weight = format!("{}.weight", name);
        let prefixed_weight = format!("{}{}", self.core.prefix, with_weight);
        self.core
            .weights
            .get(&prefixed)
            .or_else(|| self.core.weights.get(&prefixed_weight))
            .or_else(|| self.core.weights.get(name))
            .or_else(|| self.core.weights.get(&with_weight))
            .or_else(|| {
                // EXL3 checkpoints store the quantized lm_head under the bare
                // base name `lm_head` (no `.weight` suffix). Some models tie
                // embeddings and have no separate lm_head at all — fall back
                // to the embedding matrix in that case.
                if name == "lm_head.weight" {
                    self.core.weights.get("lm_head").or_else(|| {
                        self.core
                            .weights
                            .get(&format!("{}embed_tokens.weight", self.core.prefix))
                    })
                } else {
                    None
                }
            })
    }
    /// Resolve a quantized linear with the same prefix/alias logic as `w`.
    fn ql(&self, name: &str) -> Option<&QuantLinear> {
        let prefixed = format!("{}{}", self.core.prefix, name);
        let with_weight = format!("{}.weight", name);
        let prefixed_weight = format!("{}{}", self.core.prefix, with_weight);
        self.core
            .linears
            .get(&prefixed)
            .or_else(|| self.core.linears.get(&prefixed_weight))
            .or_else(|| self.core.linears.get(name))
            .or_else(|| self.core.linears.get(&with_weight))
            .or_else(|| {
                if name == "lm_head.weight" {
                    self.core.linears.get("lm_head")
                } else {
                    None
                }
            })
    }
    fn v(&self, name: &str) -> Option<&Vec<f32>> {
        let prefixed = format!("{}{}", self.core.prefix, name);
        self.core
            .vectors
            .get(&prefixed)
            .or_else(|| self.core.vectors.get(name))
    }

    /// Resolve a 1-D tensor, failing loudly with its name when absent. Silent
    /// zero fallbacks here previously produced confident-looking garbage.
    fn vec(&self, name: &str) -> Result<&[f32], String> {
        self.v(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("missing vector tensor `{}`", name))
    }

    // -----------------------------------------------------------------------
    // Forward pass helpers
    // -----------------------------------------------------------------------

    /// Zero-centered RMSNorm: `out = x * rsqrt(mean(x^2)+eps) * (1+w)`.
    fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let scale = 1.0 / (ss / x.len() as f32 + eps).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, &v)| v * scale * (1.0 + w.get(i).copied().unwrap_or(0.0)))
            .collect()
    }

    /// Gated RMSNorm (used by linear attention): norm(x) * w * silu(gate).
    /// HF instantiates it with `head_v_dim`, so normalization is per-head
    /// (groups of `group` elements), not over the whole vector.
    fn rms_norm_gated(x: &[f32], gate: &[f32], w: &[f32], eps: f32, group: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for start in (0..x.len()).step_by(group) {
            let end = (start + group).min(x.len());
            let ss: f32 = x[start..end].iter().map(|v| v * v).sum();
            let scale = 1.0 / (ss / (end - start) as f32 + eps).sqrt();
            for i in start..end {
                let wv = w.get(i - start).copied().unwrap_or(1.0); // weight is per-head (group-sized)
                let n = x[i] * scale * wv;
                let g = gate.get(i).copied().unwrap_or(0.0);
                out[i] = n * (g / (1.0 + (-g).exp()));
            }
        }
        out
    }

    /// Split an interleaved q_proj output into per-head query and gate vectors.
    /// Head `h` occupies `[h*2*hd .. h*2*hd + 2*hd)`: query in the first half,
    /// gate in the second half (HF `chunk(q.view(..., hd*2), 2, -1)`).
    fn split_q_gate(qp: &[f32], nh: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0f32; nh * hd];
        let mut gate = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let base = h * 2 * hd;
            q[h * hd..(h + 1) * hd].copy_from_slice(&qp[base..base + hd]);
            gate[h * hd..(h + 1) * hd].copy_from_slice(&qp[base + hd..base + 2 * hd]);
        }
        (q, gate)
    }

    /// GQA attention for all query heads, including the per-element sigmoid
    /// gate. `k`/`v` hold `shape.tlen` cached positions laid out as
    /// `(tlen, nkv, hd)`; query head `h` reads KV head `h / (nh / nkv)`.
    fn gqa_attention(q: &[f32], gate: &[f32], k: &[f32], v: &[f32], shape: AttnShape) -> Vec<f32> {
        let AttnShape { nh, nkv, hd, tlen } = shape;
        let scale = 1.0 / (hd as f32).sqrt();
        let groups = nh / nkv;
        let kv = nkv * hd;
        let mut out = vec![0.0f32; nh * hd];
        let mut scores = vec![0.0f32; tlen];
        for h in 0..nh {
            let kv_head = h / groups;
            let qoff = h * hd;
            for (t, score) in scores.iter_mut().enumerate() {
                let koff = t * kv + kv_head * hd;
                *score = dot(&q[qoff..qoff + hd], &k[koff..koff + hd]) * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            for (t, &a) in scores.iter().enumerate() {
                let voff = t * kv + kv_head * hd;
                for d in 0..hd {
                    out[qoff + d] += a * v[voff + d];
                }
            }
            for d in 0..hd {
                out[qoff + d] *= 1.0 / (1.0 + (-gate[qoff + d]).exp());
            }
        }
        out
    }

    /// One Gated DeltaNet recurrence step for a single value head. `state` is
    /// a `(kd, vd)` row-major slice, updated in place; the head's output is
    /// written to `scratch.o[..vd]`. The output uses the *updated* state, as
    /// in HF's `chunk_gated_delta_rule` / `recurrent_update`.
    fn deltanet_step(
        state: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: f32,
        beta: f32,
        scratch: &mut DeltaNetScratch,
    ) {
        let (kd, vd) = (scratch.kd, scratch.vd);
        let DeltaNetScratch {
            qh,
            kh,
            mem,
            delta,
            o,
            ..
        } = scratch;
        // L2-normalize q and k (HF's +1e-6 epsilon); scale q by 1/sqrt(kd).
        let qn = (q.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        let kn = (k.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        for d in 0..kd {
            qh[d] = q[d] / qn / (kd as f32).sqrt();
            kh[d] = k[d] / kn;
        }
        // Decay the state by exp(g).
        let decay = g.exp();
        for s in state.iter_mut() {
            *s *= decay;
        }
        // mem[j] = sum_i S[i,j] * k[i].
        mem.fill(0.0);
        for (i, &kv) in kh.iter().enumerate() {
            if kv == 0.0 {
                continue;
            }
            let soff = i * vd;
            for j in 0..vd {
                mem[j] += state[soff + j] * kv;
            }
        }
        // delta[j] = (v[j] - mem[j]) * beta, then S += k (x) delta.
        for j in 0..vd {
            delta[j] = (v[j] - mem[j]) * beta;
        }
        for (i, &kv) in kh.iter().enumerate() {
            if kv == 0.0 {
                continue;
            }
            let soff = i * vd;
            for j in 0..vd {
                state[soff + j] += kv * delta[j];
            }
        }
        // o[j] = sum_i S[i,j] * q[i] (updated state).
        o.fill(0.0);
        for (i, &qv) in qh.iter().enumerate() {
            if qv == 0.0 {
                continue;
            }
            let soff = i * vd;
            for j in 0..vd {
                o[j] += state[soff + j] * qv;
            }
        }
    }

    /// Linear layer: prefer a plain f32 matrix (fast) when available, then the
    /// packed quantized weight. For tied-embedding models `lm_head` shares the
    /// embedding matrix, so preferring the plain matrix avoids decoding the
    /// single largest weight (vocab x hidden) on every token.
    fn linear(&self, x: &[f32], name: &str) -> Result<Vec<f32>, String> {
        if let Some(w) = self.w(name) {
            let mut y = vec![0.0f32; w.rows];
            matmul_t(x, 1, x.len(), w, &mut y, self.core.threads);
            return Ok(y);
        }
        if let Some(ql) = self.ql(name) {
            return Ok(ql.matmul(x, self.core.threads));
        }
        // Tied checkpoints keep the embedding table as fp16 only; the same
        // table is the lm_head. Decode it on the fly via the threaded f16
        // matmul instead of maintaining a full f32 vocab x hidden matrix.
        if name == "lm_head.weight" {
            let embed = format!("{}embed_tokens.weight", self.core.prefix);
            if let Some(w) = self
                .core
                .embeddings_f16
                .get(&embed)
                .or_else(|| self.core.embeddings_f16.get("model.embed_tokens.weight"))
            {
                let mut y = vec![0.0f32; w.rows];
                matmul_f16(x, w, &mut y, self.core.threads);
                return Ok(y);
            }
        }
        Err(format!("missing linear weight `{}`", name))
    }

    /// Apply RoPE to the first `dim` elements of each head in `x`
    /// (head-major layout: head h starts at h*head_len).
    fn apply_rope(&self, x: &mut [f32], head_len: usize, num_heads: usize, pos: usize) {
        let half = self.core.rope_half;
        if half == 0 || pos >= self.core.context_size {
            return;
        }
        let table_base = pos * half;
        for h in 0..num_heads {
            let base = h * head_len;
            for d in 0..half {
                let c = self.core.rope_cos[table_base + d];
                let s = self.core.rope_sin[table_base + d];
                let a = x[base + d];
                let b = x[base + d + half];
                x[base + d] = a * c - b * s;
                x[base + d + half] = b * c + a * s;
            }
        }
    }

    /// One full-attention (GQA) layer.
    fn forward_full_attn(
        &mut self,
        layer: usize,
        x: &[f32],
        state_idx: usize,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.core.config;
        let norm_w = self.vec(&format!("layers.{}.input_layernorm.weight", layer))?;
        let xn = Self::rms_norm(x, norm_w, cfg.rms_norm_eps);

        let qp = self.linear(&xn, &format!("layers.{}.self_attn.q_proj", layer))?;
        let mut kp = self.linear(&xn, &format!("layers.{}.self_attn.k_proj", layer))?;
        let vp = self.linear(&xn, &format!("layers.{}.self_attn.v_proj", layer))?;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;

        if qp.len() < 2 * nh * hd || kp.len() < nkv * hd || vp.len() < nkv * hd {
            return Err(format!(
                "layer {}: unexpected q/k/v projection shapes",
                layer
            ));
        }

        // q_proj output is interleaved per head: head h has [query(hd), gate(hd)].
        let (mut q, gate) = Self::split_q_gate(&qp, nh, hd);

        // Per-head RMSNorm (zero-centered) on q and k.
        let qn_w = self.vec(&format!("layers.{}.self_attn.q_norm.weight", layer))?;
        let kn_w = self.vec(&format!("layers.{}.self_attn.k_norm.weight", layer))?;
        for h in 0..nh {
            let s = &mut q[h * hd..(h + 1) * hd];
            let ss: f32 = s.iter().map(|v| v * v).sum();
            let sc = 1.0 / (ss / hd as f32 + cfg.rms_norm_eps).sqrt();
            for (d, v) in s.iter_mut().enumerate() {
                *v = *v * sc * (1.0 + qn_w.get(d).copied().unwrap_or(0.0));
            }
        }
        for h in 0..nkv {
            let s = &mut kp[h * hd..(h + 1) * hd];
            let ss: f32 = s.iter().map(|v| v * v).sum();
            let sc = 1.0 / (ss / hd as f32 + cfg.rms_norm_eps).sqrt();
            for (d, v) in s.iter_mut().enumerate() {
                *v = *v * sc * (1.0 + kn_w.get(d).copied().unwrap_or(0.0));
            }
        }

        // RoPE (text-only: all position IDs equal, mrope is a no-op).
        self.apply_rope(&mut q, hd, nh, pos);
        let mut k = kp[..nkv * hd].to_vec();
        self.apply_rope(&mut k, hd, nkv, pos);

        // Write the new K/V row as fp16 bits (halves context memory).
        let kv = nkv * hd;
        if pos < self.core.context_size {
            let st = &mut self.full_states[state_idx];
            for (dst, &src) in st.k[pos * kv..(pos + 1) * kv].iter_mut().zip(&k) {
                *dst = f32_to_half(src);
            }
            for (dst, &src) in st.v[pos * kv..(pos + 1) * kv].iter_mut().zip(&vp[..kv]) {
                *dst = f32_to_half(src);
            }
        }

        // Causal attention over cached keys/values, then sigmoid(gate).
        // Decode the used prefix through the f16 LUT once per token rather
        // than doing a slow powi-based conversion at every score/value read.
        let tlen = pos.min(self.core.context_size - 1) + 1;
        let st = &self.full_states[state_idx];
        let k_f32: Vec<f32> = st.k[..tlen * kv].iter().map(|&b| half_f32(b)).collect();
        let v_f32: Vec<f32> = st.v[..tlen * kv].iter().map(|&b| half_f32(b)).collect();
        let attn = Self::gqa_attention(&q, &gate, &k_f32, &v_f32, AttnShape { nh, nkv, hd, tlen });

        // Output projection.
        let y = self.linear(&attn, &format!("layers.{}.self_attn.o_proj", layer))?;
        if y.len() != cfg.hidden_size {
            return Err(format!("layer {}: unexpected o_proj shape", layer));
        }
        Ok(y)
    }

    /// One linear-attention (Gated DeltaNet) layer.
    fn forward_linear_attn(
        &mut self,
        layer: usize,
        x: &[f32],
        state_idx: usize,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.core.config;
        let norm_w = self.vec(&format!("layers.{}.input_layernorm.weight", layer))?;
        let xn = Self::rms_norm(x, norm_w, cfg.rms_norm_eps);

        let qkv = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_qkv", layer))?;
        let z = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_z", layer))?;
        let b = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_b", layer))?;
        let a = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_a", layer))?;

        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nkh = cfg.linear_num_key_heads;
        let nvh = cfg.linear_num_value_heads;
        let key_dim = kd * nkh;
        let val_dim = vd * nvh;
        let conv_dim = key_dim * 2 + val_dim;
        if qkv.len() < conv_dim || z.len() < val_dim {
            return Err(format!(
                "layer {}: unexpected linear-attn projection shapes",
                layer
            ));
        }

        // Depthwise causal conv1d (kernel = conv kernel dim) + SiLU.
        let cw = self.vec(&format!("layers.{}.linear_attn.conv1d.weight", layer))?;
        let kernel = cfg.linear_conv_kernel_dim;
        if cw.len() < kernel * conv_dim {
            return Err(format!(
                "layer {}: conv1d.weight has {} values, expected {}",
                layer,
                cw.len(),
                kernel * conv_dim
            ));
        }
        let cw = cw.to_vec(); // release the immutable borrow of `self`
        let hist = kernel - 1; // number of stored previous qkv vectors
        let st = &mut self.linear_states[state_idx];
        // Build the window [oldest ... newest] from the circular buffer of
        // the last `hist` vectors, with the current qkv as the newest entry.
        let mut window = vec![0.0f32; kernel * conv_dim];
        for j in 0..hist {
            let idx = (st.conv_pos + j) % hist;
            window[j * conv_dim..(j + 1) * conv_dim]
                .copy_from_slice(&st.conv[idx * conv_dim..(idx + 1) * conv_dim]);
        }
        window[hist * conv_dim..kernel * conv_dim].copy_from_slice(&qkv[..conv_dim]);
        // Overwrite the oldest slot with the new qkv, then advance the pointer.
        st.conv[st.conv_pos * conv_dim..(st.conv_pos + 1) * conv_dim]
            .copy_from_slice(&qkv[..conv_dim]);
        st.conv_pos = (st.conv_pos + 1) % hist;

        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut v = 0.0f32;
            for j in 0..kernel {
                // conv1d.weight is (conv_dim, 1, kernel): channel-major.
                v += cw[c * kernel + j] * window[j * conv_dim + c];
            }
            conv_out[c] = v / (1.0 + (-v).exp()); // silu
        }

        // Split into query, key, value.
        let q = &conv_out[..key_dim];
        let k = &conv_out[key_dim..2 * key_dim];
        let v = &conv_out[2 * key_dim..2 * key_dim + val_dim];

        // Repeat key heads to match value heads (num_v / num_k = 2).
        let rep = nvh / nkh;

        // g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b).
        let alog = self.vec(&format!("layers.{}.linear_attn.A_log", layer))?;
        let dtbias = self.vec(&format!("layers.{}.linear_attn.dt_bias", layer))?;
        if a.len() < nvh || b.len() < nvh || alog.len() < nvh || dtbias.len() < nvh {
            return Err(format!(
                "layer {}: gated deltanet parameter vectors are too short",
                layer
            ));
        }
        let mut g = vec![0.0f32; nvh];
        let mut beta = vec![0.0f32; nvh];
        for h in 0..nvh {
            let ab = a[h] + dtbias[h];
            let sp = if ab > 20.0 { ab } else { (1.0 + ab.exp()).ln() };
            g[h] = -alog[h].exp() * sp;
            let bb = b[h];
            beta[h] = 1.0 / (1.0 + (-bb).exp());
        }

        // Recurrent state update (fp32), per value head. Scratch buffers are
        // allocated once per layer and reused for every head/token.
        let st = &mut self.linear_states[state_idx];
        let mut scratch = DeltaNetScratch {
            kd,
            vd,
            qh: vec![0.0f32; kd],
            kh: vec![0.0f32; kd],
            mem: vec![0.0f32; vd],
            delta: vec![0.0f32; vd],
            o: vec![0.0f32; vd],
        };
        let mut o = vec![0.0f32; val_dim];
        for vh in 0..nvh {
            let kh = vh / rep; // key heads are repeated to match value heads
            let qbase = kh * kd;
            let kbase = kh * kd;
            let vbase = vh * vd;
            let sbase = vh * kd * vd;
            Self::deltanet_step(
                &mut st.state[sbase..sbase + kd * vd],
                &q[qbase..qbase + kd],
                &k[kbase..kbase + kd],
                &v[vbase..vbase + vd],
                g[vh],
                beta[vh],
                &mut scratch,
            );
            o[vbase..vbase + vd].copy_from_slice(&scratch.o[..vd]);
        }

        // Gated RMSNorm: norm(o) * norm.weight * silu(z).
        let norm_w = self.vec(&format!("layers.{}.linear_attn.norm.weight", layer))?;
        let o_norm = Self::rms_norm_gated(&o, &z, norm_w, cfg.rms_norm_eps, vd);

        // Output projection.
        let y = self.linear(&o_norm, &format!("layers.{}.linear_attn.out_proj", layer))?;
        if y.len() != cfg.hidden_size {
            return Err(format!("layer {}: unexpected out_proj shape", layer));
        }
        Ok(y)
    }

    /// Look up one embedding row. Untied models store the table as fp16 and
    /// decode just one row; tied models keep the f32 matrix (also used by the
    /// tied lm_head). The prefixed fp16 key is tried first, so a multimodal
    /// checkpoint (`model.language_model.`) resolves correctly.
    fn embedding_row(&self, token: u32) -> Result<Vec<f32>, String> {
        let id = token as usize;
        let embed_name = format!("{}embed_tokens.weight", self.core.prefix);
        if let Some(emb) = self
            .core
            .embeddings_f16
            .get(&embed_name)
            .or_else(|| self.core.embeddings_f16.get("model.embed_tokens.weight"))
        {
            if id >= emb.rows {
                return Err(format!(
                    "token id {} is outside embedding range ({} rows)",
                    token, emb.rows
                ));
            }
            return Ok(emb.row_f32(id));
        }
        let embed = self
            .w("embed_tokens.weight")
            .ok_or_else(|| "missing embedding tensor `embed_tokens.weight`".to_string())?;
        if id >= embed.rows {
            return Err(format!(
                "token id {} is outside embedding range ({} rows)",
                token, embed.rows
            ));
        }
        Ok(embed.row(id).to_vec())
    }

    fn forward_token(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let num_layers = self.core.config.num_layers;
        let hs = self.core.config.hidden_size;
        let vocab = self.core.config.vocab_size;
        let eps = self.core.config.rms_norm_eps;
        let layer_types = self.core.config.layer_types.clone();
        let intv = self.core.config.full_attention_interval;

        let mut x = self.embedding_row(token)?;

        let pos = self.position;
        let mut full_idx = 0usize;
        let mut linear_idx = 0usize;
        for i in 0..num_layers {
            let is_full = if i < layer_types.len() {
                layer_types[i] == "full_attention"
            } else {
                (i + 1) % intv == 0
            };
            let attn = if is_full {
                let y = self.forward_full_attn(i, &x, full_idx, pos)?;
                full_idx += 1;
                y
            } else {
                let y = self.forward_linear_attn(i, &x, linear_idx)?;
                linear_idx += 1;
                y
            };
            for (j, v) in attn.iter().enumerate() {
                x[j] += v;
            }

            // MLP.
            let norm_w = self.vec(&format!("layers.{}.post_attention_layernorm.weight", i))?;
            let xn = Self::rms_norm(&x, norm_w, eps);
            let gate = self.linear(&xn, &format!("layers.{}.mlp.gate_proj", i))?;
            let up = self.linear(&xn, &format!("layers.{}.mlp.up_proj", i))?;
            if gate.len() != up.len() || gate.is_empty() {
                return Err(format!("layer {}: mismatched MLP gate/up shapes", i));
            }
            let mut act = vec![0.0f32; gate.len()];
            for j in 0..gate.len() {
                let g = gate[j];
                act[j] = (g / (1.0 + (-g).exp())) * up[j];
            }
            let mlp_out = self.linear(&act, &format!("layers.{}.mlp.down_proj", i))?;
            if mlp_out.len() != hs {
                return Err(format!("layer {}: unexpected down_proj shape", i));
            }
            for (j, v) in mlp_out.iter().enumerate() {
                x[j] += v;
            }
        }

        self.position += 1;

        // Final norm + lm_head.
        let final_w = self.vec("norm.weight")?;
        let xf = Self::rms_norm(&x, final_w, eps);
        let logits = self.linear(&xf, "lm_head.weight")?;
        if logits.len() != vocab {
            return Err(format!(
                "lm_head produced {} logits, expected {}",
                logits.len(),
                vocab
            ));
        }
        Ok(logits)
    }

    pub fn forward(&mut self, input_ids: &[u32]) -> Result<Vec<f32>, String> {
        if input_ids.len() > self.core.context_size.saturating_sub(self.position) {
            return Err(format!(
                "context size exceeded: position {} + {} input tokens > context {}",
                self.position,
                input_ids.len(),
                self.core.context_size
            ));
        }
        let mut logits = vec![0.0f32; self.core.config.vocab_size];
        for &t in input_ids {
            logits = self.forward_token(t)?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
impl Model {
    /// Deterministic tiny full-attention model used by HTTP/engine tests.
    /// It needs no EXL3 files: all weights are small synthetic matrices.
    pub(crate) fn tiny() -> Model {
        fn mat(rows: usize, cols: usize, seed: usize) -> Matrix {
            let data = (0..rows * cols)
                .map(|i| (((i + seed * 17) as f32) * 0.031).sin() * 0.2)
                .collect();
            Matrix::from_vec(rows, cols, data)
        }

        let context_size = 64usize;
        let config = ModelConfig {
            architecture: "Qwen3_5ForCausalLM".into(),
            num_layers: 1,
            hidden_size: 4,
            vocab_size: 261,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 4,
            layer_types: vec!["full_attention".into()],
            full_attention_interval: 4,
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 4,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 1e4,
            partial_rotary_factor: 0.5,
            tie_word_embeddings: false,
            eos_token_id: None,
            enable_thinking: None,
        };

        let mut weights = HashMap::new();
        weights.insert("model.embed_tokens.weight".into(), mat(261, 4, 1));
        weights.insert("model.layers.0.self_attn.q_proj".into(), mat(8, 4, 2));
        weights.insert("model.layers.0.self_attn.k_proj".into(), mat(4, 4, 3));
        weights.insert("model.layers.0.self_attn.v_proj".into(), mat(4, 4, 4));
        weights.insert("model.layers.0.self_attn.o_proj".into(), mat(4, 4, 5));
        weights.insert("model.layers.0.mlp.gate_proj".into(), mat(4, 4, 6));
        weights.insert("model.layers.0.mlp.up_proj".into(), mat(4, 4, 7));
        weights.insert("model.layers.0.mlp.down_proj".into(), mat(4, 4, 8));
        weights.insert("lm_head.weight".into(), mat(261, 4, 9));

        let mut vectors = HashMap::new();
        for name in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
        ] {
            vectors.insert(name.into(), vec![0.0f32; 4]);
        }

        let rotary = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
        let rope_half = rotary / 2;
        let mut rope_cos = vec![0.0f32; context_size * rope_half];
        let mut rope_sin = vec![0.0f32; context_size * rope_half];
        for pos in 0..context_size {
            for d in 0..rope_half {
                let freq = 1.0 / config.rope_theta.powf(2.0 * d as f32 / rotary as f32);
                let (s, c) = (pos as f32 * freq).sin_cos();
                rope_cos[pos * rope_half + d] = c;
                rope_sin[pos * rope_half + d] = s;
            }
        }

        let full_states = vec![FullAttnState {
            k: vec![0u16; context_size * 4],
            v: vec![0u16; context_size * 4],
        }];
        let core = Arc::new(ModelCore {
            config,
            prefix: "model.".into(),
            weights,
            embeddings_f16: HashMap::new(),
            linears: HashMap::new(),
            vectors,
            threads: 1,
            context_size,
            rope_cos,
            rope_sin,
            rope_half,
            full_template: full_states.clone(),
            linear_template: Vec::new(),
        });
        Model {
            core,
            position: 0,
            full_states,
            linear_states: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_row_resolves_multimodal_fp16_embedding() {
        let mut m = Model::tiny();
        // Untied multimodal checkpoint: there is no f32 embedding in
        // `weights`; the fp16 table lives under the language-model prefix.
        m.core_mut().weights.remove("model.embed_tokens.weight");
        m.core_mut().prefix = "model.language_model.".into();
        m.core_mut().embeddings_f16.insert(
            "model.language_model.embed_tokens.weight".into(),
            F16Matrix {
                rows: 2,
                cols: 4,
                data: vec![0x3c00; 8], // fp16 1.0
            },
        );
        assert_eq!(m.embedding_row(1).unwrap(), vec![1.0f32; 4]);

        // A wrong prefix must fail loudly rather than mis-tokenize.
        m.core_mut().prefix = "model.".into();
        assert!(m.embedding_row(0).is_err());
    }

    #[test]
    fn kv_cache_stores_fp16_bits() {
        let m = Model::tiny();
        let kv = m.core.config.num_key_value_heads * m.core.config.head_dim;
        assert_eq!(m.full_states[0].k.len(), m.core.context_size * kv);
        assert_eq!(m.full_states[0].v.len(), m.core.context_size * kv);
        // A fresh cache is all-zero fp16 bits, and the LUT decoder used by
        // attention round-trips a representative value exactly.
        assert!(m.full_states[0].k.iter().all(|&b| b == 0));
        assert!(m.full_states[0].v.iter().all(|&b| b == 0));
        let bits = f32_to_half(1.25);
        assert!((half_f32(bits) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn f16_embedding_matmul_matches_f32() {
        let rows = 4usize;
        let cols = 3usize;
        let f32_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.25 - 0.5).collect();
        let f16 = F16Matrix {
            rows,
            cols,
            data: f32_data.iter().map(|&v| f32_to_half(v)).collect(),
        };
        let x = [0.5f32, -1.0, 2.0];
        let mut got = vec![0.0f32; rows];
        matmul_f16(&x, &f16, &mut got, 2);
        for j in 0..rows {
            let expect: f32 = x
                .iter()
                .zip(&f32_data[j * cols..(j + 1) * cols])
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (got[j] - expect).abs() < 1e-3,
                "row {}: got {} expected {}",
                j,
                got[j],
                expect
            );
        }
    }

    #[test]
    fn spawned_instance_shares_weights_but_not_state() {
        let mut m = Model::tiny();
        let other = m.spawn();
        // A forward pass on one instance must not move the other's position.
        assert_eq!(other.position, 0);
        m.forward(&[0]).unwrap();
        assert_eq!(m.position, 1);
        assert_eq!(other.position, 0);
        assert_eq!(m.full_states[0].k.len(), other.full_states[0].k.len());
        // Both see the same config from the shared core.
        assert_eq!(m.config().vocab_size, other.config().vocab_size);
    }

    #[test]
    fn tied_f16_embedding_supports_forward() {
        let mut m = Model::tiny();
        // Simulate a tied checkpoint: the embedding is fp16-only and the
        // separate f32 lm_head matrix is absent, so lm_head must read the
        // embedding table through the f16 matmul path.
        let emb = m
            .core_mut()
            .weights
            .remove("model.embed_tokens.weight")
            .unwrap();
        m.core_mut().weights.remove("lm_head.weight");
        m.core_mut().embeddings_f16.insert(
            "model.embed_tokens.weight".into(),
            F16Matrix {
                rows: emb.rows,
                cols: emb.cols,
                data: emb.data.iter().map(|&v| f32_to_half(v)).collect(),
            },
        );
        let logits = m.forward(&[0]).unwrap();
        assert_eq!(logits.len(), m.core.config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rms_norm_matches_hand_computation() {
        // x = [3, 4] -> mean(x^2) = 12.5, scale = 1/sqrt(12.5).
        let scale = 1.0 / 12.5f32.sqrt();
        let out = Model::rms_norm(&[3.0, 4.0], &[1.0, 0.0], 0.0);
        assert!((out[0] - 3.0 * scale * 2.0).abs() < 1e-6); // (1 + w[0]) = 2
        assert!((out[1] - 4.0 * scale).abs() < 1e-6); // (1 + w[1]) = 1

        // Missing weight entries default to 0, i.e. multiplier 1.
        let bare = Model::rms_norm(&[3.0, 4.0], &[], 0.0);
        assert!((bare[0] - 3.0 * scale).abs() < 1e-6);
        assert!((bare[1] - 4.0 * scale).abs() < 1e-6);

        // eps keeps all-zero input finite.
        assert_eq!(
            Model::rms_norm(&[0.0, 0.0], &[0.0, 0.0], 1e-6),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn rms_norm_gated_normalizes_per_head() {
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let x = [3.0f32, 4.0, 0.0, 8.0];
        let gate = [0.0f32, 1.0, 2.0, -1.0];
        let w = [1.0f32, 0.5]; // weight is group-sized, reused per head
        let out = Model::rms_norm_gated(&x, &gate, &w, 0.0, 2);

        // Head 0: x=[3,4], mean(x^2)=12.5; head 1: x=[0,8], mean=32.
        let s0 = 1.0 / 12.5f32.sqrt();
        let s1 = 1.0 / 32.0f32.sqrt();
        assert!((out[0] - 3.0 * s0 * 1.0 * silu(0.0)).abs() < 1e-6);
        assert!((out[1] - 4.0 * s0 * 0.5 * silu(1.0)).abs() < 1e-6);
        assert!(out[2].abs() < 1e-6);
        assert!((out[3] - 8.0 * s1 * 0.5 * silu(-1.0)).abs() < 1e-6);
    }

    #[test]
    fn rope_is_partial_and_position_indexed() {
        let m = Model::tiny(); // head_dim=4, partial_rotary_factor=0.5 -> 2 rotary dims
        assert_eq!(m.core.rope_half, 1);
        assert!((m.core.rope_cos[0] - 1.0).abs() < 1e-6);
        assert!(m.core.rope_sin[0].abs() < 1e-6);
        assert!((m.core.rope_cos[1] - 1.0f32.cos()).abs() < 1e-6);
        assert!((m.core.rope_sin[1] - 1.0f32.sin()).abs() < 1e-6);

        // Position 0 is the identity rotation.
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        m.apply_rope(&mut x, 4, 1, 0);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);

        // Position 1 rotates only the first 2 dims (rotate-half layout).
        m.apply_rope(&mut x, 4, 1, 1);
        let (c, s) = (m.core.rope_cos[1], m.core.rope_sin[1]);
        assert!((x[0] - (1.0 * c - 2.0 * s)).abs() < 1e-6);
        assert!((x[1] - (2.0 * c + 1.0 * s)).abs() < 1e-6);
        assert_eq!(x[2], 3.0); // dims past the rotary width are untouched
        assert_eq!(x[3], 4.0);
    }

    #[test]
    fn q_proj_interleaving_matches_per_head_query_gate() {
        let qp: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let (q, gate) = Model::split_q_gate(&qp, 2, 2);
        assert_eq!(q, vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(gate, vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn gqa_attention_groups_heads_and_applies_gate() {
        let (nh, nkv, hd, tlen) = (4usize, 2usize, 2usize, 2usize);
        let mut q = vec![0.0f32; nh * hd];
        let mut gate = vec![0.0f32; nh * hd];
        for h in 0..nh {
            for d in 0..hd {
                q[h * hd + d] = 0.5 * h as f32 + d as f32;
                gate[h * hd + d] = if h % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let mut k = vec![0.0f32; tlen * nkv * hd];
        let mut v = vec![0.0f32; tlen * nkv * hd];
        for t in 0..tlen {
            for h in 0..nkv {
                for d in 0..hd {
                    let i = (t * nkv + h) * hd + d;
                    k[i] = 1.0 + 10.0 * t as f32 + 4.0 * h as f32 + d as f32;
                    v[i] = 2.0 + 100.0 * t as f32 + 20.0 * h as f32 + d as f32;
                }
            }
        }

        let got = Model::gqa_attention(&q, &gate, &k, &v, AttnShape { nh, nkv, hd, tlen });
        for h in 0..nh {
            let kv_head = h / (nh / nkv); // heads 0,1 -> kv 0; heads 2,3 -> kv 1
            let mut scores = [0.0f32; 2];
            for (t, score) in scores.iter_mut().enumerate() {
                let koff = (t * nkv + kv_head) * hd;
                *score = (q[h * hd] * k[koff] + q[h * hd + 1] * k[koff + 1]) / (hd as f32).sqrt();
            }
            let max = scores[0].max(scores[1]);
            let e = [(scores[0] - max).exp(), (scores[1] - max).exp()];
            let a = [e[0] / (e[0] + e[1]), e[1] / (e[0] + e[1])];
            for d in 0..hd {
                let v0 = v[kv_head * hd + d];
                let v1 = v[(nkv + kv_head) * hd + d];
                let mut expect = a[0] * v0 + a[1] * v1;
                let g = gate[h * hd + d];
                expect *= 1.0 / (1.0 + (-g).exp());
                assert!(
                    (got[h * hd + d] - expect).abs() < 1e-5,
                    "head {} dim {}: got {} expected {}",
                    h,
                    d,
                    got[h * hd + d],
                    expect
                );
            }
        }
    }

    #[test]
    fn deltanet_step_uses_updated_state_and_decay() {
        let (kd, vd) = (2usize, 2usize);
        let mut state = vec![0.0f32; kd * vd];
        let mut scratch = DeltaNetScratch {
            kd,
            vd,
            qh: vec![0.0; kd],
            kh: vec![0.0; kd],
            mem: vec![0.0; vd],
            delta: vec![0.0; vd],
            o: vec![0.0; vd],
        };
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0];
        let v = [2.0f32, 3.0];
        let qscale = 1.0 / 2.0f32.sqrt(); // 1/sqrt(kd) with q = [1, 0]

        // Step 1 from a zero state: S += k (x) ((v - 0) * 0.5), then o = S @ q.
        Model::deltanet_step(&mut state, &q, &k, &v, 0.0, 0.5, &mut scratch);
        assert!((state[0] - 1.0).abs() < 1e-4);
        assert!((state[1] - 1.5).abs() < 1e-4);
        assert!(state[2].abs() < 1e-6 && state[3].abs() < 1e-6);
        assert!((scratch.o[0] - qscale).abs() < 1e-4);
        assert!((scratch.o[1] - 1.5 * qscale).abs() < 1e-4);

        // Step 2: mem = [1, 1.5], delta = [(2-1)*0.5, (3-1.5)*0.5].
        Model::deltanet_step(&mut state, &q, &k, &v, 0.0, 0.5, &mut scratch);
        assert!((state[0] - 1.5).abs() < 1e-4);
        assert!((state[1] - 2.25).abs() < 1e-4);
        assert!((scratch.o[0] - 1.5 * qscale).abs() < 1e-4);
        assert!((scratch.o[1] - 2.25 * qscale).abs() < 1e-4);

        // Decay: exp(g) = 0.5 halves the state before the (beta = 0) update.
        Model::deltanet_step(&mut state, &q, &k, &v, 0.5f32.ln(), 0.0, &mut scratch);
        assert!((state[0] - 0.75).abs() < 1e-4);
        assert!((state[1] - 1.125).abs() < 1e-4);
    }
}
