//! Qwen3.5 text decoder: config, weights, KV cache, and forward pass.

use std::collections::HashMap;

use crate::exl3::{Exl3Model, QuantLinear};
use crate::json::Json;
use crate::tensor::{dot, matmul_t, Matrix};

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
}

/// State for one full-attention layer: KV cache.
struct FullAttnState {
    k: Vec<f32>, // (max_ctx, num_kv_heads, head_dim)
    v: Vec<f32>,
}

/// State for one linear-attention layer: convolution buffer + recurrent state.
struct LinearAttnState {
    /// Circular buffer of the last `kernel` qkv vectors (oldest at index 0).
    conv: Vec<f32>, // (kernel, conv_dim)
    conv_pos: usize,
    /// Recurrent state S, one (value_dim x key_dim) matrix per value head.
    state: Vec<f32>, // (num_value_heads, value_dim, key_dim)
}

pub struct Model {
    pub config: ModelConfig,
    prefix: String,
    weights: HashMap<String, Matrix>,
    linears: HashMap<String, QuantLinear>,
    vectors: HashMap<String, Vec<f32>>,
    threads: usize,
    context_size: usize,
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

        // Detect tensor prefix.
        let has_language = ex
            .weights
            .contains_key("model.language_model.embed_tokens.weight")
            || ex.w("model.language_model.embed_tokens.weight").is_some();
        let prefix = if has_language {
            "model.language_model.".to_string()
        } else {
            "model.".to_string()
        };

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
        };

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
                    k: vec![0.0; context_size * kv],
                    v: vec![0.0; context_size * kv],
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

        Ok(Model {
            config,
            prefix,
            weights: ex.weights,
            linears: ex.linears,
            vectors: ex.vectors,
            threads,
            context_size,
            position: 0,
            full_states,
            linear_states,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for s in &mut self.full_states {
            s.k.fill(0.0);
            s.v.fill(0.0);
        }
        for s in &mut self.linear_states {
            s.conv.fill(0.0);
            s.conv_pos = 0;
            s.state.fill(0.0);
        }
    }

    /// Return a weight matrix by unprefixed name, trying the language-model
    /// prefix first and then the bare name (e.g. top-level `lm_head.weight` in
    /// multimodal checkpoints).
    fn w(&self, name: &str) -> Option<&Matrix> {
        let prefixed = format!("{}{}", self.prefix, name);
        let with_weight = format!("{}.weight", name);
        let prefixed_weight = format!("{}{}", self.prefix, with_weight);
        self.weights
            .get(&prefixed)
            .or_else(|| self.weights.get(&prefixed_weight))
            .or_else(|| self.weights.get(name))
            .or_else(|| self.weights.get(&with_weight))
            .or_else(|| {
                // EXL3 checkpoints store the quantized lm_head under the bare
                // base name `lm_head` (no `.weight` suffix). Some models tie
                // embeddings and have no separate lm_head at all — fall back
                // to the embedding matrix in that case.
                if name == "lm_head.weight" {
                    self.weights.get("lm_head").or_else(|| {
                        self.weights
                            .get(&format!("{}embed_tokens.weight", self.prefix))
                    })
                } else {
                    None
                }
            })
    }
    /// Resolve a quantized linear with the same prefix/alias logic as `w`.
    fn ql(&self, name: &str) -> Option<&QuantLinear> {
        let prefixed = format!("{}{}", self.prefix, name);
        let with_weight = format!("{}.weight", name);
        let prefixed_weight = format!("{}{}", self.prefix, with_weight);
        self.linears
            .get(&prefixed)
            .or_else(|| self.linears.get(&prefixed_weight))
            .or_else(|| self.linears.get(name))
            .or_else(|| self.linears.get(&with_weight))
            .or_else(|| {
                if name == "lm_head.weight" {
                    self.linears.get("lm_head")
                } else {
                    None
                }
            })
    }
    fn v(&self, name: &str) -> Option<&Vec<f32>> {
        let prefixed = format!("{}{}", self.prefix, name);
        self.vectors
            .get(&prefixed)
            .or_else(|| self.vectors.get(name))
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

    /// Linear layer: prefer a plain f32 matrix (fast) when available, then the
    /// packed quantized weight. For tied-embedding models `lm_head` shares the
    /// embedding matrix, so preferring the plain matrix avoids decoding the
    /// single largest weight (vocab x hidden) on every token.
    fn linear(&self, x: &[f32], name: &str) -> Vec<f32> {
        if let Some(w) = self.w(name) {
            let mut y = vec![0.0f32; w.rows];
            matmul_t(x, 1, x.len(), w, &mut y, self.threads);
            return y;
        }
        if let Some(ql) = self.ql(name) {
            return ql.matmul(x, self.threads);
        }
        vec![0.0; 0]
    }

    /// Apply RoPE to the first `dim` elements of each head in `x`
    /// (head-major layout: head h starts at h*head_len).
    fn apply_rope(x: &mut [f32], head_len: usize, num_heads: usize, pos: usize, cfg: &ModelConfig) {
        let rotary = (head_len as f32 * cfg.partial_rotary_factor) as usize;
        let half = rotary / 2;
        if half == 0 {
            return;
        }
        for h in 0..num_heads {
            let base = h * head_len;
            for d in 0..half {
                let freq = 1.0 / cfg.rope_theta.powf(2.0 * d as f32 / rotary as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
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
    ) -> Vec<f32> {
        let cfg = &self.config;
        let norm_w = self
            .v(&format!("layers.{}.input_layernorm.weight", layer))
            .cloned()
            .unwrap_or_default();
        let xn = Self::rms_norm(x, &norm_w, cfg.rms_norm_eps);

        let qp = self.linear(&xn, &format!("layers.{}.self_attn.q_proj", layer));
        let mut kp = self.linear(&xn, &format!("layers.{}.self_attn.k_proj", layer));
        let vp = self.linear(&xn, &format!("layers.{}.self_attn.v_proj", layer));
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;

        if qp.len() < 2 * nh * hd || kp.len() < nkv * hd {
            return vec![0.0; cfg.hidden_size];
        }

        // q_proj output is interleaved per head: head h has [query(hd), gate(hd)].
        let mut q = vec![0.0f32; nh * hd];
        let mut gate = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let base = h * 2 * hd;
            q[h * hd..(h + 1) * hd].copy_from_slice(&qp[base..base + hd]);
            gate[h * hd..(h + 1) * hd].copy_from_slice(&qp[base + hd..base + 2 * hd]);
        }

        // Per-head RMSNorm (zero-centered) on q and k.
        let qn_w = self
            .v(&format!("layers.{}.self_attn.q_norm.weight", layer))
            .cloned()
            .unwrap_or_default();
        let kn_w = self
            .v(&format!("layers.{}.self_attn.k_norm.weight", layer))
            .cloned()
            .unwrap_or_default();
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
        Self::apply_rope(&mut q, hd, nh, pos, cfg);
        let mut k = kp[..nkv * hd].to_vec();
        Self::apply_rope(&mut k, hd, nkv, pos, cfg);

        // Write to KV cache.
        let kv = nkv * hd;
        if pos < self.context_size {
            let st = &mut self.full_states[state_idx];
            st.k[pos * kv..(pos + 1) * kv].copy_from_slice(&k);
            st.v[pos * kv..(pos + 1) * kv].copy_from_slice(&vp[..kv]);
        }

        // Attention over cached keys/values.
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn = vec![0.0f32; nh * hd];
        let groups = nh / nkv;
        let tlen = pos.min(self.context_size - 1) + 1;
        let mut scores = vec![0.0f32; tlen];
        for h in 0..nh {
            let kv_head = h / groups;
            let qoff = h * hd;
            for (t, score) in scores.iter_mut().enumerate() {
                let koff = t * kv + kv_head * hd;
                let st = &self.full_states[state_idx];
                *score = dot(&q[qoff..qoff + hd], &st.k[koff..koff + hd]) * scale;
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
            attn[qoff..qoff + hd].fill(0.0);
            for (t, &a) in scores.iter().enumerate() {
                let voff = t * kv + kv_head * hd;
                let st = &self.full_states[state_idx];
                for d in 0..hd {
                    attn[qoff + d] += a * st.v[voff + d];
                }
            }
        }

        // Gating with sigmoid(gate).
        for i in 0..nh * hd {
            let g = gate[i];
            attn[i] *= 1.0 / (1.0 + (-g).exp());
        }

        // Output projection.
        let y = self.linear(&attn, &format!("layers.{}.self_attn.o_proj", layer));
        if y.len() != cfg.hidden_size {
            return vec![0.0; cfg.hidden_size];
        }
        y
    }

    /// One linear-attention (Gated DeltaNet) layer.
    fn forward_linear_attn(&mut self, layer: usize, x: &[f32], state_idx: usize) -> Vec<f32> {
        let cfg = &self.config;
        let norm_w = self
            .v(&format!("layers.{}.input_layernorm.weight", layer))
            .cloned()
            .unwrap_or_default();
        let xn = Self::rms_norm(x, &norm_w, cfg.rms_norm_eps);

        let qkv = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_qkv", layer));
        let z = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_z", layer));
        let b = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_b", layer));
        let a = self.linear(&xn, &format!("layers.{}.linear_attn.in_proj_a", layer));

        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nkh = cfg.linear_num_key_heads;
        let nvh = cfg.linear_num_value_heads;
        let key_dim = kd * nkh;
        let val_dim = vd * nvh;
        let conv_dim = key_dim * 2 + val_dim;
        if qkv.len() < conv_dim {
            return vec![0.0; cfg.hidden_size];
        }

        // Depthwise causal conv1d (kernel = conv kernel dim) + SiLU.
        let cw = self
            .v(&format!("layers.{}.linear_attn.conv1d.weight", layer))
            .cloned()
            .unwrap_or_default();
        let kernel = cfg.linear_conv_kernel_dim;
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
        if cw.len() >= kernel * conv_dim {
            for c in 0..conv_dim {
                let mut v = 0.0f32;
                for j in 0..kernel {
                    // conv1d.weight is (conv_dim, 1, kernel): channel-major.
                    v += cw[c * kernel + j] * window[j * conv_dim + c];
                }
                conv_out[c] = v / (1.0 + (-v).exp()); // silu
            }
        }

        // Split into query, key, value.
        let q = &conv_out[..key_dim];
        let k = &conv_out[key_dim..2 * key_dim];
        let v = &conv_out[2 * key_dim..2 * key_dim + val_dim];

        // Repeat key heads to match value heads (num_v / num_k = 2).
        let rep = nvh / nkh;

        // g = -exp(A_log) * softplus(a + dt_bias); beta = sigmoid(b).
        let alog = self
            .v(&format!("layers.{}.linear_attn.A_log", layer))
            .cloned()
            .unwrap_or_default();
        let dtbias = self
            .v(&format!("layers.{}.linear_attn.dt_bias", layer))
            .cloned()
            .unwrap_or_default();
        let mut g = vec![0.0f32; nvh];
        let mut beta = vec![0.0f32; nvh];
        for h in 0..nvh {
            let ab = a.get(h).copied().unwrap_or(0.0) + dtbias.get(h).copied().unwrap_or(0.0);
            let sp = if ab > 20.0 { ab } else { (1.0 + ab.exp()).ln() };
            g[h] = -alog.get(h).copied().unwrap_or(0.0).exp() * sp;
            let bb = b.get(h).copied().unwrap_or(0.0);
            beta[h] = 1.0 / (1.0 + (-bb).exp());
        }

        // Recurrent state update (fp32), per value head.
        let st = &mut self.linear_states[state_idx];
        let mut o = vec![0.0f32; val_dim];
        for vh in 0..nvh {
            let kh = vh / rep;
            let qbase = kh * kd;
            let kbase = kh * kd;
            let vbase = vh * vd;
            let sbase = vh * kd * vd;

            // L2-normalize q and k; scale q by 1/sqrt(kd).
            let mut qh = vec![0.0f32; kd];
            let mut kh_vec = vec![0.0f32; kd];
            let qn = (q[qbase..qbase + kd].iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            let kn = (k[kbase..kbase + kd].iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            for d in 0..kd {
                qh[d] = q[qbase + d] / qn / (kd as f32).sqrt();
                kh_vec[d] = k[kbase + d] / kn;
            }

            // Decay state.
            let decay = g[vh].exp();
            for i in 0..kd * vd {
                st.state[sbase + i] *= decay;
            }

            // kv_mem[j] = sum_i S[i,j] * k[i].
            let mut mem = vec![0.0f32; vd];
            for (i, &kv) in kh_vec.iter().enumerate() {
                if kv == 0.0 {
                    continue;
                }
                let soff = sbase + i * vd;
                for j in 0..vd {
                    mem[j] += st.state[soff + j] * kv;
                }
            }

            // delta[j] = (v[j] - mem[j]) * beta.
            let bta = beta[vh];
            let mut delta = vec![0.0f32; vd];
            for j in 0..vd {
                delta[j] = (v[vbase + j] - mem[j]) * bta;
            }

            // S += k (x) delta.
            for (i, &kv) in kh_vec.iter().enumerate() {
                if kv == 0.0 {
                    continue;
                }
                let soff = sbase + i * vd;
                for j in 0..vd {
                    st.state[soff + j] += kv * delta[j];
                }
            }

            // o[j] = sum_i S[i,j] * q[i].
            for (i, &qv) in qh.iter().enumerate() {
                if qv == 0.0 {
                    continue;
                }
                let soff = sbase + i * vd;
                for j in 0..vd {
                    o[vbase + j] += st.state[soff + j] * qv;
                }
            }
        }

        // Gated RMSNorm: norm(o) * norm.weight * silu(z).
        let norm_w = self
            .v(&format!("layers.{}.linear_attn.norm.weight", layer))
            .cloned()
            .unwrap_or_default();
        let o_norm = Self::rms_norm_gated(&o, &z, &norm_w, cfg.rms_norm_eps, vd);

        // Output projection.
        let y = self.linear(&o_norm, &format!("layers.{}.linear_attn.out_proj", layer));
        if y.len() != cfg.hidden_size {
            return vec![0.0; cfg.hidden_size];
        }
        y
    }

    fn forward_token(&mut self, token: u32) -> Vec<f32> {
        let num_layers = self.config.num_layers;
        let hs = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;
        let layer_types = self.config.layer_types.clone();
        let intv = self.config.full_attention_interval;

        // Embedding lookup.
        let embed = match self.w("embed_tokens.weight") {
            Some(w) => w,
            None => return vec![0.0; vocab],
        };
        let mut x = vec![0.0f32; hs];
        if (token as usize) < embed.rows {
            x.copy_from_slice(embed.row(token as usize));
        }

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
                let y = self.forward_full_attn(i, &x, full_idx, pos);
                full_idx += 1;
                y
            } else {
                let y = self.forward_linear_attn(i, &x, linear_idx);
                linear_idx += 1;
                y
            };
            for (j, v) in attn.iter().enumerate() {
                x[j] += v;
            }

            // MLP.
            let norm_w = self
                .v(&format!("layers.{}.post_attention_layernorm.weight", i))
                .cloned()
                .unwrap_or_default();
            let xn = Self::rms_norm(&x, &norm_w, eps);
            let gate = self.linear(&xn, &format!("layers.{}.mlp.gate_proj", i));
            let up = self.linear(&xn, &format!("layers.{}.mlp.up_proj", i));
            let mut act = vec![0.0f32; gate.len()];
            for j in 0..gate.len().min(up.len()) {
                let g = gate[j];
                act[j] = (g / (1.0 + (-g).exp())) * up[j];
            }
            let mlp_out = self.linear(&act, &format!("layers.{}.mlp.down_proj", i));
            for (j, v) in mlp_out.iter().enumerate() {
                x[j] += v;
            }
        }

        self.position += 1;

        // Final norm + lm_head.
        let final_w = self.v("norm.weight").cloned().unwrap_or_default();
        let xf = Self::rms_norm(&x, &final_w, eps);
        let logits = self.linear(&xf, "lm_head.weight");
        if logits.len() != vocab {
            return vec![0.0; vocab];
        }
        logits
    }

    pub fn forward(&mut self, input_ids: &[u32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &t in input_ids {
            logits = self.forward_token(t);
        }
        logits
    }
}
