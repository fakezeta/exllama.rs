//! Device-resident Qwen3.5 decoder (built only with a `gpu` feature).
//!
//! This is a *separate* forward path from the CPU one in `model.rs`. All
//! weights, activations and recurrent state stay on the device for the whole
//! request; the only host transfers are the input token ids (at load time the
//! weights are uploaded once) and the final logits, copied back once per
//! `Model::forward` call. There is no per-layer host/device traffic.
//!
//! Weights are fp16 on device (codebook outputs are fp16 by construction),
//! activations are fp32 with a short fp16 cast around each GEMM.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use half::f16;

use crate::exl3::Exl3Model;
use crate::model::ModelConfig;

trait CandleExt<T> {
    fn ce(self) -> Result<T, String>;
}
impl<T> CandleExt<T> for candle_core::Result<T> {
    fn ce(self) -> Result<T, String> {
        self.map_err(|e| e.to_string())
    }
}

/// Candle's matmul requires *contiguous* operands on CUDA. `permute`,
/// `transpose` and `t` return strided views, so a CPU-device test can pass
/// while the real GPU rejects the same op with
/// "matmul is only supported for contiguous tensors". All runtime matmuls go
/// through here; `.contiguous()` is a cheap no-op for already-contiguous
/// tensors.
fn matmul_c(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    a.contiguous().ce()?.matmul(&b.contiguous().ce()?).ce()
}

/// Config values the device forward needs (copied out of `ModelConfig`).
#[derive(Clone)]
struct GpuCfg {
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    layer_types: Vec<String>,
    full_attention_interval: usize,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    rms_norm_eps: f32,
    context_size: usize,
    rope_half: usize,
}

/// Immutable device weights, shared by every instance (`Arc`).
pub(crate) struct GpuWeights {
    device: Device,
    prefix: String,
    cfg: GpuCfg,
    /// Plain matrices and fp16 embedding tables, both (out, in) F16.
    mats: HashMap<String, Tensor>,
    /// Quantized linears: (folded `B = (H_n @ W_hat^T @ H_k)^T` (n,k) F16,
    /// suh (k,) F32, svh (n,) F32). Both Hadamards are baked into `B` at load,
    /// so a decode step is one GEMV instead of three.
    qhat: HashMap<String, (Tensor, Tensor, Tensor)>,
    /// 1-D tensors (norms, conv weights, A_log, dt_bias), F32.
    vecs: HashMap<String, Tensor>,
    /// RoPE tables, (context, rope_half) F32.
    rope_cos: Tensor,
    rope_sin: Tensor,
    /// Normalized Sylvester H128, (128, 128) F32.
    h128: Tensor,
}

impl GpuWeights {
    /// Decode and upload every weight once. Nothing is uploaded during inference.
    pub(crate) fn build(
        ex: &Exl3Model,
        config: &ModelConfig,
        context_size: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        rope_half: usize,
        device: &Device,
    ) -> Result<Arc<Self>, String> {
        // Both attention paths map query heads onto KV heads by integer
        // division. A non-divisible count would index past the KV head array
        // in `narrow` (or silently group the wrong heads), so reject it up
        // front with a clear message.
        if config.num_key_value_heads == 0
            || !config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads)
        {
            return Err(format!(
                "num_attention_heads ({}) must be a multiple of num_key_value_heads ({})",
                config.num_attention_heads, config.num_key_value_heads
            ));
        }
        if config.linear_num_key_heads == 0
            || !config
                .linear_num_value_heads
                .is_multiple_of(config.linear_num_key_heads)
        {
            return Err(format!(
                "linear_num_value_heads ({}) must be a multiple of linear_num_key_heads ({})",
                config.linear_num_value_heads, config.linear_num_key_heads
            ));
        }
        let mut mats = HashMap::new();
        for (name, m) in &ex.weights {
            if m.data.len() != m.rows * m.cols {
                return Err(format!(
                    "{}: weight buffer has {} elements, expected {} ({}, {})",
                    name,
                    m.data.len(),
                    m.rows * m.cols,
                    m.rows,
                    m.cols
                ));
            }
            let vals: Vec<f16> = m.data.iter().map(|&v| f16::from_f32(v)).collect();
            let t = Tensor::from_slice(&vals, (m.rows, m.cols), device)
                .map_err(|e| format!("{}: upload matrix: {}", name, e))?;
            mats.insert(name.clone(), t);
        }
        for (name, emb) in &ex.embeddings_f16 {
            if emb.data.len() != emb.rows * emb.cols {
                return Err(format!(
                    "{}: embedding buffer has {} elements, expected {} ({}, {})",
                    name,
                    emb.data.len(),
                    emb.rows * emb.cols,
                    emb.rows,
                    emb.cols
                ));
            }
            let vals: Vec<f16> = emb.data.iter().map(|&b| f16::from_bits(b)).collect();
            let t = Tensor::from_slice(&vals, (emb.rows, emb.cols), device)
                .map_err(|e| format!("{}: upload embedding: {}", name, e))?;
            mats.insert(name.clone(), t);
        }
        let scale = 1.0 / (128.0f32).sqrt();
        let mut h = vec![0.0f32; 128 * 128];
        for i in 0..128 {
            for j in 0..128 {
                h[i * 128 + j] = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
            }
        }
        let h128 = Tensor::from_slice(&h, (128, 128), device).ce()?;
        // `upload_device` folds both Hadamards into W_hat on the host and
        // uploads only the final matrix, so the device never needs the
        // full-size intermediates a device-side fold would allocate.
        let mut qhat = HashMap::new();
        for (name, ql) in &ex.linears {
            let (w, suh, svh) = ql.upload_device(device)?;
            qhat.insert(name.clone(), (w, suh, svh));
        }
        let mut vecs = HashMap::new();
        for (name, v) in &ex.vectors {
            let t = Tensor::from_slice(v, (v.len(),), device).ce()?;
            vecs.insert(name.clone(), t);
        }
        let rope_cos_t = Tensor::from_slice(rope_cos, (context_size, rope_half), device).ce()?;
        let rope_sin_t = Tensor::from_slice(rope_sin, (context_size, rope_half), device).ce()?;
        eprintln!(
            "[gpu] weights uploaded: {} matrices, {} quantized linears, {} vectors",
            mats.len(),
            qhat.len(),
            vecs.len()
        );
        Ok(Arc::new(Self {
            device: device.clone(),
            prefix: ex.prefix.clone(),
            cfg: GpuCfg {
                num_layers: config.num_layers,
                hidden_size: config.hidden_size,
                vocab_size: config.vocab_size,
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                layer_types: config.layer_types.clone(),
                full_attention_interval: config.full_attention_interval,
                linear_conv_kernel_dim: config.linear_conv_kernel_dim,
                linear_key_head_dim: config.linear_key_head_dim,
                linear_num_key_heads: config.linear_num_key_heads,
                linear_num_value_heads: config.linear_num_value_heads,
                linear_value_head_dim: config.linear_value_head_dim,
                rms_norm_eps: config.rms_norm_eps,
                context_size,
                rope_half,
            },
            mats,
            qhat,
            vecs,
            rope_cos: rope_cos_t,
            rope_sin: rope_sin_t,
            h128,
        }))
    }

    fn candidates(&self, name: &str) -> Vec<String> {
        let mut out = vec![
            format!("{}{}", self.prefix, name),
            format!("{}{}.weight", self.prefix, name),
            name.to_string(),
            format!("{}.weight", name),
        ];
        if name == "lm_head.weight" {
            // EXL3 stores the (possibly quantized) head under the bare base
            // name. The embedding table is handled as a *last* resort in
            // `linear()` — putting it here would shadow an untied quantized
            // head (both have shape (vocab, hidden), so no error is raised).
            out.push("lm_head".to_string());
        }
        out
    }

    fn mat(&self, name: &str) -> Option<&Tensor> {
        self.candidates(name).iter().find_map(|k| self.mats.get(k))
    }

    fn q(&self, name: &str) -> Option<&(Tensor, Tensor, Tensor)> {
        self.candidates(name).iter().find_map(|k| self.qhat.get(k))
    }

    fn vec(&self, name: &str) -> Result<&Tensor, String> {
        let prefixed = format!("{}{}", self.prefix, name);
        self.vecs
            .get(&prefixed)
            .or_else(|| self.vecs.get(name))
            .ok_or_else(|| format!("missing vector tensor `{}`", name))
    }

    /// hadamard H128 applied independently to each 128-element block.
    fn hadamard(&self, v: &Tensor) -> Result<Tensor, String> {
        let len = v.elem_count();
        if len == 0 || !len.is_multiple_of(128) {
            return Err(format!(
                "Hadamard needs a multiple of 128 elements, got {}",
                len
            ));
        }
        let blocks = len / 128;
        let out = matmul_c(&v.reshape((blocks, 128)).ce()?, &self.h128)?;
        out.flatten_all().ce()
    }

    /// `y = W @ x`, W stored (out, in) F16. The GEMM is done on device.
    fn matmul_w(w: &Tensor, x: &Tensor) -> Result<Tensor, String> {
        let k = x.dim(0).ce()?;
        let in_dim = w.dim(1).ce()?;
        if in_dim != k {
            return Err(format!(
                "matmul: weight input dim {} does not match input length {}",
                in_dim, k
            ));
        }
        let x16 = x.to_dtype(DType::F16).ce()?.reshape((k, 1)).ce()?;
        let y = matmul_c(w, &x16)?;
        y.to_dtype(DType::F32).ce()?.flatten_all().ce()
    }

    /// One EXL3 quantized linear, entirely on device. Both Hadamards are
    /// already folded into `w` at load time, so this is
    /// `y = svh * (B @ (x * suh))` — a single GEMV.
    fn q_matmul(&self, parts: &(Tensor, Tensor, Tensor), x: &Tensor) -> Result<Tensor, String> {
        let (w, suh, svh) = parts;
        if x.elem_count() != suh.elem_count() {
            return Err(format!(
                "quantized linear input has {} elements but suh has {}",
                x.elem_count(),
                suh.elem_count()
            ));
        }
        let xr = x.mul(suh).ce()?;
        let t = Self::matmul_w(w, &xr)?;
        t.mul(svh).ce()
    }

    /// Resolve the embedding table (shared by the token lookup and the tied
    /// lm_head fallback).
    fn embedding_table(&self) -> Option<&Tensor> {
        let name = format!("{}embed_tokens.weight", self.prefix);
        self.mats
            .get(&name)
            .or_else(|| self.mats.get("model.embed_tokens.weight"))
    }

    fn linear(&self, name: &str, x: &Tensor) -> Result<Tensor, String> {
        if let Some(w) = self.mat(name) {
            return Self::matmul_w(w, x);
        }
        if let Some(q) = self.q(name) {
            return self.q_matmul(q, x);
        }
        // Tied checkpoints have no separate lm_head: the embedding table is
        // the output projection. Only try it after the plain and quantized
        // heads so an untied quantized lm_head is never shadowed.
        if name == "lm_head.weight" {
            if let Some(w) = self.embedding_table() {
                return Self::matmul_w(w, x);
            }
        }
        Err(format!("missing linear weight `{}`", name))
    }

    fn embedding(&self, token: u32) -> Result<Tensor, String> {
        let table = self
            .embedding_table()
            .ok_or_else(|| "missing embedding tensor `embed_tokens.weight`".to_string())?;
        let row = table.get(token as usize).ce()?;
        row.to_dtype(DType::F32).ce()
    }

    fn rms_norm(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, String> {
        // Reduce over the *last* dim. For a 1-D `(hidden,)` vector that is the
        // whole vector; for the 2-D `(heads, head_dim)` q/k norms it must be
        // one scale *per head* — reducing over dim 0 instead would share a
        // scale across heads and diverge from CPU `Model::rms_norm` (which
        // normalizes each q/k head independently).
        let rank = x.rank();
        let expected_w = if rank == 2 {
            x.dim(1).ce()?
        } else {
            x.elem_count()
        };
        if w.elem_count() != expected_w {
            return Err(format!(
                "RMSNorm weight has {} elements, expected {} (input rank {})",
                w.elem_count(),
                expected_w,
                rank
            ));
        }
        let ss = x.sqr().ce()?.mean_keepdim(rank - 1).ce()?;
        let scale = ss
            .affine(1.0, self.cfg.rms_norm_eps as f64)
            .ce()?
            .sqrt()
            .ce()?
            .recip()
            .ce()?;
        let w1 = w.affine(1.0, 1.0).ce()?;
        // `w` is 1-D (head_dim,). `x` can be 1-D (hidden,) or 2-D
        // (heads, head_dim) for q/k norms — candle's `.mul()` needs *exact*
        // shapes, so use `.broadcast_mul()`.
        x.broadcast_mul(&scale).ce()?.broadcast_mul(&w1).ce()
    }

    fn rope(
        &self,
        x: &Tensor,
        heads: usize,
        head_len: usize,
        pos: usize,
    ) -> Result<Tensor, String> {
        let half = self.cfg.rope_half;
        if half == 0 || pos >= self.cfg.context_size {
            return Ok(x.clone());
        }
        let x2 = x.reshape((heads, head_len)).ce()?;
        let rotary = half * 2;
        if head_len < rotary {
            return Err(format!(
                "partial rotary dim {} exceeds head_dim {}",
                rotary, head_len
            ));
        }
        // Only the first `rotary` dims are rotated — the rest of each head is
        // passed through unchanged (CPU `Model::apply_rope`). Dropping the tail
        // silently would shrink q/k and break the KV cache write.
        let a = x2.narrow(1, 0, half).ce()?;
        let b = x2.narrow(1, half, half).ce()?;
        // Tables are (context, rope_half) row-major: one *row* per position.
        let c = self
            .rope_cos
            .narrow(0, pos, 1)
            .ce()?
            .reshape((1, half))
            .ce()?;
        let s = self
            .rope_sin
            .narrow(0, pos, 1)
            .ce()?
            .reshape((1, half))
            .ce()?;
        let na = a
            .broadcast_mul(&c)
            .ce()?
            .sub(&b.broadcast_mul(&s).ce()?)
            .ce()?;
        let nb = b
            .broadcast_mul(&c)
            .ce()?
            .add(&a.broadcast_mul(&s).ce()?)
            .ce()?;
        let rotated = Tensor::cat(&[na, nb], 1).ce()?;
        let out = if head_len > rotary {
            let tail = x2.narrow(1, rotary, head_len - rotary).ce()?;
            Tensor::cat(&[rotated, tail], 1).ce()?
        } else {
            rotated
        };
        out.flatten_all().ce()
    }

    fn is_full(&self, layer: usize) -> bool {
        if layer < self.cfg.layer_types.len() {
            self.cfg.layer_types[layer] == "full_attention"
        } else {
            (layer + 1).is_multiple_of(self.cfg.full_attention_interval)
        }
    }
}

/// Mutable per-instance device state.
pub(crate) struct GpuState {
    position: usize,
    /// (context, nkv * hd), fp16 bits.
    full: Vec<(Tensor, Tensor)>,
    /// conv (hist, conv_dim) and recurrent state (nvh, kd, vd), fp32.
    linear: Vec<(Tensor, usize, Tensor)>,
    /// Logits from the most recent forward, kept on device for sampling.
    logits: Option<Tensor>,
}

/// Compact post-prefill snapshot (only the used KV prefix).
pub(crate) struct GpuSnapshot {
    position: usize,
    full: Vec<(Tensor, Tensor)>,
    linear: Vec<(Tensor, usize, Tensor)>,
    /// Post-prefill logits, so `n > 1` choices can sample without a host copy.
    logits: Option<Tensor>,
}

impl GpuState {
    fn new(w: &GpuWeights) -> Result<Self, String> {
        let cfg = &w.cfg;
        let mut full = Vec::new();
        let mut linear = Vec::new();
        for i in 0..cfg.num_layers {
            if w.is_full(i) {
                let shape = (cfg.context_size, cfg.num_key_value_heads * cfg.head_dim);
                full.push((
                    Tensor::zeros(shape, DType::F16, &w.device).ce()?,
                    Tensor::zeros(shape, DType::F16, &w.device).ce()?,
                ));
            } else {
                let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
                let val_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
                let conv_dim = key_dim * 2 + val_dim;
                let hist = cfg.linear_conv_kernel_dim - 1;
                linear.push((
                    Tensor::zeros((hist, conv_dim), DType::F32, &w.device).ce()?,
                    0,
                    Tensor::zeros(
                        (
                            cfg.linear_num_value_heads,
                            cfg.linear_key_head_dim,
                            cfg.linear_value_head_dim,
                        ),
                        DType::F32,
                        &w.device,
                    )
                    .ce()?,
                ));
            }
        }
        Ok(Self {
            position: 0,
            full,
            linear,
            logits: None,
        })
    }

    fn snapshot(&self) -> Result<GpuSnapshot, String> {
        let used = self.position;
        Ok(GpuSnapshot {
            position: self.position,
            full: self
                .full
                .iter()
                .map(|(k, v)| {
                    Ok((
                        k.narrow(0, 0, used).ce()?.copy().ce()?,
                        v.narrow(0, 0, used).ce()?.copy().ce()?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?,
            linear: self
                .linear
                .iter()
                .map(|(c, p, s)| Ok((c.copy().ce()?, *p, s.copy().ce()?)))
                .collect::<Result<Vec<_>, String>>()?,
            logits: self.logits.as_ref().map(|t| t.copy().expect("copy logits")),
        })
    }

    fn restore(&mut self, w: &GpuWeights, snap: &GpuSnapshot) -> Result<(), String> {
        self.position = snap.position;
        for ((k, v), (sk, sv)) in self.full.iter_mut().zip(&snap.full) {
            let zeros_k = Tensor::zeros(k.shape(), DType::F16, &w.device).ce()?;
            let zeros_v = Tensor::zeros(v.shape(), DType::F16, &w.device).ce()?;
            *k = zeros_k.slice_scatter(sk, 0, 0).ce()?;
            *v = zeros_v.slice_scatter(sv, 0, 0).ce()?;
        }
        for ((c, p, s), (sc, sp, ss)) in self.linear.iter_mut().zip(&snap.linear) {
            *c = Tensor::zeros(c.shape(), DType::F32, &w.device)
                .ce()?
                .slice_scatter(sc, 0, 0)
                .ce()?;
            *p = *sp;
            *s = Tensor::zeros(s.shape(), DType::F32, &w.device)
                .ce()?
                .slice_scatter(ss, 0, 0)
                .ce()?;
        }
        self.logits = snap
            .logits
            .as_ref()
            .map(|t| t.copy().expect("restore logits"));
        Ok(())
    }
}

/// A device-resident runner: shared immutable weights + per-instance state.
pub(crate) struct GpuRunner {
    weights: Arc<GpuWeights>,
    state: GpuState,
}

impl GpuRunner {
    pub(crate) fn new(weights: Arc<GpuWeights>) -> Result<Self, String> {
        let state = GpuState::new(&weights)?;
        Ok(Self { weights, state })
    }

    pub(crate) fn spawn(&self) -> Result<Self, String> {
        Self::new(Arc::clone(&self.weights))
    }

    pub(crate) fn reset(&mut self) -> Result<(), String> {
        self.state = GpuState::new(&self.weights)?;
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> Result<GpuSnapshot, String> {
        self.state.snapshot()
    }

    pub(crate) fn restore(&mut self, snap: &GpuSnapshot) -> Result<(), String> {
        self.state.restore(&self.weights, snap)
    }

    pub(crate) fn position(&self) -> usize {
        self.state.position
    }

    /// Run the tokens on device, keeping the resulting logits on device.
    /// Callers that sample on device use this so only a token id crosses back.
    pub(crate) fn forward_logits(&mut self, ids: &[u32]) -> Result<(), String> {
        // Reject an overlong prompt before mutating any state. (The CPU
        // wrapper checks this too, but the guard here keeps direct callers
        // from leaving a half-written KV/conv state behind on error.)
        let room = self
            .weights
            .cfg
            .context_size
            .saturating_sub(self.state.position);
        if ids.len() > room {
            return Err(format!(
                "context size exceeded: position {} + {} input tokens > context {}",
                self.state.position,
                ids.len(),
                self.weights.cfg.context_size
            ));
        }
        if ids.is_empty() {
            self.state.logits = Some(
                Tensor::zeros(
                    (self.weights.cfg.vocab_size,),
                    DType::F32,
                    &self.weights.device,
                )
                .ce()?,
            );
            return Ok(());
        }
        let mut logits = Tensor::zeros(
            (self.weights.cfg.vocab_size,),
            DType::F32,
            &self.weights.device,
        )
        .ce()?;
        for &t in ids {
            logits = self.forward_token(t)?;
        }
        self.state.logits = Some(logits);
        Ok(())
    }

    /// Run the tokens on device and copy back the final logits (host-sampling
    /// path and tests).
    pub(crate) fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        self.forward_logits(ids)?;
        self.state
            .logits
            .as_ref()
            .ok_or_else(|| "no logits after forward".to_string())?
            .to_vec1::<f32>()
            .ce()
    }

    pub(crate) fn has_logits(&self) -> bool {
        self.state.logits.is_some()
    }

    fn forward_full_attn(
        &mut self,
        layer: usize,
        x: &Tensor,
        idx: usize,
    ) -> Result<Tensor, String> {
        let w = &self.weights;
        let cfg = &w.cfg;
        let hd = cfg.head_dim;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let norm_w = w.vec(&format!("layers.{}.input_layernorm.weight", layer))?;
        let xn = w.rms_norm(x, norm_w)?;
        let qp = w.linear(&format!("layers.{}.self_attn.q_proj", layer), &xn)?;
        let kp = w.linear(&format!("layers.{}.self_attn.k_proj", layer), &xn)?;
        let vp = w.linear(&format!("layers.{}.self_attn.v_proj", layer), &xn)?;
        if qp.elem_count() != 2 * nh * hd
            || kp.elem_count() != nkv * hd
            || vp.elem_count() != nkv * hd
        {
            return Err(format!(
                "layer {}: q/k/v projection sizes are {}/{}/{} (expected {}/{}/{})",
                layer,
                qp.elem_count(),
                kp.elem_count(),
                vp.elem_count(),
                2 * nh * hd,
                nkv * hd,
                nkv * hd
            ));
        }

        // Per head the projection is [query(hd) | gate(hd)], i.e. (nh, hd*2)
        // — NOT (nh, 2, hd). Matches CPU `split_q_gate`.
        let qp2 = qp.reshape((nh, hd * 2)).ce()?;
        let q = qp2.narrow(1, 0, hd).ce()?.flatten_all().ce()?;
        let gate = qp2.narrow(1, hd, hd).ce()?.flatten_all().ce()?;
        if q.elem_count() != nh * hd || gate.elem_count() != nh * hd {
            return Err(format!(
                "layer {}: split q/gate have {}/{} elements, expected {} each",
                layer,
                q.elem_count(),
                gate.elem_count(),
                nh * hd
            ));
        }

        let qn_w = w.vec(&format!("layers.{}.self_attn.q_norm.weight", layer))?;
        let kn_w = w.vec(&format!("layers.{}.self_attn.k_norm.weight", layer))?;
        let qn = w
            .rms_norm(&q.reshape((nh, hd)).ce()?, qn_w)?
            .flatten_all()
            .ce()?;
        let kn = w
            .rms_norm(&kp.reshape((nkv, hd)).ce()?, kn_w)?
            .flatten_all()
            .ce()?;
        let pos = self.state.position;
        let q = w.rope(&qn, nh, hd, pos)?;
        let k = w.rope(&kn, nkv, hd, pos)?;

        // Write this position's K/V row into the fp16 cache.
        let kv = nkv * hd;
        let (kc, vc) = &mut self.state.full[idx];
        if q.elem_count() != nh * hd || k.elem_count() != kv {
            return Err(format!(
                "layer {}: RoPE output sizes are q={} (want {}) and k={} (want nkv*head_dim={}; nkv={}, head_dim={}, rope_half={})",
                layer,
                q.elem_count(),
                nh * hd,
                k.elem_count(),
                kv,
                nkv,
                hd,
                w.cfg.rope_half
            ));
        }
        let krow = k.reshape((1, kv)).ce()?.to_dtype(DType::F16).ce()?;
        let vrow = vp
            .narrow(0, 0, kv)
            .ce()?
            .reshape((1, kv))
            .ce()?
            .to_dtype(DType::F16)
            .ce()?;
        *kc = kc.slice_scatter(&krow, 0, pos).ce()?;
        *vc = vc.slice_scatter(&vrow, 0, pos).ce()?;

        let attn = gqa_attention(kc, vc, &q, &gate, pos + 1, nh, nkv, hd)?;
        let y = w.linear(&format!("layers.{}.self_attn.o_proj", layer), &attn)?;
        if y.dim(0).ce()? != cfg.hidden_size {
            return Err(format!("layer {}: unexpected o_proj shape", layer));
        }
        Ok(y)
    }

    fn forward_linear_attn(&mut self, layer: usize, x: &Tensor) -> Result<Tensor, String> {
        let w = &self.weights;
        let cfg = &w.cfg;
        let eps = cfg.rms_norm_eps;
        let norm_w = w.vec(&format!("layers.{}.input_layernorm.weight", layer))?;
        let xn = w.rms_norm(x, norm_w)?;
        let qkv = w.linear(&format!("layers.{}.linear_attn.in_proj_qkv", layer), &xn)?;
        let z = w.linear(&format!("layers.{}.linear_attn.in_proj_z", layer), &xn)?;
        let b = w.linear(&format!("layers.{}.linear_attn.in_proj_b", layer), &xn)?;
        let a = w.linear(&format!("layers.{}.linear_attn.in_proj_a", layer), &xn)?;

        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let nkh = cfg.linear_num_key_heads;
        let nvh = cfg.linear_num_value_heads;
        let key_dim = kd * nkh;
        let val_dim = vd * nvh;
        let conv_dim = key_dim * 2 + val_dim;
        let kernel = cfg.linear_conv_kernel_dim;
        let cw = w.vec(&format!("layers.{}.linear_attn.conv1d.weight", layer))?;
        if cw.elem_count() != conv_dim * kernel {
            return Err(format!(
                "layer {}: conv1d weight has {} elements, expected {} (conv_dim={}, kernel={})",
                layer,
                cw.elem_count(),
                conv_dim * kernel,
                conv_dim,
                kernel
            ));
        }
        let cw = cw.reshape((conv_dim, kernel)).ce()?;

        let (conv, conv_pos, sstate) = &mut self.state.linear[layer_idx(w, layer, false)];
        let hist = kernel - 1;
        let mut window_parts: Vec<Tensor> = Vec::with_capacity(hist + 1);
        for j in 0..hist {
            let idx = (*conv_pos + j) % hist;
            window_parts.push(conv.narrow(0, idx, 1).ce()?.copy().ce()?);
        }
        window_parts.push(
            qkv.narrow(0, 0, conv_dim)
                .ce()?
                .reshape((1, conv_dim))
                .ce()?,
        );
        let window = Tensor::cat(&window_parts, 0).ce()?;
        *conv = conv
            .slice_scatter(
                &qkv.narrow(0, 0, conv_dim)
                    .ce()?
                    .reshape((1, conv_dim))
                    .ce()?,
                0,
                *conv_pos,
            )
            .ce()?;
        *conv_pos = (*conv_pos + 1) % hist;
        let conv_out = window.t().ce()?.mul(&cw).ce()?.sum(1).ce()?.silu().ce()?;

        let q = conv_out.narrow(0, 0, key_dim).ce()?;
        let k = conv_out.narrow(0, key_dim, key_dim).ce()?;
        let v = conv_out.narrow(0, 2 * key_dim, val_dim).ce()?;

        let alog = w.vec(&format!("layers.{}.linear_attn.A_log", layer))?;
        let dtbias = w.vec(&format!("layers.{}.linear_attn.dt_bias", layer))?;
        if a.elem_count() != nvh
            || b.elem_count() != nvh
            || alog.elem_count() != nvh
            || dtbias.elem_count() != nvh
        {
            return Err(format!(
                "layer {}: in_proj_a/b, A_log, dt_bias must each have {} elements (got {}/{}/{}/{})",
                layer,
                nvh,
                a.elem_count(),
                b.elem_count(),
                alog.elem_count(),
                dtbias.elem_count()
            ));
        }
        let ab = a.add(dtbias).ce()?;
        // Stable softplus: max(x,0) + log(1 + exp(-|x|)).
        let softplus = ab
            .abs()
            .ce()?
            .neg()
            .ce()?
            .exp()
            .ce()?
            .affine(1.0, 1.0)
            .ce()?
            .log()
            .ce()?
            .add(&ab.clamp(0.0, f64::INFINITY).ce()?)
            .ce()?;
        let g = alog.exp().ce()?.neg().ce()?.mul(&softplus).ce()?;
        let beta = b
            .neg()
            .ce()?
            .exp()
            .ce()?
            .affine(1.0, 1.0)
            .ce()?
            .recip()
            .ce()?;

        // Recurrent update for *all* value heads in one batched device op.
        // The old loop pulled `g[vh]`/`beta[vh]` to the host with `to_vec0()`
        // for every head — nvh (32) * layers (18) full device synchronizations
        // per token — and launched ~15 kernels per head. Keeping the decay and
        // gating factors on device removes both the syncs and most launches.
        let (new_state, o_all) =
            deltanet_step_all(sstate, &q, &k, &v, &g, &beta, nkh, nvh, kd, vd)?;
        *sstate = new_state;

        let norm_w2 = w.vec(&format!("layers.{}.linear_attn.norm.weight", layer))?;
        if norm_w2.elem_count() != vd || z.elem_count() != nvh * vd {
            return Err(format!(
                "layer {}: norm.weight has {} elements (expected {}) and in_proj_z has {} (expected {})",
                layer,
                norm_w2.elem_count(),
                vd,
                z.elem_count(),
                nvh * vd
            ));
        }
        let o2 = o_all.reshape((nvh, vd)).ce()?;
        let ss = o2.sqr().ce()?.sum_keepdim(1).ce()?;
        let scale = ss
            .affine(1.0 / vd as f64, 0.0)
            .ce()?
            .affine(1.0, eps as f64)
            .ce()?
            .sqrt()
            .ce()?
            .recip()
            .ce()?;
        let nw = norm_w2.reshape((1, vd)).ce()?;
        let z2 = z.reshape((nvh, vd)).ce()?;
        let o_norm = o2
            .broadcast_mul(&scale)
            .ce()?
            // `nw` is (1, vd) and `z2` is (nvh, vd); `.mul()` requires exact
            // shapes in candle, so broadcast both against (nvh, vd).
            .broadcast_mul(&nw)
            .ce()?
            .broadcast_mul(&z2.silu().ce()?)
            .ce()?
            .flatten_all()
            .ce()?;
        let y = w.linear(&format!("layers.{}.linear_attn.out_proj", layer), &o_norm)?;
        if y.dim(0).ce()? != cfg.hidden_size {
            return Err(format!("layer {}: unexpected out_proj shape", layer));
        }
        Ok(y)
    }

    fn forward_token(&mut self, token: u32) -> Result<Tensor, String> {
        let weights = Arc::clone(&self.weights);
        let w = weights.as_ref();
        let cfg = &w.cfg;
        let pos = self.state.position;
        if pos >= cfg.context_size {
            return Err("context size exceeded".to_string());
        }
        let mut x = w.embedding(token)?;
        let mut full_idx = 0usize;
        for i in 0..cfg.num_layers {
            let (attn, next_full) = if w.is_full(i) {
                (self.forward_full_attn(i, &x, full_idx)?, full_idx + 1)
            } else {
                (self.forward_linear_attn(i, &x)?, full_idx)
            };
            full_idx = next_full;
            x = x.add(&attn).ce()?;
            let norm_w = w.vec(&format!("layers.{}.post_attention_layernorm.weight", i))?;
            let xn = w.rms_norm(&x, norm_w)?;
            let gate = w.linear(&format!("layers.{}.mlp.gate_proj", i), &xn)?;
            let up = w.linear(&format!("layers.{}.mlp.up_proj", i), &xn)?;
            let act = gate.silu().ce()?.mul(&up).ce()?;
            let mlp = w.linear(&format!("layers.{}.mlp.down_proj", i), &act)?;
            x = x.add(&mlp).ce()?;
        }
        self.state.position += 1;
        let final_w = w.vec("norm.weight")?;
        let xf = w.rms_norm(&x, final_w)?;
        let logits = w.linear("lm_head.weight", &xf)?;
        if logits.dim(0).ce()? != cfg.vocab_size {
            return Err(format!(
                "lm_head produced {} logits, expected {}",
                logits.dim(0).ce()?,
                cfg.vocab_size
            ));
        }
        Ok(logits)
    }
}

/// Sampling parameters mirrored from `server::Params` for the device path.
#[derive(Clone, Debug)]
pub(crate) struct SampleSpec {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    /// How many top logprobs to return (0 = none).
    pub top_n: usize,
}

/// Compact result of one device sampling step. Only a token id, its logprob
/// and a handful of top alternatives cross back to the host — never the full
/// vocabulary of logits.
#[derive(Clone, Debug)]
pub(crate) struct DeviceSample {
    pub token: u32,
    pub logprob: f32,
    /// Descending (token id, logprob) pairs, empty unless `top_n > 0`.
    pub top: Vec<(u32, f32)>,
}

impl GpuRunner {
    /// Sample from the logits produced by the last `forward_logits` call,
    /// entirely on device. `u` is a host-provided uniform in `[0, 1)` from the
    /// request RNG (one f32, instead of a vocabulary-sized logits copy).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(
        &self,
        spec: &SampleSpec,
        history: &[u32],
        u: f32,
    ) -> Result<DeviceSample, String> {
        let dev = &self.weights.device;
        let vocab = self.weights.cfg.vocab_size;
        let logits = self
            .state
            .logits
            .as_ref()
            .ok_or_else(|| "device sample called before any forward".to_string())?;
        if logits.elem_count() != vocab {
            return Err(format!(
                "device sample: logits have {} elements, expected vocab {}",
                logits.elem_count(),
                vocab
            ));
        }
        let to_i64 = |ids: &[u32]| -> Vec<i64> { ids.iter().map(|&t| t as i64).collect() };
        // The host `prepare_logits` silently ignores ids outside the vocab;
        // device `scatter` would not, so filter them to the same behavior.
        let history: Vec<u32> = history
            .iter()
            .copied()
            .filter(|&t| (t as usize) < vocab)
            .collect();
        let logit_bias: Vec<(u32, f32)> = spec
            .logit_bias
            .iter()
            .copied()
            .filter(|(id, _)| (*id as usize) < vocab)
            .collect();

        // ---- logit processors (mirrors server::prepare_logits) ----
        let need_penalty = (spec.repetition_penalty > 0.0 && spec.repetition_penalty != 1.0)
            || spec.presence_penalty != 0.0
            || spec.frequency_penalty != 0.0;
        let mut p = logits.clone();
        if (need_penalty && !history.is_empty()) || !logit_bias.is_empty() {
            let zero =
                || -> Result<Tensor, String> { Tensor::zeros((vocab,), DType::F32, dev).ce() };
            if need_penalty && !history.is_empty() {
                let idx = Tensor::from_slice(&to_i64(&history), (history.len(),), dev).ce()?;
                let ones = Tensor::ones((history.len(),), DType::F32, dev).ce()?;
                let seen = zero()?.scatter(&idx, &ones, 0).ce()?;
                if spec.repetition_penalty > 0.0 && spec.repetition_penalty != 1.0 {
                    let rep = spec.repetition_penalty;
                    // f = positive ? 1/rep : rep, applied only where seen.
                    let pos = logits.gt(0.0).ce()?.to_dtype(DType::F32).ce()?;
                    let neg = pos.affine(-1.0, 1.0).ce()?;
                    let f = pos
                        .affine((1.0 / rep) as f64, 0.0)
                        .ce()?
                        .broadcast_add(&neg.affine(rep as f64, 0.0).ce()?)
                        .ce()?;
                    // f - 1 (affine is self*scale + shift: use 1/-1 here,
                    // affine(-1, 1) would give 1 - f and invert the penalty).
                    let mult = f
                        .affine(1.0, -1.0)
                        .ce()?
                        .broadcast_mul(&seen)
                        .ce()?
                        .affine(1.0, 1.0)
                        .ce()?;
                    p = p.broadcast_mul(&mult).ce()?;
                }
                if spec.presence_penalty != 0.0 || spec.frequency_penalty != 0.0 {
                    let ones = Tensor::ones((history.len(),), DType::F32, dev).ce()?;
                    let counts = zero()?.scatter_add(&idx, &ones, 0).ce()?;
                    let add = seen.affine(-spec.presence_penalty as f64, 0.0).ce()?;
                    let add = if spec.frequency_penalty != 0.0 {
                        add.broadcast_sub(&counts.affine(spec.frequency_penalty as f64, 0.0).ce()?)
                            .ce()?
                    } else {
                        add
                    };
                    p = p.broadcast_add(&add).ce()?;
                }
            }
            if !logit_bias.is_empty() {
                let ids: Vec<u32> = logit_bias.iter().map(|&(id, _)| id).collect();
                let vals: Vec<f32> = logit_bias.iter().map(|&(_, b)| b).collect();
                let idx = Tensor::from_slice(&to_i64(&ids), (ids.len(),), dev).ce()?;
                let b = Tensor::from_slice(&vals, (vals.len(),), dev).ce()?;
                let bias = zero()?.scatter(&idx, &b, 0).ce()?;
                p = p.broadcast_add(&bias).ce()?;
            }
        }

        // logprob(v) = v - max - ln(sum(exp(v - max))), over the *adjusted*
        // distribution before temperature / top-k / top-p (vLLM convention).
        let max_p = p.max(0).ce()?;
        let sum_p = p.broadcast_sub(&max_p).ce()?.exp().ce()?.sum(0).ce()?;
        let lognorm = sum_p.log().ce()?.broadcast_add(&max_p).ce()?;

        // Top logprobs: repeated argmax on device (candle 0.11 has no topk).
        // Each pass masks the previously selected logit to -inf.
        let mut top_ids: Vec<Tensor> = Vec::new();
        let mut top_vals: Vec<Tensor> = Vec::new();
        if spec.top_n > 0 {
            let mut masked = p.clone();
            for _ in 0..spec.top_n.min(vocab) {
                let id = masked
                    .argmax(0)
                    .ce()?
                    .to_dtype(DType::I64)
                    .ce()?
                    .reshape((1,))
                    .ce()?;
                let val = masked.gather(&id, 0).ce()?;
                masked = masked
                    .scatter(
                        &id,
                        &Tensor::from_slice(&[f32::NEG_INFINITY], (1,), dev).ce()?,
                        0,
                    )
                    .ce()?;
                top_ids.push(id);
                top_vals.push(val);
            }
        }

        // ---- sampling transforms ----
        let temp = spec.temperature;
        let token_id = if temp <= 0.0 {
            p.argmax(0).ce()?.to_dtype(DType::I64).ce()?
        } else {
            let scores = p.affine((1.0 / temp) as f64, 0.0).ce()?;
            let smax = scores.max(0).ce()?;
            let w = scores.broadcast_sub(&smax).ce()?.exp().ce()?;
            let top_k_on = spec.top_k >= 1 && spec.top_k < vocab;
            let mut sel = if top_k_on {
                Tensor::zeros((vocab,), DType::F32, dev).ce()?
            } else {
                Tensor::ones((vocab,), DType::F32, dev).ce()?
            };
            if top_k_on {
                let mut masked = scores.clone();
                let mut ids = Vec::with_capacity(spec.top_k);
                for _ in 0..spec.top_k {
                    let id = masked
                        .argmax(0)
                        .ce()?
                        .to_dtype(DType::I64)
                        .ce()?
                        .reshape((1,))
                        .ce()?;
                    ids.push(id.clone());
                    masked = masked
                        .scatter(
                            &id,
                            &Tensor::from_slice(&[f32::NEG_INFINITY], (1,), dev).ce()?,
                            0,
                        )
                        .ce()?;
                }
                let idx = Tensor::cat(&ids, 0).ce()?;
                sel = sel
                    .scatter(&idx, &Tensor::ones((spec.top_k,), DType::F32, dev).ce()?, 0)
                    .ce()?;
            }
            if spec.top_p > 0.0 && spec.top_p < 1.0 {
                // Nucleus = smallest set of highest-logit tokens whose mass is
                // >= top_p. exp is monotonic, so a threshold search on the
                // logit value finds exactly the sorted-prefix the host sampler
                // uses, without needing a sort (candle 0.11 has no topk/sort).
                // Mass is measured over the candidates already kept by top-k,
                // matching the host sampler's top-k then top-p order.
                let target = w
                    .broadcast_mul(&sel)
                    .ce()?
                    .sum(0)
                    .ce()?
                    .affine(spec.top_p as f64, 0.0)
                    .ce()?;
                let mut lo = smax
                    .broadcast_add(&Tensor::from_slice(&[-40.0f32], (1,), dev).ce()?)
                    .ce()?;
                let mut hi = smax.clone();
                for _ in 0..32 {
                    let mid = lo.broadcast_add(&hi).ce()?.affine(0.5, 0.0).ce()?;
                    let sel_mid = scores
                        .ge(&mid.broadcast_as((vocab,)).ce()?)
                        .ce()?
                        .to_dtype(DType::F32)
                        .ce()?;
                    let mass = w
                        .broadcast_mul(&sel_mid)
                        .ce()?
                        .broadcast_mul(&sel)
                        .ce()?
                        .sum(0)
                        .ce()?;
                    let take = mass.ge(&target).ce()?.to_dtype(DType::F32).ce()?;
                    let keep = take.affine(-1.0, 1.0).ce()?;
                    lo = mid
                        .broadcast_mul(&take)
                        .ce()?
                        .broadcast_add(&lo.broadcast_mul(&keep).ce()?)
                        .ce()?;
                    hi = mid
                        .broadcast_mul(&keep)
                        .ce()?
                        .broadcast_add(&hi.broadcast_mul(&take).ce()?)
                        .ce()?;
                }
                let nucleus = scores
                    .ge(&lo.broadcast_as((vocab,)).ce()?)
                    .ce()?
                    .to_dtype(DType::F32)
                    .ce()?;
                sel = sel.broadcast_mul(&nucleus).ce()?;
            }
            let ws = w.broadcast_mul(&sel).ce()?;
            let cdf = ws.cumsum(0).ce()?;
            let total = cdf.get(vocab - 1).ce()?;
            let target = total.affine(u as f64, 0.0).ce()?;
            cdf.ge(&target.broadcast_as((vocab,)).ce()?)
                .ce()?
                .to_dtype(DType::F32)
                .ce()?
                .argmax(0)
                .ce()?
                .to_dtype(DType::I64)
                .ce()?
        };
        let token1 = token_id.reshape((1,)).ce()?;
        let chosen = p.gather(&token1, 0).ce()?;
        let lp = chosen.broadcast_sub(&lognorm).ce()?;

        // One small host transfer: token id, its logprob, and top pairs.
        let mut parts = vec![
            token_id.to_dtype(DType::F32).ce()?.reshape((1,)).ce()?,
            lp.reshape((1,)).ce()?,
        ];
        for (id, value) in top_ids.iter().zip(&top_vals) {
            parts.push(id.to_dtype(DType::F32).ce()?.reshape((1,)).ce()?);
            parts.push(value.broadcast_sub(&lognorm).ce()?.reshape((1,)).ce()?);
        }
        let out = Tensor::cat(&parts, 0).ce()?.to_vec1::<f32>().ce()?;
        let token = out[0] as u32;
        let logprob = out[1];
        let top = (2..out.len())
            .step_by(2)
            .map(|i| (out[i] as u32, out[i + 1]))
            .collect();
        Ok(DeviceSample {
            token,
            logprob,
            top,
        })
    }
}

/// Map a layer number to its index in the linear-attention state list.
fn layer_idx(w: &GpuWeights, layer: usize, _is_full: bool) -> usize {
    (0..layer).filter(|&i| !w.is_full(i)).count()
}

/// Fold both 128-block Hadamards into an EXL3 W_hat *at load time*, so each
/// decode step needs one GEMV instead of two Hadamard GEMMs plus the W_hat GEMV.
///
/// `w` is W_hat `(n, k)` F16, `h128` the normalized Sylvester H128 F32. Returns
/// `B = H_n @ W_hat @ H_k`, also `(n, k)` F16. `H_n` is applied to each 128-row
/// block and `H_k` to each 128-column block (both H matrices are symmetric and
/// orthogonal). `(n, k)` is preserved — transposing it silently breaks every
/// non-square quantized linear with an opaque `matmul` dimension error.
pub(crate) fn fold_hadamards(w: &Tensor, h128: &Tensor) -> Result<Tensor, String> {
    // W_hat is stored (n, k): output features first, input features second.
    let n = w.dim(0).ce()?;
    let k = w.dim(1).ce()?;
    if n == 0 || k == 0 || !n.is_multiple_of(128) || !k.is_multiple_of(128) {
        return Err(format!(
            "W_hat dims {}x{} must both be non-zero multiples of 128",
            n, k
        ));
    }
    let h = h128.to_dtype(DType::F16).ce()?;
    // Left H_n: H @ block for each 128-row block of W_hat -> (n, k).
    let nb = n / 128;
    let rows = matmul_c(&h.repeat((nb, 1, 1)).ce()?, &w.reshape((nb, 128, k)).ce()?)?
        .reshape((n, k))
        .ce()?;
    // Right H_k: block @ H^T for each 128-column block -> (n, k).
    // H is symmetric, so block @ H^T == H @ block.
    let kb = k / 128;
    let cols = matmul_c(&rows.reshape((n * kb, 128)).ce()?, &h.t().ce()?)?
        .reshape((n, k))
        .ce()?;
    Ok(cols)
}

/// Batched GQA with a per-head output gate. `kc`/`vc` are `(context, nkv*hd)`
/// fp16 caches, `q`/`gate` are `(nh*hd,)` fp32. Returns the `(nh*hd,)` fp32
/// attention output in head-major order.
///
/// Grouping the heads into `(nkv, groups, hd)` replaces the old per-query-head
/// loop (~16 tiny kernels per head, per layer, per token) with a handful of
/// batched matmuls. The math is identical to that loop;
/// `gqa_batched_matches_per_head` pins it against a literal transcription of
/// the old implementation.
#[allow(clippy::too_many_arguments)]
fn gqa_attention(
    kc: &Tensor,
    vc: &Tensor,
    q: &Tensor,
    gate: &Tensor,
    tlen: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Result<Tensor, String> {
    let scale = 1.0 / (hd as f32).sqrt();
    let groups = nh / nkv;
    // Cache row layout is (nkv, hd); select the used prefix only.
    let kmat = kc
        .narrow(0, 0, tlen)
        .ce()?
        .reshape((tlen, nkv, hd))
        .ce()?
        .to_dtype(DType::F32)
        .ce()?;
    let vmat = vc
        .narrow(0, 0, tlen)
        .ce()?
        .reshape((tlen, nkv, hd))
        .ce()?
        .to_dtype(DType::F32)
        .ce()?;
    let qg = q.reshape((nkv, groups, hd)).ce()?;
    let gate2 = gate.reshape((nkv, groups, hd)).ce()?;
    // scores[kh, t, g] = sum_d kmat[kh, t, d] * qg[kh, g, d]
    let scores = matmul_c(&kmat.permute((1, 0, 2)).ce()?, &qg.transpose(1, 2).ce()?)?
        .affine(scale as f64, 0.0)
        .ce()?;
    let m = scores.max_keepdim(1).ce()?;
    let e = scores.broadcast_sub(&m).ce()?.exp().ce()?;
    let probs = e.broadcast_div(&e.sum_keepdim(1).ce()?).ce()?;
    // out[kh, g, d] = sum_t probs[kh, t, g] * vmat[kh, t, d]
    let out = matmul_c(&vmat.permute((1, 2, 0)).ce()?, &probs)?
        .transpose(1, 2)
        .ce()?;
    let sig = gate2
        .neg()
        .ce()?
        .exp()
        .ce()?
        .affine(1.0, 1.0)
        .ce()?
        .recip()
        .ce()?;
    out.mul(&sig).ce()?.flatten_all().ce()
}

/// One Gated DeltaNet recurrence step, entirely on device.
#[allow(clippy::too_many_arguments)]
fn deltanet_step(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: f32,
    beta: f32,
    kd: usize,
    vd: usize,
) -> Result<(Tensor, Tensor), String> {
    let qn = k_add_eps(q)?;
    let kn = k_add_eps(k)?;
    let qh = q
        .broadcast_div(&qn)
        .ce()?
        .affine(1.0 / (kd as f64).sqrt(), 0.0)
        .ce()?;
    let kh = k.broadcast_div(&kn).ce()?;
    let s = state.affine(g.exp() as f64, 0.0).ce()?;
    let mem = matmul_c(&s.t().ce()?, &kh.reshape((kd, 1)).ce()?)?
        .flatten_all()
        .ce()?;
    let delta = v.sub(&mem).ce()?.affine(beta as f64, 0.0).ce()?;
    let s2 = s
        .add(&matmul_c(
            &kh.reshape((kd, 1)).ce()?,
            &delta.reshape((1, vd)).ce()?,
        )?)
        .ce()?;
    let o = matmul_c(&s2.t().ce()?, &qh.reshape((kd, 1)).ce()?)?
        .flatten_all()
        .ce()?;
    Ok((s2, o))
}

fn k_add_eps(v: &Tensor) -> Result<Tensor, String> {
    v.sqr()
        .ce()?
        .sum(0)
        .ce()?
        .affine(1.0, 1e-6)
        .ce()?
        .sqrt()
        .ce()
}

/// Row-wise L2 norm with the same `+1e-6` epsilon as `k_add_eps`, for a
/// `(rows, dim)` tensor. Returns `(rows,)`.
fn k_add_eps_rows(v: &Tensor) -> Result<Tensor, String> {
    v.sqr()
        .ce()?
        .sum(1)
        .ce()?
        .affine(1.0, 1e-6)
        .ce()?
        .sqrt()
        .ce()
}

/// Batched Gated DeltaNet recurrence: every value head advances in one set of
/// device ops. This is numerically identical to calling `deltanet_step` once
/// per head, but with no `to_vec0()` host round-trip for `g`/`beta` and only a
/// handful of batched matmuls instead of ~15 small kernels per head.
///
/// `state` is `(nvh, kd, vd)`, `q`/`k` are `(nkh * kd,)`, `v` is `(nvh * vd,)`,
/// and `g`/`beta` are `(nvh,)`. Heads are laid out as in the CPU forward: key
/// head `vh / rep` is shared by `rep = nvh / nkh` consecutive value heads.
#[allow(clippy::too_many_arguments)]
fn deltanet_step_all(
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    nkh: usize,
    nvh: usize,
    kd: usize,
    vd: usize,
) -> Result<(Tensor, Tensor), String> {
    let rep = nvh / nkh;
    // q/k are per key head; repeat each key head `rep` times so head i lines
    // up with value heads i*rep .. i*rep+rep. unsqueeze(1) + repeat gives
    // [h0,h0,h1,h1,...] order (a plain repeat of the matrix would give
    // [h0,h1,...,h0,h1,...] instead).
    let repeat_heads = |t: &Tensor| -> Result<Tensor, String> {
        t.reshape((nkh, kd))
            .ce()?
            .unsqueeze(1)
            .ce()?
            .repeat((1, rep, 1))
            .ce()?
            .reshape((nvh, kd))
            .ce()
    };
    let qh = repeat_heads(q)?;
    let kh = repeat_heads(k)?;
    let qh = qh
        .broadcast_div(&k_add_eps_rows(&qh)?.unsqueeze(1).ce()?)
        .ce()?
        .affine(1.0 / (kd as f64).sqrt(), 0.0)
        .ce()?;
    let kh = kh
        .broadcast_div(&k_add_eps_rows(&kh)?.unsqueeze(1).ce()?)
        .ce()?;

    // exp(g) < 1 decays each head's recurrent state; broadcast over (kd, vd).
    let s = state
        .broadcast_mul(&g.exp().ce()?.reshape((nvh, 1, 1)).ce()?)
        .ce()?;
    // (nvh, kd, 1)
    let kh_col = kh.unsqueeze(2).ce()?;
    // mem = S^T @ k_hat -> (nvh, vd)
    let mem = matmul_c(&s.transpose(1, 2).ce()?, &kh_col)?
        .squeeze(2)
        .ce()?;
    let v2 = v.reshape((nvh, vd)).ce()?;
    let delta = v2
        .sub(&mem)
        .ce()?
        .broadcast_mul(&beta.unsqueeze(1).ce()?)
        .ce()?;
    // S' = S + k_hat (x) delta, then o = S'^T @ q_hat.
    let s2 = s.add(&matmul_c(&kh_col, &delta.unsqueeze(1).ce()?)?).ce()?;
    let o = matmul_c(&s2.transpose(1, 2).ce()?, &qh.unsqueeze(2).ce()?)?
        .squeeze(2)
        .ce()?;
    Ok((s2, o.flatten_all().ce()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_matrix_matches_butterfly() {
        // H is symmetric and normalized: H @ v must equal the CPU butterfly.
        let mut h = vec![0.0f32; 128 * 128];
        let scale = 1.0 / (128.0f32).sqrt();
        for i in 0..128 {
            for j in 0..128 {
                h[i * 128 + j] = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
            }
        }
        let v: Vec<f32> = (0..128)
            .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let mut expected = v.clone();
        crate::exl3::hadamard128(&mut expected, 0, 1, scale);
        let dev = Device::Cpu;
        let ht = Tensor::from_slice(&h, (128, 128), &dev).unwrap();
        let vt = Tensor::from_slice(&v, (1, 128), &dev).unwrap();
        let got = ht.matmul(&vt.t().unwrap()).unwrap().flatten_all().unwrap();
        let got = got.to_vec1::<f32>().unwrap();
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }

    #[test]
    fn rms_norm_device_matches_hand_computation() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[3.0f32, 4.0, 0.0, -2.0], (4,), &dev).unwrap();
        let w = Tensor::from_slice(&[0.0f32, 0.5, -0.5, 1.0], (4,), &dev).unwrap();
        let cfg = GpuCfg {
            num_layers: 0,
            hidden_size: 4,
            vocab_size: 4,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            layer_types: Vec::new(),
            full_attention_interval: 1,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 1,
            rope_half: 0,
        };
        let gw = GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats: HashMap::new(),
            qhat: HashMap::new(),
            vecs: HashMap::new(),
            rope_cos: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            rope_sin: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            h128: Tensor::zeros((128, 128), DType::F32, &dev).unwrap(),
        };
        let got = gw.rms_norm(&x, &w).unwrap().to_vec1::<f32>().unwrap();
        let ss: f32 = [3.0f32, 4.0, 0.0, -2.0].iter().map(|v| v * v).sum();
        let scale = 1.0 / (ss / 4.0 + 1e-6).sqrt();
        let expected: Vec<f32> = [3.0f32, 4.0, 0.0, -2.0]
            .iter()
            .zip([0.0f32, 0.5, -0.5, 1.0].iter())
            .map(|(v, w)| v * scale * (1.0 + w))
            .collect();
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }

    /// q_norm/k_norm pass a 2-D `(heads, head_dim)` tensor. Each head must be
    /// normalized independently, matching CPU `forward_full_attn` — reducing
    /// across heads instead produces one scale per feature shared by all
    /// heads, which silently corrupts attention.
    #[test]
    fn rms_norm_2d_normalizes_each_head() {
        let dev = Device::Cpu;
        let cfg = GpuCfg {
            num_layers: 0,
            hidden_size: 2,
            vocab_size: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 2,
            layer_types: Vec::new(),
            full_attention_interval: 1,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 1,
            rope_half: 0,
        };
        let gw = GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats: HashMap::new(),
            qhat: HashMap::new(),
            vecs: HashMap::new(),
            rope_cos: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            rope_sin: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            h128: Tensor::zeros((128, 128), DType::F32, &dev).unwrap(),
        };
        let x = Tensor::from_slice(&[3.0f32, 4.0, 5.0, 12.0], (2, 2), &dev).unwrap();
        let w = Tensor::from_slice(&[0.0f32, 0.0], (2,), &dev).unwrap();
        let got = gw
            .rms_norm(&x, &w)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let scale = |a: f32, b: f32| 1.0 / ((a * a + b * b) / 2.0 + 1e-6).sqrt();
        let (s0, s1) = (scale(3.0, 4.0), scale(5.0, 12.0));
        let expected = [3.0 * s0, 4.0 * s0, 5.0 * s1, 12.0 * s1];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }

    /// Partial rotary: only the first `2*half` dims of each head rotate; the
    /// remaining dims must pass through untouched (CPU `apply_rope`).
    #[test]
    fn rope_preserves_unrotated_tail() {
        let dev = Device::Cpu;
        let (heads, hd, half) = (2usize, 8usize, 2usize);
        let cfg = GpuCfg {
            num_layers: 0,
            hidden_size: hd,
            vocab_size: 4,
            num_attention_heads: heads,
            num_key_value_heads: heads,
            head_dim: hd,
            layer_types: Vec::new(),
            full_attention_interval: 1,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 2,
            rope_half: half,
        };
        let gw = GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats: HashMap::new(),
            qhat: HashMap::new(),
            vecs: HashMap::new(),
            // pos 0: cos=0.8, sin=0.6 for both rotary pairs.
            rope_cos: Tensor::from_slice(&[0.8f32, 0.8, 0.8, 0.8], (2, half), &dev).unwrap(),
            rope_sin: Tensor::from_slice(&[0.6f32, 0.6, 0.6, 0.6], (2, half), &dev).unwrap(),
            h128: Tensor::zeros((128, 128), DType::F32, &dev).unwrap(),
        };
        let x: Vec<f32> = (0..(heads * hd)).map(|i| i as f32).collect();
        let got = gw
            .rope(
                &Tensor::from_slice(&x, (heads * hd,), &dev).unwrap(),
                heads,
                hd,
                0,
            )
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got.len(), heads * hd, "rope must preserve head length");
        for h in 0..heads {
            let base = h * hd;
            for d in 0..half {
                let (a, b) = (x[base + d], x[base + d + half]);
                assert!((got[base + d] - (a * 0.8 - b * 0.6)).abs() < 1e-5);
                assert!((got[base + d + half] - (b * 0.8 + a * 0.6)).abs() < 1e-5);
            }
            // Unrotated tail (dims 4..7) is copied verbatim.
            for d in (half * 2)..hd {
                assert_eq!(got[base + d], x[base + d], "tail dim {} changed", d);
            }
        }
    }

    #[test]
    fn deltanet_step_device_matches_cpu_recurrence() {
        let dev = Device::Cpu;
        let (kd, vd) = (2usize, 3usize);
        let s_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6]; // (kd,vd)
        let q = [1.0f32, 2.0];
        let k = [0.5f32, -0.5];
        let v = [1.0f32, 0.0, -1.0];
        let (g, beta) = (-0.1f32, 0.7f32);
        let st = Tensor::from_slice(&s_data, (kd, vd), &dev).unwrap();
        let qt = Tensor::from_slice(&q, (kd,), &dev).unwrap();
        let kt = Tensor::from_slice(&k, (kd,), &dev).unwrap();
        let vt = Tensor::from_slice(&v, (vd,), &dev).unwrap();
        let (got_s, got_o) = deltanet_step(&st, &qt, &kt, &vt, g, beta, kd, vd).unwrap();

        // CPU reference, line-for-line with Model::deltanet_step.
        let qn = (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        let kn = (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        let qh: Vec<f32> = q.iter().map(|x| x / qn / (kd as f32).sqrt()).collect();
        let kh: Vec<f32> = k.iter().map(|x| x / kn).collect();
        let mut s = s_data;
        for x in &mut s {
            *x *= g.exp();
        }
        let mut mem = [0.0f32; 3];
        for (i, kv) in kh.iter().enumerate() {
            for j in 0..vd {
                mem[j] += s[i * vd + j] * kv;
            }
        }
        let delta: Vec<f32> = (0..vd).map(|j| (v[j] - mem[j]) * beta).collect();
        for (i, kv) in kh.iter().enumerate() {
            for j in 0..vd {
                s[i * vd + j] += kv * delta[j];
            }
        }
        let mut o = [0.0f32; 3];
        for (i, qv) in qh.iter().enumerate() {
            for j in 0..vd {
                o[j] += s[i * vd + j] * qv;
            }
        }
        let got_s = got_s.to_vec2::<f32>().unwrap();
        for i in 0..kd {
            for j in 0..vd {
                assert!((got_s[i][j] - s[i * vd + j]).abs() < 1e-5);
            }
        }
        let got_o = got_o.to_vec1::<f32>().unwrap();
        for j in 0..vd {
            assert!((got_o[j] - o[j]).abs() < 1e-5, "{} vs {}", got_o[j], o[j]);
        }
    }

    /// The batched recurrence must be numerically identical to running the
    /// original per-head `deltanet_step` for each value head. This guards the
    /// hot-path rewrite that removed nvh host syncs per layer per token.
    #[test]
    fn deltanet_all_matches_per_head() {
        let dev = Device::Cpu;
        let (nkh, nvh, kd, vd) = (2usize, 4usize, 2usize, 3usize);
        let rep = nvh / nkh;
        let sdata: Vec<f32> = (0..nvh * kd * vd)
            .map(|i| ((i * 7) % 13) as f32 / 10.0 - 0.6)
            .collect();
        let qd: Vec<f32> = (0..nkh * kd)
            .map(|i| ((i * 5) % 11) as f32 / 5.0 - 1.0)
            .collect();
        let kd_data: Vec<f32> = (0..nkh * kd)
            .map(|i| ((i * 3) % 7) as f32 / 4.0 - 0.75)
            .collect();
        let vd_data: Vec<f32> = (0..nvh * vd)
            .map(|i| ((i * 11) % 9) as f32 / 6.0 - 0.6)
            .collect();
        let gd: Vec<f32> = (0..nvh).map(|i| -0.1 - i as f32 * 0.05).collect();
        let bd: Vec<f32> = (0..nvh).map(|i| 0.2 + i as f32 * 0.1).collect();

        let st = Tensor::from_slice(&sdata, (nvh, kd, vd), &dev).unwrap();
        let qt = Tensor::from_slice(&qd, (nkh * kd,), &dev).unwrap();
        let kt = Tensor::from_slice(&kd_data, (nkh * kd,), &dev).unwrap();
        let vt = Tensor::from_slice(&vd_data, (nvh * vd,), &dev).unwrap();
        let gt = Tensor::from_slice(&gd, (nvh,), &dev).unwrap();
        let bt = Tensor::from_slice(&bd, (nvh,), &dev).unwrap();
        let (got_s, got_o) =
            deltanet_step_all(&st, &qt, &kt, &vt, &gt, &bt, nkh, nvh, kd, vd).unwrap();

        // Literal transcription of the old per-head hot loop.
        let mut ref_s = vec![0.0f32; nvh * kd * vd];
        let mut ref_o = vec![0.0f32; nvh * vd];
        for vh in 0..nvh {
            let ki = vh / rep;
            let (ns, o) = deltanet_step(
                &Tensor::from_slice(&sdata[vh * kd * vd..(vh + 1) * kd * vd], (kd, vd), &dev)
                    .unwrap(),
                &Tensor::from_slice(&qd[ki * kd..(ki + 1) * kd], (kd,), &dev).unwrap(),
                &Tensor::from_slice(&kd_data[ki * kd..(ki + 1) * kd], (kd,), &dev).unwrap(),
                &Tensor::from_slice(&vd_data[vh * vd..(vh + 1) * vd], (vd,), &dev).unwrap(),
                gd[vh],
                bd[vh],
                kd,
                vd,
            )
            .unwrap();
            let ns = ns.reshape((kd * vd,)).unwrap().to_vec1::<f32>().unwrap();
            ref_s[vh * kd * vd..(vh + 1) * kd * vd].copy_from_slice(&ns);
            ref_o[vh * vd..(vh + 1) * vd].copy_from_slice(&o.to_vec1::<f32>().unwrap());
        }
        let got_s = got_s
            .reshape((nvh * kd * vd,))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_o = got_o.to_vec1::<f32>().unwrap();
        for (i, (a, b)) in got_s.iter().zip(&ref_s).enumerate() {
            assert!((a - b).abs() < 1e-5, "state[{}]: {} vs {}", i, a, b);
        }
        for (i, (a, b)) in got_o.iter().zip(&ref_o).enumerate() {
            assert!((a - b).abs() < 1e-5, "out[{}]: {} vs {}", i, a, b);
        }
    }

    /// Batched GQA must match the old per-query-head loop element for element.
    #[test]
    fn gqa_batched_matches_per_head() {
        let dev = Device::Cpu;
        let (tlen, nkv, groups, hd) = (4usize, 2usize, 2usize, 3usize);
        let nh = nkv * groups;
        // Multiples of 1/8 are exact in fp16, so cache conversion adds no noise.
        let kd_data: Vec<f16> = (0..tlen * nkv * hd)
            .map(|i| f16::from_f32(((i * 13) % 17) as f32 / 8.0 - 1.0))
            .collect();
        let vd_data: Vec<f16> = (0..tlen * nkv * hd)
            .map(|i| f16::from_f32(((i * 5) % 11) as f32 / 8.0 - 1.0))
            .collect();
        let qd: Vec<f32> = (0..nh * hd)
            .map(|i| ((i * 7) % 11) as f32 / 5.0 - 1.0)
            .collect();
        let gd: Vec<f32> = (0..nh * hd)
            .map(|i| ((i * 3) % 9) as f32 / 4.0 - 1.0)
            .collect();
        let kc = Tensor::from_slice(&kd_data, (tlen, nkv * hd), &dev).unwrap();
        let vc = Tensor::from_slice(&vd_data, (tlen, nkv * hd), &dev).unwrap();
        let q = Tensor::from_slice(&qd, (nh * hd,), &dev).unwrap();
        let gate = Tensor::from_slice(&gd, (nh * hd,), &dev).unwrap();

        let got = gqa_attention(&kc, &vc, &q, &gate, tlen, nh, nkv, hd)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let kmat_all = kc
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((tlen, nkv, hd))
            .unwrap();
        let vmat_all = vc
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((tlen, nkv, hd))
            .unwrap();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut expected = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let kh = h / groups;
            let kmat = kmat_all.narrow(1, kh, 1).unwrap().squeeze(1).unwrap();
            let vmat = vmat_all.narrow(1, kh, 1).unwrap().squeeze(1).unwrap();
            let qh = q.narrow(0, h * hd, hd).unwrap().reshape((1, hd)).unwrap();
            let mut scores = kmat
                .matmul(&qh.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap();
            scores = scores.affine(scale as f64, 0.0).unwrap();
            let m = scores.max_keepdim(0).unwrap();
            let e = scores.broadcast_sub(&m).unwrap().exp().unwrap();
            let probs = e.broadcast_div(&e.sum_keepdim(0).unwrap()).unwrap();
            let mut out = probs
                .reshape((1, tlen))
                .unwrap()
                .matmul(&vmat)
                .unwrap()
                .flatten_all()
                .unwrap();
            let gh = gate.narrow(0, h * hd, hd).unwrap();
            let sig = gh
                .neg()
                .unwrap()
                .exp()
                .unwrap()
                .affine(1.0, 1.0)
                .unwrap()
                .recip()
                .unwrap();
            out = out.mul(&sig).unwrap();
            expected[h * hd..(h + 1) * hd].copy_from_slice(&out.to_vec1::<f32>().unwrap());
        }
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() < 1e-4, "attn[{}]: {} vs {}", i, a, b);
        }
    }

    fn tiny_mat(dev: &Device, rows: usize, cols: usize) -> Tensor {
        let data: Vec<f16> = (0..rows * cols)
            .map(|i| f16::from_f32(((i * 7) % 13) as f32 / 10.0 - 0.6))
            .collect();
        Tensor::from_slice(&data, (rows, cols), dev).unwrap()
    }

    fn tiny_f32_mat(dev: &Device, rows: usize, cols: usize) -> Tensor {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32 / 20.0).collect();
        Tensor::from_slice(&data, (rows, cols), dev).unwrap()
    }

    fn tiny_vec(dev: &Device, len: usize) -> Tensor {
        let data: Vec<f32> = (0..len).map(|i| (i % 5) as f32 / 100.0).collect();
        Tensor::from_slice(&data, (len,), dev).unwrap()
    }

    /// Runs the whole device forward on candle's CPU device with a tiny model.
    /// This catches shape/broadcast bugs in every branch without needing a GPU.
    #[test]
    fn device_forward_runs_end_to_end_on_cpu() {
        let dev = Device::Cpu;
        let cfg = GpuCfg {
            num_layers: 2,
            hidden_size: 4,
            vocab_size: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            layer_types: vec!["linear_attention".into(), "full_attention".into()],
            full_attention_interval: 1,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 2,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_value_head_dim: 2,
            rms_norm_eps: 1e-6,
            context_size: 4,
            // Partial rotary (rotary=2 < head_dim=4) — the real-model case.
            // A full-rotation fixture would not exercise the unrotated tail.
            rope_half: 1,
        };
        let mut mats = HashMap::new();
        let mut vecs = HashMap::new();
        let m = |name: &str, r: usize, c: usize, mats: &mut HashMap<String, Tensor>| {
            mats.insert(name.to_string(), tiny_mat(&dev, r, c));
        };
        let v = |name: &str, n: usize, vecs: &mut HashMap<String, Tensor>| {
            vecs.insert(name.to_string(), tiny_vec(&dev, n));
        };

        m("embed_tokens.weight", 4, 4, &mut mats);
        m("lm_head.weight", 4, 4, &mut mats);
        // Layer 0: linear attention (key_dim=4, val_dim=8, conv_dim=16).
        m("layers.0.linear_attn.in_proj_qkv", 16, 4, &mut mats);
        m("layers.0.linear_attn.in_proj_z", 8, 4, &mut mats);
        m("layers.0.linear_attn.in_proj_b", 4, 4, &mut mats);
        m("layers.0.linear_attn.in_proj_a", 4, 4, &mut mats);
        m("layers.0.linear_attn.out_proj", 4, 8, &mut mats);
        m("layers.0.mlp.gate_proj", 8, 4, &mut mats);
        m("layers.0.mlp.up_proj", 8, 4, &mut mats);
        m("layers.0.mlp.down_proj", 4, 8, &mut mats);
        // Layer 1: full attention (nh=2, hd=4 -> q_proj out = 2*2*4).
        m("layers.1.self_attn.q_proj", 16, 4, &mut mats);
        m("layers.1.self_attn.k_proj", 4, 4, &mut mats);
        m("layers.1.self_attn.v_proj", 4, 4, &mut mats);
        m("layers.1.self_attn.o_proj", 4, 8, &mut mats);
        m("layers.1.mlp.gate_proj", 8, 4, &mut mats);
        m("layers.1.mlp.up_proj", 8, 4, &mut mats);
        m("layers.1.mlp.down_proj", 4, 8, &mut mats);

        for name in [
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.1.input_layernorm.weight",
            "layers.1.post_attention_layernorm.weight",
            "norm.weight",
        ] {
            v(name, 4, &mut vecs);
        }
        v("layers.0.linear_attn.conv1d.weight", 32, &mut vecs);
        v("layers.0.linear_attn.A_log", 4, &mut vecs);
        v("layers.0.linear_attn.dt_bias", 4, &mut vecs);
        v("layers.0.linear_attn.norm.weight", 2, &mut vecs);
        v("layers.1.self_attn.q_norm.weight", 4, &mut vecs);
        v("layers.1.self_attn.k_norm.weight", 4, &mut vecs);

        let gw = Arc::new(GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats,
            qhat: HashMap::new(),
            vecs,
            rope_cos: tiny_f32_mat(&dev, 4, 1),
            rope_sin: tiny_f32_mat(&dev, 4, 1),
            h128: Tensor::zeros((128, 128), DType::F32, &dev).unwrap(),
        });
        let mut runner = GpuRunner::new(gw).unwrap();
        // Prefill three tokens (exercises full attention with tlen > 1).
        let logits = runner.forward(&[0, 1, 2]).unwrap();
        assert_eq!(logits.len(), 4);
        assert!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
        // Decode one more token (continues both attention states).
        let logits = runner.forward(&[3]).unwrap();
        assert_eq!(logits.len(), 4);
        assert!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    }

    #[test]
    fn quantized_matmul_runs_on_device() {
        let dev = Device::Cpu;
        let (k, n) = (128usize, 128usize);
        let wf: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32(((i * 11) % 17) as f32 / 20.0 - 0.4))
            .collect();
        let su: Vec<f32> = (0..k).map(|i| 1.0 + (i % 3) as f32 / 10.0).collect();
        let sv: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32 / 10.0).collect();
        let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 / 10.0 - 0.3).collect();
        let parts = (
            Tensor::from_slice(&wf, (n, k), &dev).unwrap(),
            Tensor::from_slice(&su, (k,), &dev).unwrap(),
            Tensor::from_slice(&sv, (n,), &dev).unwrap(),
        );
        let xt = Tensor::from_slice(&x, (k,), &dev).unwrap();
        let cfg = GpuCfg {
            num_layers: 0,
            hidden_size: k,
            vocab_size: n,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
            layer_types: Vec::new(),
            full_attention_interval: 1,
            linear_conv_kernel_dim: 1,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 1,
            rope_half: 0,
        };
        let gw = GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats: HashMap::new(),
            qhat: HashMap::new(),
            vecs: HashMap::new(),
            rope_cos: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            rope_sin: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            h128: {
                let scale = 1.0 / 128.0f32.sqrt();
                let mut h = vec![0.0f32; 128 * 128];
                for i in 0..128 {
                    for j in 0..128 {
                        h[i * 128 + j] = if (i & j).count_ones() % 2 == 0 {
                            scale
                        } else {
                            -scale
                        };
                    }
                }
                Tensor::from_slice(&h, (128, 128), &dev).unwrap()
            },
        };
        // q_matmul now consumes the load-time-folded weight.
        let folded = (
            fold_hadamards(&parts.0, &gw.h128).unwrap(),
            parts.1.clone(),
            parts.2.clone(),
        );
        let got = gw.q_matmul(&folded, &xt).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got.len(), n);
        assert!(got.iter().all(|v| v.is_finite()));
    }

    /// Folding both Hadamards into W_hat must not change the linear result
    /// beyond the expected fp16 rounding (the old path mixed in f32, the new
    /// one bakes the mix into the fp16 weight).
    #[test]
    fn folded_q_matmul_matches_unfolded() {
        let dev = Device::Cpu;
        // Non-square: k=128 (one H_k block), n=256 (two H_n blocks).
        let (k, n) = (128usize, 256usize);
        let wf: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32(((i * 11) % 17) as f32 / 20.0 - 0.4))
            .collect();
        let su: Vec<f32> = (0..k).map(|i| 1.0 + (i % 3) as f32 / 10.0).collect();
        let sv: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32 / 10.0).collect();
        let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 / 10.0 - 0.3).collect();
        let scale = 1.0 / 128.0f32.sqrt();
        let mut h = vec![0.0f32; 128 * 128];
        for i in 0..128 {
            for j in 0..128 {
                h[i * 128 + j] = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
            }
        }
        let h128 = Tensor::from_slice(&h, (128, 128), &dev).unwrap();
        let cfg = GpuCfg {
            num_layers: 0,
            hidden_size: k,
            vocab_size: n,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
            layer_types: Vec::new(),
            full_attention_interval: 1,
            linear_conv_kernel_dim: 1,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 1,
            rope_half: 0,
        };
        let gw = GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats: HashMap::new(),
            qhat: HashMap::new(),
            vecs: HashMap::new(),
            rope_cos: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            rope_sin: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            h128: h128.clone(),
        };
        let w = Tensor::from_slice(&wf, (n, k), &dev).unwrap();
        let suh = Tensor::from_slice(&su, (k,), &dev).unwrap();
        let svh = Tensor::from_slice(&sv, (n,), &dev).unwrap();
        let xt = Tensor::from_slice(&x, (k,), &dev).unwrap();

        // Old decode path: two f32 Hadamard GEMMs around one f16 W_hat GEMV.
        let xr = gw.hadamard(&xt.mul(&suh).unwrap()).unwrap();
        let t = GpuWeights::matmul_w(&w, &xr).unwrap();
        let t = gw.hadamard(&t).unwrap();
        let expected = t.mul(&svh).unwrap().to_vec1::<f32>().unwrap();

        // New decode path: one GEMV. Test both the device-side reference fold
        // and the host fold used by `upload_device` in production.
        let folded = (fold_hadamards(&w, &h128).unwrap(), suh.clone(), svh.clone());
        let got = gw.q_matmul(&folded, &xt).unwrap().to_vec1::<f32>().unwrap();
        let host_bits = crate::exl3::fold_hadamards_host(&wf, n, k).unwrap();
        let host_parts = (
            Tensor::from_slice(&host_bits, (n, k), &dev).unwrap(),
            suh.clone(),
            svh.clone(),
        );
        let host_got = gw
            .q_matmul(&host_parts, &xt)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Observed max error ~0.3% of the largest logit (extra fp16 roundings
        // from baking the Hadamards into the weight).
        let scale = got.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (i, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 5e-3 + 5e-3 * scale,
                "logit {}: device-folded {} vs unfolded {}",
                i,
                a,
                b
            );
            assert!(
                (host_got[i] - b).abs() < 5e-3 + 5e-3 * scale,
                "logit {}: host-folded {} vs unfolded {}",
                i,
                host_got[i],
                b
            );
        }
    }

    /// Guard specifically against the v0.5.3 load bug where `fold_hadamards`
    /// read W_hat's `(n, k)` dims backwards and transposed the result, so any
    /// non-square quantized linear failed with a `matmul` dimension error.
    #[test]
    fn fold_preserves_shape_for_non_square_weights() {
        let dev = Device::Cpu;
        let scale = 1.0 / 128.0f32.sqrt();
        let mut h = vec![0.0f32; 128 * 128];
        for i in 0..128 {
            for j in 0..128 {
                h[i * 128 + j] = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
            }
        }
        let h128 = Tensor::from_slice(&h, (128, 128), &dev).unwrap();
        for (n, k) in [(256usize, 128usize), (128, 256), (1024, 128), (128, 1024)] {
            let w = Tensor::from_slice(&vec![f16::from_f32(0.1); n * k], (n, k), &dev).unwrap();
            let folded = fold_hadamards(&w, &h128).unwrap();
            assert_eq!(folded.dims(), &[n, k], "folded shape for {}x{}", n, k);
            let x = Tensor::from_slice(&vec![0.5f32; k], (k,), &dev).unwrap();
            let out = GpuWeights::matmul_w(&folded, &x).unwrap();
            assert_eq!(out.elem_count(), n, "output length for {}x{}", n, k);
        }
    }

    /// Untied checkpoints quantize lm_head under the bare name `lm_head`,
    /// while still storing an embedding table. `linear()` must use the
    /// quantized head, not the (correctly shaped) embedding fallback.
    #[test]
    fn linear_prefers_quantized_lm_head_over_tied_embedding() {
        let dev = Device::Cpu;
        let (k, n) = (128usize, 128usize);
        let wf: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32(((i * 11) % 17) as f32 / 20.0 - 0.4))
            .collect();
        let su: Vec<f32> = (0..k).map(|i| 1.0 + (i % 3) as f32 / 10.0).collect();
        let sv: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32 / 10.0).collect();
        let parts = (
            Tensor::from_slice(&wf, (n, k), &dev).unwrap(),
            Tensor::from_slice(&su, (k,), &dev).unwrap(),
            Tensor::from_slice(&sv, (n,), &dev).unwrap(),
        );
        let emb: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32(((i * 3) % 7) as f32 / 10.0))
            .collect();
        let mut mats = HashMap::new();
        mats.insert(
            "embed_tokens.weight".to_string(),
            Tensor::from_slice(&emb, (n, k), &dev).unwrap(),
        );
        let mut qhat = HashMap::new();
        let cfg = GpuCfg {
            num_layers: 0,
            hidden_size: k,
            vocab_size: n,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 1,
            layer_types: Vec::new(),
            full_attention_interval: 1,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 1,
            rope_half: 0,
        };
        let scale = 1.0 / 128.0f32.sqrt();
        let mut h = vec![0.0f32; 128 * 128];
        for i in 0..128 {
            for j in 0..128 {
                h[i * 128 + j] = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
            }
        }
        let h128 = Tensor::from_slice(&h, (128, 128), &dev).unwrap();
        let folded = (
            fold_hadamards(&parts.0, &h128).unwrap(),
            parts.1.clone(),
            parts.2.clone(),
        );
        qhat.insert("lm_head".to_string(), folded.clone());
        let gw = GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats,
            qhat,
            vecs: HashMap::new(),
            rope_cos: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            rope_sin: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            h128,
        };
        let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 / 10.0 - 0.3).collect();
        let xt = Tensor::from_slice(&x, (k,), &dev).unwrap();
        let got = gw
            .linear("lm_head.weight", &xt)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = gw.q_matmul(&folded, &xt).unwrap().to_vec1::<f32>().unwrap();
        let tied = GpuWeights::matmul_w(gw.embedding_table().unwrap(), &xt)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for i in 0..n {
            assert!((got[i] - expected[i]).abs() < 1e-5, "logit {} mismatch", i);
        }
        // Guard against a vacuous pass: the two paths must actually differ.
        assert!(
            got.iter()
                .zip(tied.iter())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "quantized head and tied embedding produced identical logits"
        );
    }

    #[test]
    fn f16_matmul_matches_f32_dot() {
        let dev = Device::Cpu;
        let w_data: Vec<f32> = (0..12)
            .map(|i| ((i * 17) % 23) as f32 / 10.0 - 1.0)
            .collect();
        let x_data: Vec<f32> = (0..4).map(|i| (i as f32) * 0.25 - 0.3).collect();
        let wf: Vec<f16> = w_data.iter().map(|&v| f16::from_f32(v)).collect();
        let w = Tensor::from_slice(&wf, (3, 4), &dev).unwrap();
        let x = Tensor::from_slice(&x_data, (4,), &dev).unwrap();
        let got = GpuWeights::matmul_w(&w, &x)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for j in 0..3 {
            let expect: f32 = (0..4)
                .map(|i| f16::from_f32(w_data[j * 4 + i]).to_f32() * x_data[i])
                .sum();
            // f16 GEMM accumulation order differs across candle backends
            // (macOS observed ~2e-4 on a ~0.06 result); allow a small absolute
            // tolerance while still catching real arithmetic errors.
            assert!((got[j] - expect).abs() < 1e-3, "{} vs {}", got[j], expect);
        }
    }

    /// A runner whose logits are a small host-provided vector, so the device
    /// sampler can be compared directly against the host sampler.
    fn sampler_runner(logits: &[f32]) -> GpuRunner {
        let dev = Device::Cpu;
        let cfg = GpuCfg {
            num_layers: 0,
            hidden_size: 4,
            vocab_size: logits.len(),
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            layer_types: Vec::new(),
            full_attention_interval: 1,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 1,
            rope_half: 0,
        };
        let gw = Arc::new(GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats: HashMap::new(),
            qhat: HashMap::new(),
            vecs: HashMap::new(),
            rope_cos: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            rope_sin: Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
            h128: Tensor::zeros((128, 128), DType::F32, &dev).unwrap(),
        });
        let mut r = GpuRunner::new(gw).unwrap();
        r.state.logits = Some(Tensor::from_slice(logits, (logits.len(),), &dev).unwrap());
        r
    }

    fn sample_logits() -> Vec<f32> {
        (0..64).map(|i| ((i * 7) % 23) as f32 / 5.0 - 2.0).collect()
    }

    fn sample_params(spec: &SampleSpec) -> crate::server::Params {
        crate::server::Params {
            temperature: spec.temperature,
            top_p: spec.top_p,
            top_k: spec.top_k,
            repetition_penalty: spec.repetition_penalty,
            presence_penalty: spec.presence_penalty,
            frequency_penalty: spec.frequency_penalty,
            logit_bias: spec.logit_bias.clone(),
            ..crate::server::Params::default()
        }
    }

    /// Greedy and temperature-only sampling must agree with the host sampler,
    /// including all logit penalties/bias, for the same RNG draw.
    #[test]
    fn device_sample_matches_host_greedy_and_softmax() {
        let logits = sample_logits();
        let history = [3u32, 5, 3, 9];
        let base = SampleSpec {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.5,
            presence_penalty: 0.2,
            frequency_penalty: 0.1,
            logit_bias: vec![(2, 3.0), (11, -2.0)],
            top_n: 0,
        };
        // Greedy: no randomness involved.
        let got = sampler_runner(&logits)
            .sample(&base, &history, 0.5)
            .unwrap();
        let host = crate::server::sample(
            &logits,
            &sample_params(&base),
            &mut crate::rng::Rng::new(1),
            &history,
        );
        assert_eq!(got.token, host, "greedy token");

        // Plain temperature sampling: compare against a transcription of
        // sample_softmax with the exact same uniform.
        let spec = SampleSpec {
            temperature: 0.8,
            ..base.clone()
        };
        let u = 0.37f64;
        let got = sampler_runner(&logits)
            .sample(&spec, &history, u as f32)
            .unwrap();
        let mut buf = Vec::new();
        let p = crate::server::prepare_logits(&logits, &sample_params(&spec), &history, &mut buf);
        let max = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max) / spec.temperature;
        let sum: f64 = p
            .iter()
            .map(|&l| (l / spec.temperature - max).exp() as f64)
            .sum();
        let target = u * sum;
        let mut cum = 0.0f64;
        let mut expect = (p.len() - 1) as u32;
        for (i, &l) in p.iter().enumerate() {
            cum += (l / spec.temperature - max).exp() as f64;
            if cum >= target {
                expect = i as u32;
                break;
            }
        }
        assert_eq!(got.token, expect, "softmax token");
    }

    /// top-k and top-p nucleus sampling must select the same token as the host
    /// sampler for the same uniform draw.
    #[test]
    fn device_sample_matches_host_top_k_and_top_p() {
        let logits = sample_logits();
        let history = [7u32, 7, 12];
        let spec = SampleSpec {
            temperature: 0.9,
            top_p: 0.7,
            top_k: 5,
            repetition_penalty: 1.2,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
            top_n: 0,
        };
        let u = 0.61f64;
        let got = sampler_runner(&logits)
            .sample(&spec, &history, u as f32)
            .unwrap();

        let mut buf = Vec::new();
        let p = crate::server::prepare_logits(&logits, &sample_params(&spec), &history, &mut buf);
        let by_id: Vec<f64> = p.iter().map(|&l| (l / spec.temperature) as f64).collect();
        let mut ranked: Vec<(usize, f64)> = by_id.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(spec.top_k);
        let max = ranked[0].1;
        let w = |i: usize| (by_id[i] - max).exp();
        let sum: f64 = ranked.iter().map(|&(i, _)| w(i)).sum();
        let mass = sum * spec.top_p as f64;
        // Nucleus: descending-score prefix until cum >= top_p.
        let mut nucleus = std::collections::HashSet::new();
        let mut cum = 0.0f64;
        for &(i, _) in &ranked {
            nucleus.insert(i);
            cum += w(i);
            if cum >= mass {
                break;
            }
        }
        // The device walks the selected set in token-id order (no sort).
        // Inverse-CDF sampling is order-independent in distribution; with the
        // same enumeration and uniform it must pick the same id.
        let total: f64 = nucleus.iter().map(|&i| w(i)).sum();
        let target = u * total;
        let mut cum = 0.0f64;
        let mut expect = *nucleus.iter().max().unwrap() as u32;
        for i in 0..p.len() {
            if !nucleus.contains(&i) {
                continue;
            }
            cum += w(i);
            if cum >= target {
                expect = i as u32;
                break;
            }
        }
        assert_eq!(got.token, expect);
    }

    /// The compact logprobs returned from device must match the host's
    /// `logprob_info` (same penalty-adjusted distribution, before sampling
    /// transforms), both for the chosen token and the top alternatives.
    #[test]
    fn device_logprobs_match_host() {
        let logits = sample_logits();
        let history = [4u32, 4, 4, 20];
        let spec = SampleSpec {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.3,
            presence_penalty: 0.1,
            frequency_penalty: 0.05,
            logit_bias: vec![(6, 1.5)],
            top_n: 4,
        };
        let got = sampler_runner(&logits)
            .sample(&spec, &history, 0.5)
            .unwrap();
        let mut buf = Vec::new();
        let p = crate::server::prepare_logits(&logits, &sample_params(&spec), &history, &mut buf);
        let max = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = p.iter().map(|&v| (v as f64 - max).exp()).sum();
        let lp = |v: f32| (v as f64 - max - sum.ln()) as f32;
        let mut top: Vec<(usize, f32)> = p.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        top.truncate(spec.top_n);
        assert_eq!(got.top.len(), top.len());
        for ((id, value), (eid, evalue)) in got.top.iter().zip(&top) {
            assert_eq!(*id, *eid as u32);
            assert!(
                (value - lp(*evalue)).abs() < 1e-3,
                "{} vs {}",
                value,
                lp(*evalue)
            );
        }
        assert!(
            (got.logprob - lp(p[got.token as usize])).abs() < 1e-3,
            "chosen logprob {} vs {}",
            got.logprob,
            lp(p[got.token as usize])
        );
    }

    /// The post-prefill snapshot must carry device logits, otherwise `n > 1`
    /// choices would sample from stale/absent logits after `restore`.
    #[test]
    fn snapshot_restores_device_logits() {
        let logits = sample_logits();
        let runner = sampler_runner(&logits);
        let snap = runner.snapshot().unwrap();
        let mut runner2 = sampler_runner(&vec![0.0; logits.len()]);
        runner2.restore(&snap).unwrap();
        let got = runner2
            .state
            .logits
            .as_ref()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, logits);
    }

    /// Forward -> sample -> forward -> sample must work entirely on device.
    #[test]
    fn device_forward_then_sample_runs_end_to_end_on_cpu() {
        let dev = Device::Cpu;
        let cfg = GpuCfg {
            num_layers: 1,
            hidden_size: 4,
            vocab_size: 8,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            layer_types: vec!["full_attention".into()],
            full_attention_interval: 1,
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 1,
            rms_norm_eps: 1e-6,
            context_size: 8,
            rope_half: 0,
        };
        let mut mats = HashMap::new();
        for (name, r, c) in [
            ("embed_tokens.weight", 8, 4),
            ("lm_head.weight", 8, 4),
            ("layers.0.self_attn.q_proj", 8, 4),
            ("layers.0.self_attn.k_proj", 4, 4),
            ("layers.0.self_attn.v_proj", 4, 4),
            ("layers.0.self_attn.o_proj", 4, 4),
            ("layers.0.mlp.gate_proj", 8, 4),
            ("layers.0.mlp.up_proj", 8, 4),
            ("layers.0.mlp.down_proj", 4, 8),
        ] {
            mats.insert(name.to_string(), tiny_mat(&dev, r, c));
        }
        let mut vecs = HashMap::new();
        for name in [
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "norm.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
        ] {
            vecs.insert(name.to_string(), tiny_vec(&dev, 4));
        }
        let gw = Arc::new(GpuWeights {
            device: dev.clone(),
            prefix: String::new(),
            cfg,
            mats,
            qhat: HashMap::new(),
            vecs,
            rope_cos: Tensor::zeros((8, 1), DType::F32, &dev).unwrap(),
            rope_sin: Tensor::zeros((8, 1), DType::F32, &dev).unwrap(),
            h128: Tensor::zeros((128, 128), DType::F32, &dev).unwrap(),
        });
        let mut runner = GpuRunner::new(gw).unwrap();
        let spec = SampleSpec {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.2,
            presence_penalty: 0.1,
            frequency_penalty: 0.0,
            logit_bias: vec![(1, 0.5)],
            top_n: 2,
        };
        runner.forward_logits(&[0, 1, 2]).unwrap();
        let history = Vec::new();
        let first = runner.sample(&spec, &history, 0.5).unwrap();
        assert!(first.token < 8 && first.logprob.is_finite());
        assert_eq!(first.top.len(), 2);
        runner.forward_logits(&[first.token]).unwrap();
        let second = runner.sample(&spec, &[first.token], 0.25).unwrap();
        assert!(second.token < 8 && second.logprob.is_finite());
    }

    #[test]
    fn f16_bits_roundtrip_through_tensor() {
        let bits: Vec<u16> = vec![0x3c00, 0x0000, 0xbc00, 0x3e00]; // 1, 0, -1, 1.5
        let vals: Vec<f16> = bits.iter().map(|&b| f16::from_bits(b)).collect();
        let t = Tensor::from_slice(&vals, (4,), &Device::Cpu).unwrap();
        let got = t.to_vec1::<f16>().unwrap();
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    }
}
