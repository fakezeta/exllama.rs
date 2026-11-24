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
    /// Quantized linears: (W_hat (n,k) F16, suh (k,) F32, svh (n,) F32).
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
        let mut mats = HashMap::new();
        for (name, m) in &ex.weights {
            let vals: Vec<f16> = m.data.iter().map(|&v| f16::from_f32(v)).collect();
            let t = Tensor::from_slice(&vals, (m.rows, m.cols), device).ce()?;
            mats.insert(name.clone(), t);
        }
        for (name, emb) in &ex.embeddings_f16 {
            let vals: Vec<f16> = emb.data.iter().map(|&b| f16::from_bits(b)).collect();
            let t = Tensor::from_slice(&vals, (emb.rows, emb.cols), device).ce()?;
            mats.insert(name.clone(), t);
        }
        let mut qhat = HashMap::new();
        for (name, ql) in &ex.linears {
            qhat.insert(name.clone(), ql.upload_device(device)?);
        }
        let mut vecs = HashMap::new();
        for (name, v) in &ex.vectors {
            let t = Tensor::from_slice(v, (v.len(),), device).ce()?;
            vecs.insert(name.clone(), t);
        }
        let rope_cos_t = Tensor::from_slice(rope_cos, (context_size, rope_half), device).ce()?;
        let rope_sin_t = Tensor::from_slice(rope_sin, (context_size, rope_half), device).ce()?;
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
            out.push("lm_head".to_string());
            out.push(format!("{}embed_tokens.weight", self.prefix));
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
        let blocks = v.dim(0).ce()? / 128;
        let out = v.reshape((blocks, 128)).ce()?.matmul(&self.h128).ce()?;
        out.flatten_all().ce()
    }

    /// `y = W @ x`, W stored (out, in) F16. The GEMM is done on device.
    fn matmul_w(w: &Tensor, x: &Tensor) -> Result<Tensor, String> {
        let k = x.dim(0).ce()?;
        let x16 = x.to_dtype(DType::F16).ce()?.reshape((k, 1)).ce()?;
        let y = w.matmul(&x16).ce()?;
        y.to_dtype(DType::F32).ce()?.flatten_all().ce()
    }

    /// One EXL3 quantized linear, entirely on device:
    /// `y = svh * (H_n @ (W_hat^T @ (H_k @ (x * suh))))`.
    fn q_matmul(&self, parts: &(Tensor, Tensor, Tensor), x: &Tensor) -> Result<Tensor, String> {
        let (w, suh, svh) = parts;
        let xr = self.hadamard(&x.mul(suh).ce()?)?;
        let t = Self::matmul_w(w, &xr)?;
        let t = self.hadamard(&t)?;
        t.mul(svh).ce()
    }

    fn linear(&self, name: &str, x: &Tensor) -> Result<Tensor, String> {
        if let Some(w) = self.mat(name) {
            return Self::matmul_w(w, x);
        }
        if let Some(q) = self.q(name) {
            return self.q_matmul(q, x);
        }
        Err(format!("missing linear weight `{}`", name))
    }

    fn embedding(&self, token: u32) -> Result<Tensor, String> {
        let name = format!("{}embed_tokens.weight", self.prefix);
        let table = self
            .mats
            .get(&name)
            .or_else(|| self.mats.get("model.embed_tokens.weight"))
            .ok_or_else(|| "missing embedding tensor `embed_tokens.weight`".to_string())?;
        let row = table.get(token as usize).ce()?;
        row.to_dtype(DType::F32).ce()
    }

    fn rms_norm(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, String> {
        let ss = x.sqr().ce()?.mean_keepdim(0).ce()?;
        let scale = ss
            .affine(1.0, self.cfg.rms_norm_eps as f64)
            .ce()?
            .sqrt()
            .ce()?
            .recip()
            .ce()?;
        let w1 = w.affine(1.0, 1.0).ce()?;
        x.broadcast_mul(&scale).ce()?.mul(&w1).ce()
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
        let a = x2.narrow(1, 0, half).ce()?;
        let b = x2.narrow(1, half, half).ce()?;
        let c = self
            .rope_cos
            .narrow(0, pos, half)
            .ce()?
            .reshape((1, half))
            .ce()?;
        let s = self
            .rope_sin
            .narrow(0, pos, half)
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
        Tensor::cat(&[na, nb], 1).ce()?.flatten_all().ce()
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
}

/// Compact post-prefill snapshot (only the used KV prefix).
pub(crate) struct GpuSnapshot {
    position: usize,
    full: Vec<(Tensor, Tensor)>,
    linear: Vec<(Tensor, usize, Tensor)>,
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

    /// Run the tokens on device and copy back only the final logits.
    pub(crate) fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Ok(vec![0.0; self.weights.cfg.vocab_size]);
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
        logits.to_vec1::<f32>().ce()
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

        let qp2 = qp.reshape((nh, 2, hd)).ce()?;
        let q = qp2.narrow(1, 0, hd).ce()?.flatten_all().ce()?;
        let gate = qp2.narrow(1, 1, hd).ce()?.flatten_all().ce()?;

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

        let tlen = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();
        let groups = nh / nkv;
        let mut heads = Vec::with_capacity(nh);
        for h in 0..nh {
            let kh = h / groups;
            let kmat = kc
                .narrow(0, 0, tlen)
                .ce()?
                .reshape((tlen, nkv, hd))
                .ce()?
                .narrow(1, kh, hd)
                .ce()?
                .to_dtype(DType::F32)
                .ce()?;
            let qh = q.narrow(0, h * hd, hd).ce()?.reshape((1, hd)).ce()?;
            let mut scores = kmat.matmul(&qh.t().ce()?).ce()?.flatten_all().ce()?;
            scores = scores.affine(scale as f64, 0.0).ce()?;
            let m = scores.max_keepdim(0).ce()?;
            let e = scores.broadcast_sub(&m).ce()?.exp().ce()?;
            let probs = e.broadcast_div(&e.sum_keepdim(0).ce()?).ce()?;
            let vmat = vc
                .narrow(0, 0, tlen)
                .ce()?
                .reshape((tlen, nkv, hd))
                .ce()?
                .narrow(1, kh, hd)
                .ce()?
                .to_dtype(DType::F32)
                .ce()?;
            let mut out = probs
                .reshape((1, tlen))
                .ce()?
                .matmul(&vmat)
                .ce()?
                .flatten_all()
                .ce()?;
            let gh = gate.narrow(0, h * hd, hd).ce()?;
            let sig = gh
                .neg()
                .ce()?
                .exp()
                .ce()?
                .affine(1.0, 1.0)
                .ce()?
                .recip()
                .ce()?;
            out = out.mul(&sig).ce()?;
            heads.push(out);
        }
        let attn = Tensor::cat(&heads, 0).ce()?;
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
        let rep = nvh / nkh;

        let alog = w.vec(&format!("layers.{}.linear_attn.A_log", layer))?;
        let dtbias = w.vec(&format!("layers.{}.linear_attn.dt_bias", layer))?;
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

        // Recurrent update, one value head at a time; all ops stay on device.
        let mut new_heads = Vec::with_capacity(nvh);
        let mut outs = Vec::with_capacity(nvh);
        let sstate_copy = sstate.copy().ce()?;
        for vh in 0..nvh {
            let kh_idx = vh / rep;
            let qh = q.narrow(0, kh_idx * kd, kd).ce()?;
            let kh = k.narrow(0, kh_idx * kd, kd).ce()?;
            let vh_vec = v.narrow(0, vh * vd, vd).ce()?;
            let s = sstate_copy.get(vh).ce()?.reshape((kd, vd)).ce()?;
            let gi = g.get(vh).ce()?.to_vec0::<f32>().ce()?;
            let bi = beta.get(vh).ce()?.to_vec0::<f32>().ce()?;
            let (ns, o) = deltanet_step(&s, &qh, &kh, &vh_vec, gi, bi, kd, vd)?;
            new_heads.push(ns.reshape((1, kd, vd)).ce()?);
            outs.push(o);
        }
        *sstate = Tensor::cat(&new_heads, 0).ce()?;
        let o_all = Tensor::cat(&outs, 0).ce()?;

        let norm_w2 = w.vec(&format!("layers.{}.linear_attn.norm.weight", layer))?;
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
            .mul(&nw)
            .ce()?
            .mul(&z2.silu().ce()?)
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

/// Map a layer number to its index in the linear-attention state list.
fn layer_idx(w: &GpuWeights, layer: usize, _is_full: bool) -> usize {
    (0..layer).filter(|&i| !w.is_full(i)).count()
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
    let mem = s
        .t()
        .ce()?
        .matmul(&kh.reshape((kd, 1)).ce()?)
        .ce()?
        .flatten_all()
        .ce()?;
    let delta = v.sub(&mem).ce()?.affine(beta as f64, 0.0).ce()?;
    let s2 = s
        .add(
            &kh.reshape((kd, 1))
                .ce()?
                .matmul(&delta.reshape((1, vd)).ce()?)
                .ce()?,
        )
        .ce()?;
    let o = s2
        .t()
        .ce()?
        .matmul(&qh.reshape((kd, 1)).ce()?)
        .ce()?
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
            assert!((got[j] - expect).abs() < 1e-4, "{} vs {}", got[j], expect);
        }
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
