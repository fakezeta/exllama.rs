//! Fused device ops for the hot decode path.
//!
//! Each op has a numerically exact CPU implementation (used by CI and as a
//! fallback) and a single-kernel CUDA implementation. If a CUDA kernel cannot
//! be loaded or launched, the caller transparently falls back to the original
//! candle-op sequence — correctness never depends on a hand-written kernel
//! being available. Real-GPU execution of the PTX still needs an A/B run on a
//! CUDA host; CI only exercises the CPU path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, InplaceOp2, Layout, Shape, Tensor,
};
use half::f16;

/// `EXLLAMA_DISABLE_FUSED=1` skips the custom kernels and uses the original
/// candle-op sequence. Useful for A/B testing on a real GPU without rebuilding.
fn fused_enabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    !*DISABLED.get_or_init(|| std::env::var_os("EXLLAMA_DISABLE_FUSED").is_some())
}

static RMS_OK: AtomicBool = AtomicBool::new(true);
static ROPE_OK: AtomicBool = AtomicBool::new(true);

fn report_once(slot: &'static OnceLock<()>, op: &str, err: &str) {
    let _ = slot.set(());
    eprintln!("[fused] {op} kernel unavailable ({err}); using fallback for this process");
}

// ---------------------------------------------------------------------------
// Packed EXL3: fused input/output H128 + sign vectors
// ---------------------------------------------------------------------------

static INPUT_H_OK: AtomicBool = AtomicBool::new(true);

/// Normalized Sylvester H128, in place, one 128-element block. This is the
/// exact butterfly used by `exl3::hadamard128` and by the CUDA kernels; keeping
/// the same operation order makes the CPU fallback a numeric reference.
pub(crate) fn butterfly_h128(v: &mut [f32]) {
    debug_assert_eq!(v.len(), 128);
    let scale = 1.0 / (128.0f32).sqrt();
    for step in [1usize, 2, 4, 8, 16, 32, 64] {
        for i in 0..128 {
            if i & step == 0 {
                let a = v[i];
                let b = v[i + step];
                v[i] = a + b;
                v[i + step] = a - b;
            }
        }
    }
    for x in v.iter_mut() {
        *x *= scale;
    }
}

/// `xr = H128(x * suh)`, applied independently to each 128-block.
pub(crate) fn input_hadamard_f32(x: &[f32], suh: &[f32]) -> Result<Vec<f32>, String> {
    if x.len() != suh.len() || x.is_empty() || !x.len().is_multiple_of(128) {
        return Err(format!(
            "input hadamard: x={} suh={}, want equal non-zero multiples of 128",
            x.len(),
            suh.len()
        ));
    }
    let mut out: Vec<f32> = x.iter().zip(suh).map(|(&a, &b)| a * b).collect();
    for blk in out.as_chunks_mut::<128>().0 {
        butterfly_h128(blk);
    }
    Ok(out)
}

/// `y = svh * H128(t)`, applied independently to each 128-block.
pub(crate) fn output_hadamard_scale_f32(t: &[f32], svh: &[f32]) -> Result<Vec<f32>, String> {
    if t.len() != svh.len() || t.is_empty() || !t.len().is_multiple_of(128) {
        return Err(format!(
            "output hadamard: t={} svh={}, want equal non-zero multiples of 128",
            t.len(),
            svh.len()
        ));
    }
    let mut out = t.to_vec();
    for blk in out.as_chunks_mut::<128>().0 {
        butterfly_h128(blk);
    }
    for (v, &s) in out.iter_mut().zip(svh) {
        *v *= s;
    }
    Ok(out)
}

/// Fused `xr = H128(x * suh)`. Replaces `x.mul(suh)` followed by a
/// reshape/matmul/flatten Hadamard with one launch. Falls back to the exact
/// butterfly if the CUDA kernel cannot be loaded or launched.
pub(crate) fn input_hadamard(x: &Tensor, suh: &Tensor) -> Result<Tensor, String> {
    if x.elem_count() != suh.elem_count() || !x.elem_count().is_multiple_of(128) {
        return Err(format!(
            "input hadamard: x={} suh={}, want equal non-zero multiples of 128",
            x.elem_count(),
            suh.elem_count()
        ));
    }
    if !fused_enabled() || !INPUT_H_OK.load(Ordering::Relaxed) {
        return input_hadamard_fallback(x, suh);
    }
    let op = FusedInputH;
    match x.apply_op2_no_bwd(suh, &op) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "input_h128", &e.to_string());
            INPUT_H_OK.store(false, Ordering::Relaxed);
            input_hadamard_fallback(x, suh)
        }
    }
}

fn input_hadamard_fallback(x: &Tensor, suh: &Tensor) -> Result<Tensor, String> {
    use candle_core::Storage;
    let read = |t: &Tensor| -> Result<Vec<f32>, String> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Cpu(c) => {
                let all = c.as_slice::<f32>().map_err(|e| e.to_string())?;
                let start = layout.start_offset();
                Ok(all[start..start + t.elem_count()].to_vec())
            }
            _ => Err("input_h128 fallback needs CPU tensors".into()),
        }
    };
    let out = input_hadamard_f32(&read(x)?, &read(suh)?)?;
    Tensor::from_vec(out, (x.elem_count(),), x.device()).map_err(|e| e.to_string())
}

struct FusedInputH;

impl CustomOp2 for FusedInputH {
    fn name(&self) -> &'static str {
        "exlama_fused_input_h128"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let x = s1.as_slice::<f32>()?;
        let su = s2.as_slice::<f32>()?;
        let out = input_hadamard_f32(
            &x[l1.start_offset()..l1.start_offset() + l1.shape().elem_count()],
            &su[l2.start_offset()..l2.start_offset() + l2.shape().elem_count()],
        )
        .map_err(candle_core::Error::Msg)?;
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let x = s1.as_cuda_slice::<f32>()?;
        let su = s2.as_cuda_slice::<f32>()?;
        let len = l1.shape().elem_count();
        let out = unsafe { s1.device.alloc::<f32>(len)? };
        let func = s1.device.get_or_load_custom_func(
            "exl3_input_h128",
            "exllama_exl3_h128",
            INPUT_H_PTX,
        )?;
        let blocks_u = (len / 128) as u32;
        let sx = l1.start_offset() as u32;
        let ss = l2.start_offset() as u32;
        let after = 0u32;
        let scale = 1.0f32 / (128.0f32).sqrt();
        let mut args = func.builder();
        args.arg(x)
            .arg(su)
            .arg(&out)
            .arg(&blocks_u)
            .arg(&scale)
            .arg(&sx)
            .arg(&ss)
            .arg(&after);
        let cfg = LaunchConfig {
            grid_dim: (blocks_u, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * 4,
        };
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            l1.shape().clone(),
        ))
    }
}

#[cfg(feature = "gpu-cuda")]
const INPUT_H_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry exl3_input_h128(
    .param .u64 p_x,
    .param .u64 p_su,
    .param .u64 p_out,
    .param .u32 p_blocks,
    .param .f32 p_scale,
    .param .u32 p_start_x,
    .param .u32 p_start_su,
    .param .u32 p_after
)
{
    .reg .u32 %blk, %tidv, %idx, %ix, %is, %off, %off2, %partner, %bit, %after;
    .reg .u32 %sbase, %saddr;
    .reg .u64 %rd1, %rd2, %rd3, %rd4;
    .reg .f32 %val, %other, %scale, %sv;
    .reg .pred %padd, %pA, %pB;
    .shared .f32 sdata[128];

    mov.u32 %sbase, sdata;

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_su];
    ld.param.u64 %rd3, [p_out];
    ld.param.f32 %scale, [p_scale];
    ld.param.u32 %ix, [p_start_x];
    ld.param.u32 %is, [p_start_su];
    ld.param.u32 %after, [p_after];
    mov.u32 %blk, %ctaid.x;
    mov.u32 %tidv, %tid.x;
    mad.lo.u32 %idx, %blk, 128, %tidv;
    add.u32 %ix, %ix, %idx;
    add.u32 %is, %is, %idx;

    mul.wide.u32 %rd4, %ix, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.f32 %val, [%rd4];
    mul.wide.u32 %rd4, %is, 4;
    add.u64 %rd4, %rd2, %rd4;
    ld.global.f32 %sv, [%rd4];
    setp.eq.u32 %pA, %after, 0;
    @%pA mul.f32 %val, %val, %sv;

    mul.lo.u32 %off, %tidv, 4;
    add.u32 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %val;
    bar.sync 0;

    // Seven butterfly strides; sign is (-1)^(tid & stride), matching
    // exl3::hadamard128 exactly.
    xor.b32 %partner, %tidv, 1; mul.lo.u32 %off2, %partner, 4;
    add.u32 %saddr, %sbase, %off2;
    ld.shared.f32 %other, [%saddr]; and.b32 %bit, %tidv, 1;
    setp.eq.u32 %padd, %bit, 0;
    @%padd add.f32 %val, %val, %other;
    @!%padd sub.f32 %val, %other, %val;
    bar.sync 0; add.u32 %saddr, %sbase, %off; st.shared.f32 [%saddr], %val; bar.sync 0;

    xor.b32 %partner, %tidv, 2; mul.lo.u32 %off2, %partner, 4;
    add.u32 %saddr, %sbase, %off2;
    ld.shared.f32 %other, [%saddr]; and.b32 %bit, %tidv, 2;
    setp.eq.u32 %padd, %bit, 0;
    @%padd add.f32 %val, %val, %other;
    @!%padd sub.f32 %val, %other, %val;
    bar.sync 0; add.u32 %saddr, %sbase, %off; st.shared.f32 [%saddr], %val; bar.sync 0;

    xor.b32 %partner, %tidv, 4; mul.lo.u32 %off2, %partner, 4;
    add.u32 %saddr, %sbase, %off2;
    ld.shared.f32 %other, [%saddr]; and.b32 %bit, %tidv, 4;
    setp.eq.u32 %padd, %bit, 0;
    @%padd add.f32 %val, %val, %other;
    @!%padd sub.f32 %val, %other, %val;
    bar.sync 0; add.u32 %saddr, %sbase, %off; st.shared.f32 [%saddr], %val; bar.sync 0;

    xor.b32 %partner, %tidv, 8; mul.lo.u32 %off2, %partner, 4;
    add.u32 %saddr, %sbase, %off2;
    ld.shared.f32 %other, [%saddr]; and.b32 %bit, %tidv, 8;
    setp.eq.u32 %padd, %bit, 0;
    @%padd add.f32 %val, %val, %other;
    @!%padd sub.f32 %val, %other, %val;
    bar.sync 0; add.u32 %saddr, %sbase, %off; st.shared.f32 [%saddr], %val; bar.sync 0;

    xor.b32 %partner, %tidv, 16; mul.lo.u32 %off2, %partner, 4;
    add.u32 %saddr, %sbase, %off2;
    ld.shared.f32 %other, [%saddr]; and.b32 %bit, %tidv, 16;
    setp.eq.u32 %padd, %bit, 0;
    @%padd add.f32 %val, %val, %other;
    @!%padd sub.f32 %val, %other, %val;
    bar.sync 0; add.u32 %saddr, %sbase, %off; st.shared.f32 [%saddr], %val; bar.sync 0;

    xor.b32 %partner, %tidv, 32; mul.lo.u32 %off2, %partner, 4;
    add.u32 %saddr, %sbase, %off2;
    ld.shared.f32 %other, [%saddr]; and.b32 %bit, %tidv, 32;
    setp.eq.u32 %padd, %bit, 0;
    @%padd add.f32 %val, %val, %other;
    @!%padd sub.f32 %val, %other, %val;
    bar.sync 0; add.u32 %saddr, %sbase, %off; st.shared.f32 [%saddr], %val; bar.sync 0;

    xor.b32 %partner, %tidv, 64; mul.lo.u32 %off2, %partner, 4;
    add.u32 %saddr, %sbase, %off2;
    ld.shared.f32 %other, [%saddr]; and.b32 %bit, %tidv, 64;
    setp.eq.u32 %padd, %bit, 0;
    @%padd add.f32 %val, %val, %other;
    @!%padd sub.f32 %val, %other, %val;

    setp.ne.u32 %pB, %after, 0;
    @%pB mul.f32 %val, %val, %sv;
    mul.f32 %val, %val, %scale;
    mul.wide.u32 %rd4, %idx, 4;
    add.u64 %rd4, %rd3, %rd4;
    st.global.f32 [%rd4], %val;
    ret;
}
"#;

/// `out = x * rsqrt(mean(x^2) + eps) * w1`, where `w1 = 1 + w` is precomputed
/// at load. `x` is 1-D `(cols,)` or 2-D `(rows, cols)`; the reduction is always
/// over the last dim (one scale per row/head).
pub(crate) fn rms_norm(x: &Tensor, w1: &Tensor, eps: f32) -> Result<Tensor, String> {
    if !fused_enabled() || !RMS_OK.load(Ordering::Relaxed) {
        return rms_norm_fallback(x, w1, eps);
    }
    let op = FusedRmsNorm { eps };
    match x.apply_op2_no_bwd(w1, &op) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "rms_norm", &e.to_string());
            RMS_OK.store(false, Ordering::Relaxed);
            rms_norm_fallback(x, w1, eps)
        }
    }
}

fn rms_norm_fallback(x: &Tensor, w1: &Tensor, eps: f32) -> Result<Tensor, String> {
    let rank = x.rank();
    let ss = x
        .sqr()
        .map_err(|e| e.to_string())?
        .mean_keepdim(rank - 1)
        .map_err(|e| e.to_string())?;
    let scale = ss
        .affine(1.0, eps as f64)
        .map_err(|e| e.to_string())?
        .sqrt()
        .map_err(|e| e.to_string())?
        .recip()
        .map_err(|e| e.to_string())?;
    x.broadcast_mul(&scale)
        .map_err(|e| e.to_string())?
        .broadcast_mul(w1)
        .map_err(|e| e.to_string())
}

struct FusedRmsNorm {
    eps: f32,
}

impl CustomOp2 for FusedRmsNorm {
    fn name(&self) -> &'static str {
        "exllama_fused_rms_norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let x = s1.as_slice::<f32>()?;
        let w = s2.as_slice::<f32>()?;
        let dims = l1.dims();
        let (rows, cols) = if dims.len() == 2 {
            (dims[0], dims[1])
        } else {
            (1, dims[0])
        };
        let sx = l1.stride();
        let sw = l2.stride();
        let ox = l1.start_offset();
        let ow = l2.start_offset();
        let xs = if dims.len() == 2 { sx[1] } else { sx[0] };
        let ws = if sw.len() == 2 { sw[1] } else { sw[0] };
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let mut ss = 0.0f32;
            for c in 0..cols {
                let v = x[ox + r * sx[0] + c * xs];
                ss += v * v;
            }
            let inv = 1.0 / (ss / cols as f32 + self.eps).sqrt();
            for c in 0..cols {
                let xi = ox + r * sx[0] + c * xs;
                let wi = ow + c * ws;
                out[r * cols + c] = x[xi] * inv * w[wi];
            }
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        let x = s1.as_cuda_slice::<f32>()?;
        let w = s2.as_cuda_slice::<f32>()?;
        let dims = l1.dims();
        let (rows, cols) = if dims.len() == 2 {
            (dims[0], dims[1])
        } else {
            (1, dims[0])
        };
        let sx0 = if dims.len() == 2 { l1.stride()[0] } else { 0 };
        let sx1 = if dims.len() == 2 {
            l1.stride()[1]
        } else {
            l1.stride()[0]
        };
        let sw = if l2.stride().len() == 2 {
            l2.stride()[1]
        } else {
            l2.stride()[0]
        };
        let func =
            s1.device
                .get_or_load_custom_func("rmsnorm_f32", "exllama_fused_rms", RMS_NORM_PTX)?;
        let out = unsafe { s1.device.alloc::<f32>(rows * cols)? };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        // Kernel args are borrowed, so every scalar must outlive `args`.
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        let (sx0_u, sx1_u, sw_u) = (sx0 as u32, sx1 as u32, sw as u32);
        let (ox_u, ow_u) = (l1.start_offset() as u32, l2.start_offset() as u32);
        let mut args = func.builder();
        args.arg(x)
            .arg(w)
            .arg(&out)
            .arg(&rows_u)
            .arg(&cols_u)
            .arg(&sx0_u)
            .arg(&sx1_u)
            .arg(&ox_u)
            .arg(&sw_u)
            .arg(&ow_u)
            .arg(&self.eps);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            l1.shape().clone(),
        ))
    }
}

/// Partial RoPE: only the first `2 * half` dims of each head rotate; the tail
/// passes through. `x` is `(heads * head_len,)`, `cos`/`sin` are `(1, half)`.
pub(crate) fn rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_len: usize,
    half: usize,
) -> Result<Tensor, String> {
    if !fused_enabled() || !ROPE_OK.load(Ordering::Relaxed) {
        return rope_fallback(x, cos, sin, heads, head_len, half);
    }
    let op = FusedRope {
        heads,
        head_len,
        half,
    };
    match x.apply_op3_no_bwd(cos, sin, &op) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "rope", &e.to_string());
            ROPE_OK.store(false, Ordering::Relaxed);
            rope_fallback(x, cos, sin, heads, head_len, half)
        }
    }
}

fn rope_fallback(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_len: usize,
    half: usize,
) -> Result<Tensor, String> {
    let x2 = x.reshape((heads, head_len)).map_err(|e| e.to_string())?;
    let rotary = half * 2;
    let a = x2.narrow(1, 0, half).map_err(|e| e.to_string())?;
    let b = x2.narrow(1, half, half).map_err(|e| e.to_string())?;
    let na = a
        .broadcast_mul(cos)
        .map_err(|e| e.to_string())?
        .sub(&b.broadcast_mul(sin).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let nb = b
        .broadcast_mul(cos)
        .map_err(|e| e.to_string())?
        .add(&a.broadcast_mul(sin).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let rotated = Tensor::cat(&[na, nb], 1).map_err(|e| e.to_string())?;
    let out = if head_len > rotary {
        let tail = x2
            .narrow(1, rotary, head_len - rotary)
            .map_err(|e| e.to_string())?;
        Tensor::cat(&[rotated, tail], 1).map_err(|e| e.to_string())?
    } else {
        rotated
    };
    out.flatten_all().map_err(|e| e.to_string())
}

struct FusedRope {
    heads: usize,
    head_len: usize,
    half: usize,
}

impl CustomOp3 for FusedRope {
    fn name(&self) -> &'static str {
        "exllama_fused_rope"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if self.head_len < self.half * 2 {
            candle_core::bail!(
                "partial rotary dim {} exceeds head_dim {}",
                self.half * 2,
                self.head_len
            );
        }
        let x = s1.as_slice::<f32>()?;
        let cos = s2.as_slice::<f32>()?;
        let sin = s3.as_slice::<f32>()?;
        let sx = l1.stride()[0];
        let ox = l1.start_offset();
        let sc = if l2.stride().len() == 2 {
            l2.stride()[1]
        } else {
            l2.stride()[0]
        };
        let ss = if l3.stride().len() == 2 {
            l3.stride()[1]
        } else {
            l3.stride()[0]
        };
        let oc = l2.start_offset();
        let os = l3.start_offset();
        let rotary = self.half * 2;
        let mut out = vec![0.0f32; self.heads * self.head_len];
        for h in 0..self.heads {
            for d in 0..self.head_len {
                let xv = x[ox + (h * self.head_len + d) * sx];
                let v = if d < rotary {
                    // d < half uses a=d, b=d+half; d >= half uses a=d-half, b=d.
                    // `a` is always the current element x[d]; `b` is its
                    // partner across the two halves. d < half: (a*c - b*s);
                    // d >= half: (a*c + b*s), matching the original layout.
                    let (a, b) = if d < self.half {
                        (xv, x[ox + (h * self.head_len + d + self.half) * sx])
                    } else {
                        (xv, x[ox + (h * self.head_len + d - self.half) * sx])
                    };
                    let hd = if d < self.half { d } else { d - self.half };
                    let c = cos[oc + hd * sc];
                    let s = sin[os + hd * ss];
                    if d < self.half {
                        a * c - b * s
                    } else {
                        a * c + b * s
                    }
                } else {
                    xv
                };
                out[h * self.head_len + d] = v;
            }
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        let x = s1.as_cuda_slice::<f32>()?;
        let cos = s2.as_cuda_slice::<f32>()?;
        let sin = s3.as_cuda_slice::<f32>()?;
        let total = (self.heads * self.head_len) as u32;
        let func = s1
            .device
            .get_or_load_custom_func("rope_f32", "exllama_fused_rope", ROPE_PTX)?;
        let out = unsafe { s1.device.alloc::<f32>(total as usize)? };
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let sc = if l2.stride().len() == 2 {
            l2.stride()[1]
        } else {
            l2.stride()[0]
        };
        let ss = if l3.stride().len() == 2 {
            l3.stride()[1]
        } else {
            l3.stride()[0]
        };
        // Kernel args are borrowed, so every scalar must outlive `args`.
        let (heads_u, hlen_u, half_u) = (self.heads as u32, self.head_len as u32, self.half as u32);
        let (sx_u, ox_u) = (l1.stride()[0] as u32, l1.start_offset() as u32);
        let (sc_u, oc_u) = (sc as u32, l2.start_offset() as u32);
        let (ss_u, os_u) = (ss as u32, l3.start_offset() as u32);
        let mut args = func.builder();
        args.arg(x)
            .arg(cos)
            .arg(sin)
            .arg(&out)
            .arg(&heads_u)
            .arg(&hlen_u)
            .arg(&half_u)
            .arg(&sx_u)
            .arg(&ox_u)
            .arg(&sc_u)
            .arg(&oc_u)
            .arg(&ss_u)
            .arg(&os_u);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            l1.shape().clone(),
        ))
    }
}

#[cfg(feature = "gpu-cuda")]
const RMS_NORM_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry rmsnorm_f32(
    .param .u64 p_x,
    .param .u64 p_w,
    .param .u64 p_y,
    .param .u32 p_rows,
    .param .u32 p_cols,
    .param .u32 p_sx0,
    .param .u32 p_sx1,
    .param .u32 p_ox,
    .param .u32 p_sw,
    .param .u32 p_ow,
    .param .f32 p_eps
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<24>;
    .reg .f32 %f<9>;
    .reg .b64 %rd<12>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_w];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_sx0];
    ld.param.u32 %r3, [p_sx1];
    ld.param.u32 %r4, [p_ox];
    ld.param.u32 %r5, [p_sw];
    ld.param.u32 %r6, [p_ow];
    ld.param.f32 %f1, [p_eps];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mad.lo.u32 %r12, %r11, %r2, %r4;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r13, %r10;

sum_loop:
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra sum_done;
    mad.lo.u32 %r14, %r13, %r3, %r12;
    mul.wide.u32 %rd4, %r14, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f3, [%rd5];
    fma.rn.f32 %f2, %f3, %f3, %f2;
    add.u32 %r13, %r13, 256;
    bra sum_loop;

sum_done:
    mad.lo.u32 %r15, %r10, 4, 0;
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r15;
    st.shared.f32 [%r22], %f2;
    bar.sync 0;

    mov.u32 %r16, 128;
red_loop:
    setp.eq.u32 %p2, %r16, 0;
    @%p2 bra red_done;
    add.u32 %r19, %r10, %r16;
    min.u32 %r19, %r19, 255;
    mul.lo.u32 %r17, %r10, 4;
    mul.lo.u32 %r18, %r19, 4;
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r17;
    ld.shared.f32 %f4, [%r22];
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r18;
    ld.shared.f32 %f5, [%r22];
    add.f32 %f4, %f4, %f5;
    setp.lt.u32 %p3, %r10, %r16;
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r17;
    @%p3 st.shared.f32 [%r22], %f4;
    bar.sync 0;
    shr.u32 %r16, %r16, 1;
    bra red_loop;

red_done:
    ld.shared.f32 %f6, [sdata];
    cvt.rn.f32.u32 %f7, %r1;
    div.rn.f32 %f6, %f6, %f7;
    add.f32 %f6, %f6, %f1;
    rsqrt.approx.f32 %f6, %f6;
    setp.eq.u32 %p4, %r10, 0;
    @%p4 st.shared.f32 [sdata], %f6;
    bar.sync 0;
    ld.shared.f32 %f6, [sdata];

    mov.u32 %r13, %r10;
out_loop:
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra out_done;
    mad.lo.u32 %r14, %r13, %r3, %r12;
    mul.wide.u32 %rd4, %r14, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f3, [%rd5];
    mad.lo.u32 %r20, %r13, %r5, %r6;
    mul.wide.u32 %rd7, %r20, 4;
    add.u64 %rd8, %rd2, %rd7;
    ld.global.f32 %f7, [%rd8];
    mul.f32 %f3, %f3, %f6;
    mul.f32 %f3, %f3, %f7;
    mad.lo.u32 %r21, %r11, %r1, %r13;
    mul.wide.u32 %rd9, %r21, 4;
    add.u64 %rd10, %rd3, %rd9;
    st.global.f32 [%rd10], %f3;
    add.u32 %r13, %r13, 256;
    bra out_loop;

out_done:
    ret;
}
"#;

#[cfg(feature = "gpu-cuda")]
const ROPE_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry rope_f32(
    .param .u64 p_x,
    .param .u64 p_c,
    .param .u64 p_s,
    .param .u64 p_y,
    .param .u32 p_heads,
    .param .u32 p_hlen,
    .param .u32 p_half,
    .param .u32 p_sx,
    .param .u32 p_ox,
    .param .u32 p_sc,
    .param .u32 p_oc,
    .param .u32 p_ss,
    .param .u32 p_os
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<26>;
    .reg .f32 %f<9>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_c];
    ld.param.u64 %rd3, [p_s];
    ld.param.u64 %rd4, [p_y];
    ld.param.u32 %r1, [p_heads];
    ld.param.u32 %r2, [p_hlen];
    ld.param.u32 %r3, [p_half];
    ld.param.u32 %r4, [p_sx];
    ld.param.u32 %r5, [p_ox];
    ld.param.u32 %r6, [p_sc];
    ld.param.u32 %r7, [p_oc];
    ld.param.u32 %r8, [p_ss];
    ld.param.u32 %r9, [p_os];

    mov.u32 %r10, %ctaid.x;
    mov.u32 %r11, %ntid.x;
    mov.u32 %r12, %tid.x;
    mad.lo.u32 %r12, %r10, %r11, %r12;
    mul.lo.u32 %r13, %r1, %r2;
    setp.ge.u32 %p1, %r12, %r13;
    @%p1 bra done;

    div.u32 %r14, %r12, %r2;
    mul.lo.u32 %r15, %r14, %r2;
    sub.u32 %r16, %r12, %r15;

    mad.lo.u32 %r17, %r12, %r4, %r5;
    mul.wide.u32 %rd5, %r17, 4;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.f32 %f1, [%rd5];

    add.u32 %r18, %r3, %r3;
    setp.ge.u32 %p2, %r16, %r18;
    @%p2 bra tail;

    setp.lt.u32 %p3, %r16, %r3;
    // hd = d<half ? d : d-half (cos/sin index); other = d<half ? d+half : d-half.
    add.u32 %r20, %r16, %r3;
    sub.u32 %r22, %r16, %r3;
    selp.u32 %r19, %r16, %r22, %p3;
    selp.u32 %r21, %r20, %r22, %p3;

    mad.lo.u32 %r22, %r19, %r6, %r7;
    mad.lo.u32 %r23, %r19, %r8, %r9;
    mul.wide.u32 %rd6, %r22, 4;
    add.u64 %rd6, %rd2, %rd6;
    ld.global.f32 %f2, [%rd6];
    mul.wide.u32 %rd7, %r23, 4;
    add.u64 %rd7, %rd3, %rd7;
    ld.global.f32 %f3, [%rd7];

    // `other` is a dim index; add this head's base before reading x.
    add.u32 %r21, %r21, %r15;
    mad.lo.u32 %r24, %r21, %r4, %r5;
    mul.wide.u32 %rd8, %r24, 4;
    add.u64 %rd8, %rd1, %rd8;
    ld.global.f32 %f4, [%rd8];

    mul.f32 %f5, %f1, %f2;
    fma.rn.f32 %f6, %f4, %f3, %f5;
    neg.f32 %f7, %f3;
    fma.rn.f32 %f5, %f4, %f7, %f5;
    selp.f32 %f5, %f5, %f6, %p3;
    bra store;

tail:
    mov.f32 %f5, %f1;

store:
    mul.wide.u32 %rd9, %r12, 4;
    add.u64 %rd9, %rd4, %rd9;
    st.global.f32 [%rd9], %f5;

done:
    ret;
}
"#;

// ---------------------------------------------------------------------------
// Phase 2b: gated norm, DeltaNet gates, GQA softmax and sigmoid gate
// ---------------------------------------------------------------------------

static GATED_OK: AtomicBool = AtomicBool::new(true);
static GATES_OK: AtomicBool = AtomicBool::new(true);
static SOFTMAX_OK: AtomicBool = AtomicBool::new(true);
static SIGMOID_OK: AtomicBool = AtomicBool::new(true);
static QK_OK: AtomicBool = AtomicBool::new(true);
static KV_OK: AtomicBool = AtomicBool::new(true);
static CONV_OK: AtomicBool = AtomicBool::new(true);

fn read2(s: &[f32], l: &Layout, r: usize, c: usize) -> f32 {
    s[l.start_offset() + r * l.stride()[0] + c * l.stride()[1]]
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// Gated RMSNorm for the DeltaNet output: per row,
/// `o / sqrt(mean(o^2) + eps) * w * silu(z)`. `w` is the *raw* norm weight.
pub(crate) fn gated_norm(o: &Tensor, w: &Tensor, z: &Tensor, eps: f32) -> Result<Tensor, String> {
    if !fused_enabled() || !GATED_OK.load(Ordering::Relaxed) {
        return gated_norm_fallback(o, w, z, eps);
    }
    match o.apply_op3_no_bwd(w, z, &FusedGatedNorm { eps }) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "gated_norm", &e.to_string());
            GATED_OK.store(false, Ordering::Relaxed);
            gated_norm_fallback(o, w, z, eps)
        }
    }
}

fn gated_norm_fallback(o: &Tensor, w: &Tensor, z: &Tensor, eps: f32) -> Result<Tensor, String> {
    let cols = o.dim(1).map_err(|e| e.to_string())?;
    let ss = o
        .sqr()
        .map_err(|e| e.to_string())?
        .sum_keepdim(1)
        .map_err(|e| e.to_string())?;
    let scale = ss
        .affine(1.0 / cols as f64, 0.0)
        .map_err(|e| e.to_string())?
        .affine(1.0, eps as f64)
        .map_err(|e| e.to_string())?
        .sqrt()
        .map_err(|e| e.to_string())?
        .recip()
        .map_err(|e| e.to_string())?;
    o.broadcast_mul(&scale)
        .map_err(|e| e.to_string())?
        .broadcast_mul(w)
        .map_err(|e| e.to_string())?
        .broadcast_mul(&z.silu().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

struct FusedGatedNorm {
    eps: f32,
}

impl CustomOp3 for FusedGatedNorm {
    fn name(&self) -> &'static str {
        "exllama_fused_gated_norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let o = s1.as_slice::<f32>()?;
        let w = s2.as_slice::<f32>()?;
        let z = s3.as_slice::<f32>()?;
        let rows = l1.dim(0)?;
        let cols = l1.dim(1)?;
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let mut ss = 0.0f32;
            for c in 0..cols {
                let v = read2(o, l1, r, c);
                ss += v * v;
            }
            let inv = 1.0 / (ss / cols as f32 + self.eps).sqrt();
            for c in 0..cols {
                out[r * cols + c] =
                    read2(o, l1, r, c) * inv * read2(w, l2, 0, c) * silu(read2(z, l3, r, c));
            }
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let o = s1.as_cuda_slice::<f32>()?;
        let w = s2.as_cuda_slice::<f32>()?;
        let z = s3.as_cuda_slice::<f32>()?;
        let rows = l1.dim(0)?;
        let cols = l1.dim(1)?;
        let func = s1.device.get_or_load_custom_func(
            "gated_norm_f32",
            "exllama_fused_gated_norm",
            GATED_NORM_PTX,
        )?;
        let out = unsafe { s1.device.alloc::<f32>(rows * cols)? };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let cols_u = cols as u32;
        let (so0, so1, soff) = (
            l1.stride()[0] as u32,
            l1.stride()[1] as u32,
            l1.start_offset() as u32,
        );
        let (sw, woff) = (l2.stride()[1] as u32, l2.start_offset() as u32);
        let (sz0, sz1, zoff) = (
            l3.stride()[0] as u32,
            l3.stride()[1] as u32,
            l3.start_offset() as u32,
        );
        let mut args = func.builder();
        args.arg(o)
            .arg(w)
            .arg(z)
            .arg(&out)
            .arg(&cols_u)
            .arg(&so0)
            .arg(&so1)
            .arg(&soff)
            .arg(&sw)
            .arg(&woff)
            .arg(&sz0)
            .arg(&sz1)
            .arg(&zoff)
            .arg(&self.eps);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            l1.shape().clone(),
        ))
    }
}

/// DeltaNet decay `g` and write gate `beta` in one kernel.
/// `ab = a + dt_bias`, `b` and `A_log` are `(nvh,)`. Returns `(2, nvh)`:
/// row 0 is `g`, row 1 is `beta`.
pub(crate) fn deltanet_gates(ab: &Tensor, b: &Tensor, alog: &Tensor) -> Result<Tensor, String> {
    if !fused_enabled() || !GATES_OK.load(Ordering::Relaxed) {
        return deltanet_gates_fallback(ab, b, alog);
    }
    match ab.apply_op3_no_bwd(b, alog, &FusedDeltaNetGates) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "deltanet_gates", &e.to_string());
            GATES_OK.store(false, Ordering::Relaxed);
            deltanet_gates_fallback(ab, b, alog)
        }
    }
}

fn deltanet_gates_fallback(ab: &Tensor, b: &Tensor, alog: &Tensor) -> Result<Tensor, String> {
    let softplus = ab
        .abs()
        .map_err(|e| e.to_string())?
        .neg()
        .map_err(|e| e.to_string())?
        .exp()
        .map_err(|e| e.to_string())?
        .affine(1.0, 1.0)
        .map_err(|e| e.to_string())?
        .log()
        .map_err(|e| e.to_string())?
        .add(&ab.clamp(0.0, f64::INFINITY).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let g = alog
        .exp()
        .map_err(|e| e.to_string())?
        .neg()
        .map_err(|e| e.to_string())?
        .mul(&softplus)
        .map_err(|e| e.to_string())?;
    let beta = b
        .neg()
        .map_err(|e| e.to_string())?
        .exp()
        .map_err(|e| e.to_string())?
        .affine(1.0, 1.0)
        .map_err(|e| e.to_string())?
        .recip()
        .map_err(|e| e.to_string())?;
    // Match the custom op's (2, nvh) output shape so the caller's narrows work.
    let n = g.dim(0).map_err(|e| e.to_string())?;
    let g = g.reshape((1, n)).map_err(|e| e.to_string())?;
    let beta = beta.reshape((1, n)).map_err(|e| e.to_string())?;
    Tensor::cat(&[g, beta], 0).map_err(|e| e.to_string())
}

struct FusedDeltaNetGates;

impl CustomOp3 for FusedDeltaNetGates {
    fn name(&self) -> &'static str {
        "exllama_fused_deltanet_gates"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let ab = s1.as_slice::<f32>()?;
        let b = s2.as_slice::<f32>()?;
        let alog = s3.as_slice::<f32>()?;
        let n = l1.dim(0)?;
        let mut out = vec![0.0f32; 2 * n];
        for i in 0..n {
            let x = ab[l1.start_offset() + i * l1.stride()[0]];
            let ax = x.abs();
            let softplus = x.max(0.0) + (1.0 + (-ax).exp()).ln();
            out[i] = -alog[l3.start_offset() + i * l3.stride()[0]].exp() * softplus;
            let bb = b[l2.start_offset() + i * l2.stride()[0]];
            out[n + i] = 1.0 / (1.0 + (-bb).exp());
        }
        Ok((CpuStorage::F32(out), Shape::from((2, n))))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let ab = s1.as_cuda_slice::<f32>()?;
        let b = s2.as_cuda_slice::<f32>()?;
        let alog = s3.as_cuda_slice::<f32>()?;
        let n = l1.dim(0)?;
        let func = s1.device.get_or_load_custom_func(
            "deltanet_gates_f32",
            "exllama_fused_gates",
            GATES_PTX,
        )?;
        let out = unsafe { s1.device.alloc::<f32>(2 * n)? };
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (n_u, sab, oab, sb, ob, sa, oa) = (
            n as u32,
            l1.stride()[0] as u32,
            l1.start_offset() as u32,
            l2.stride()[0] as u32,
            l2.start_offset() as u32,
            l3.stride()[0] as u32,
            l3.start_offset() as u32,
        );
        let mut args = func.builder();
        args.arg(ab)
            .arg(b)
            .arg(alog)
            .arg(&out)
            .arg(&n_u)
            .arg(&sab)
            .arg(&oab)
            .arg(&sb)
            .arg(&ob)
            .arg(&sa)
            .arg(&oa);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            Shape::from((2, n)),
        ))
    }
}

/// Stable softmax over dim 1 of a `(nkv, tlen, groups)` tensor, after scaling
/// each score by `scale`.
pub(crate) fn softmax(x: &Tensor, scale: f32) -> Result<Tensor, String> {
    if !fused_enabled() || !SOFTMAX_OK.load(Ordering::Relaxed) {
        return softmax_fallback(x, scale);
    }
    match x.apply_op1_no_bwd(&FusedSoftmax { scale }) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "softmax", &e.to_string());
            SOFTMAX_OK.store(false, Ordering::Relaxed);
            softmax_fallback(x, scale)
        }
    }
}

fn softmax_fallback(x: &Tensor, scale: f32) -> Result<Tensor, String> {
    let scores = x.affine(scale as f64, 0.0).map_err(|e| e.to_string())?;
    let m = scores.max_keepdim(1).map_err(|e| e.to_string())?;
    let e = scores
        .broadcast_sub(&m)
        .map_err(|e| e.to_string())?
        .exp()
        .map_err(|e| e.to_string())?;
    e.broadcast_div(&e.sum_keepdim(1).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

struct FusedSoftmax {
    scale: f32,
}

impl CustomOp1 for FusedSoftmax {
    fn name(&self) -> &'static str {
        "exllama_fused_softmax"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let v = s.as_slice::<f32>()?;
        let (nkv, tlen, groups) = (l.dim(0)?, l.dim(1)?, l.dim(2)?);
        let mut out = vec![0.0f32; v.len()];
        for kh in 0..nkv {
            for g in 0..groups {
                let mut mx = f32::NEG_INFINITY;
                for t in 0..tlen {
                    mx = mx.max(
                        v[l.start_offset()
                            + kh * l.stride()[0]
                            + t * l.stride()[1]
                            + g * l.stride()[2]]
                            * self.scale,
                    );
                }
                let mut sum = 0.0f32;
                for t in 0..tlen {
                    let e = (v[l.start_offset()
                        + kh * l.stride()[0]
                        + t * l.stride()[1]
                        + g * l.stride()[2]]
                        * self.scale
                        - mx)
                        .exp();
                    sum += e;
                    out[kh * tlen * groups + t * groups + g] = e;
                }
                for t in 0..tlen {
                    out[kh * tlen * groups + t * groups + g] /= sum;
                }
            }
        }
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let x = s.as_cuda_slice::<f32>()?;
        let (nkv, tlen, groups) = (l.dim(0)?, l.dim(1)?, l.dim(2)?);
        let func = s.device.get_or_load_custom_func(
            "softmax_f32",
            "exllama_fused_softmax",
            SOFTMAX_PTX,
        )?;
        let out = unsafe { s.device.alloc::<f32>(x.len())? };
        let cfg = LaunchConfig {
            grid_dim: ((nkv * groups) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };
        let (tlen_u, groups_u, scale) = (tlen as u32, groups as u32, self.scale);
        let (s0, s1, s2, off) = (
            l.stride()[0] as u32,
            l.stride()[1] as u32,
            l.stride()[2] as u32,
            l.start_offset() as u32,
        );
        let mut args = func.builder();
        args.arg(x)
            .arg(&out)
            .arg(&tlen_u)
            .arg(&groups_u)
            .arg(&s0)
            .arg(&s1)
            .arg(&s2)
            .arg(&off)
            .arg(&scale);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s.device.clone()),
            l.shape().clone(),
        ))
    }
}

/// `out * sigmoid(gate)` for equally shaped tensors.
pub(crate) fn sigmoid_gate(out: &Tensor, gate: &Tensor) -> Result<Tensor, String> {
    // `out` is a transposed view in gqa_attention; both the CPU and CUDA
    // kernels below index it linearly, so make the operands contiguous first.
    let out = out.contiguous().map_err(|e| e.to_string())?;
    let gate = gate.contiguous().map_err(|e| e.to_string())?;
    if !fused_enabled() || !SIGMOID_OK.load(Ordering::Relaxed) {
        return sigmoid_gate_fallback(&out, &gate);
    }
    match out.apply_op2_no_bwd(&gate, &FusedSigmoidGate) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "sigmoid_gate", &e.to_string());
            SIGMOID_OK.store(false, Ordering::Relaxed);
            sigmoid_gate_fallback(&out, &gate)
        }
    }
}

fn sigmoid_gate_fallback(out: &Tensor, gate: &Tensor) -> Result<Tensor, String> {
    let sig = gate
        .neg()
        .map_err(|e| e.to_string())?
        .exp()
        .map_err(|e| e.to_string())?
        .affine(1.0, 1.0)
        .map_err(|e| e.to_string())?
        .recip()
        .map_err(|e| e.to_string())?;
    out.mul(&sig).map_err(|e| e.to_string())
}

struct FusedSigmoidGate;

impl CustomOp2 for FusedSigmoidGate {
    fn name(&self) -> &'static str {
        "exllama_fused_sigmoid_gate"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let a = s1.as_slice::<f32>()?;
        let b = s2.as_slice::<f32>()?;
        let n = a.len();
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let x = a[l1.start_offset() + i];
            let g = b[l2.start_offset() + i];
            out[i] = x / (1.0 + (-g).exp());
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        _l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let a = s1.as_cuda_slice::<f32>()?;
        let b = s2.as_cuda_slice::<f32>()?;
        let n = a.len();
        let func = s1.device.get_or_load_custom_func(
            "sigmoid_gate_f32",
            "exllama_fused_sigmoid_gate",
            SIGMOID_PTX,
        )?;
        let out = unsafe { s1.device.alloc::<f32>(n)? };
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_u = n as u32;
        let mut args = func.builder();
        args.arg(a).arg(b).arg(&out).arg(&n_u);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            l1.shape().clone(),
        ))
    }
}

/// L2-normalize the DeltaNet query/key heads and repeat each key head across
/// its value heads in one kernel. `q`/`k` are `(nkh * kd,)`; the result is
/// `(2, nvh, kd)` with row 0 = normalized q, row 1 = normalized k.
pub(crate) fn deltanet_qk_norm(
    q: &Tensor,
    k: &Tensor,
    nkh: usize,
    nvh: usize,
    kd: usize,
) -> Result<Tensor, String> {
    if !fused_enabled() || !QK_OK.load(Ordering::Relaxed) {
        return deltanet_qk_norm_fallback(q, k, nkh, nvh, kd);
    }
    let op = FusedDeltanetQkNorm { nkh, nvh, kd };
    match q.apply_op2_no_bwd(k, &op) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "deltanet_qk_norm", &e.to_string());
            QK_OK.store(false, Ordering::Relaxed);
            deltanet_qk_norm_fallback(q, k, nkh, nvh, kd)
        }
    }
}

fn deltanet_qk_norm_fallback(
    q: &Tensor,
    k: &Tensor,
    nkh: usize,
    nvh: usize,
    kd: usize,
) -> Result<Tensor, String> {
    let rep = nvh / nkh;
    let repeat_heads = |t: &Tensor| -> Result<Tensor, String> {
        t.reshape((nkh, kd))
            .map_err(|e| e.to_string())?
            .unsqueeze(1)
            .map_err(|e| e.to_string())?
            .repeat((1, rep, 1))
            .map_err(|e| e.to_string())?
            .reshape((nvh, kd))
            .map_err(|e| e.to_string())
    };
    let scale = 1.0 / (kd as f64).sqrt();
    let qh = repeat_heads(q)?
        .broadcast_div(
            &k_add_eps_rows_fallback(&repeat_heads(q)?)?
                .unsqueeze(1)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
        .affine(scale, 0.0)
        .map_err(|e| e.to_string())?;
    let kh = repeat_heads(k)?
        .broadcast_div(
            &k_add_eps_rows_fallback(&repeat_heads(k)?)?
                .unsqueeze(1)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    let nvh = qh.dim(0).map_err(|e| e.to_string())?;
    let kd = qh.dim(1).map_err(|e| e.to_string())?;
    Tensor::cat(&[qh, kh], 0)
        .and_then(|t| t.reshape((2, nvh, kd)))
        .map_err(|e| e.to_string())
}

fn k_add_eps_rows_fallback(v: &Tensor) -> Result<Tensor, String> {
    v.sqr()
        .map_err(|e| e.to_string())?
        .sum(1)
        .map_err(|e| e.to_string())?
        .affine(1.0, 1e-6)
        .map_err(|e| e.to_string())?
        .sqrt()
        .map_err(|e| e.to_string())
}

struct FusedDeltanetQkNorm {
    nkh: usize,
    nvh: usize,
    kd: usize,
}

impl CustomOp2 for FusedDeltanetQkNorm {
    fn name(&self) -> &'static str {
        "exllama_fused_deltanet_qk_norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let q = s1.as_slice::<f32>()?;
        let k = s2.as_slice::<f32>()?;
        let rep = self.nvh / self.nkh;
        let scale = 1.0 / (self.kd as f32).sqrt();
        let mut out = vec![0.0f32; 2 * self.nvh * self.kd];
        for vh in 0..self.nvh {
            let h = vh / rep;
            let mut qs = 0.0f32;
            let mut ks = 0.0f32;
            for d in 0..self.kd {
                let i = l1.start_offset() + h * self.kd * l1.stride()[0] + d * l1.stride()[0];
                let j = l2.start_offset() + h * self.kd * l2.stride()[0] + d * l2.stride()[0];
                qs += q[i] * q[i];
                ks += k[j] * k[j];
            }
            let qn = 1.0 / (qs + 1e-6).sqrt() * scale;
            let kn = 1.0 / (ks + 1e-6).sqrt();
            for d in 0..self.kd {
                let i = l1.start_offset() + h * self.kd * l1.stride()[0] + d * l1.stride()[0];
                let j = l2.start_offset() + h * self.kd * l2.stride()[0] + d * l2.stride()[0];
                out[vh * self.kd + d] = q[i] * qn;
                out[self.nvh * self.kd + vh * self.kd + d] = k[j] * kn;
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((2, self.nvh, self.kd))))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let q = s1.as_cuda_slice::<f32>()?;
        let k = s2.as_cuda_slice::<f32>()?;
        let func = s1.device.get_or_load_custom_func(
            "deltanet_qk_norm_f32",
            "exllama_fused_qk_norm",
            QK_NORM_PTX,
        )?;
        let out = unsafe { s1.device.alloc::<f32>(2 * self.nvh * self.kd)? };
        let cfg = LaunchConfig {
            grid_dim: (self.nvh as u32, 1, 1),
            block_dim: (self.kd as u32, 1, 1),
            shared_mem_bytes: self.kd as u32 * 4 * 2,
        };
        let (kd_u, rep_u, nvh_u) = (
            self.kd as u32,
            (self.nvh / self.nkh) as u32,
            self.nvh as u32,
        );
        let qscale = 1.0 / (self.kd as f32).sqrt();
        let (sq, oq, sk, ok) = (
            l1.stride()[0] as u32,
            l1.start_offset() as u32,
            l2.stride()[0] as u32,
            l2.start_offset() as u32,
        );
        let mut args = func.builder();
        args.arg(q)
            .arg(k)
            .arg(&out)
            .arg(&kd_u)
            .arg(&rep_u)
            .arg(&nvh_u)
            .arg(&sq)
            .arg(&oq)
            .arg(&sk)
            .arg(&ok)
            .arg(&qscale);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            Shape::from((2, self.nvh, self.kd)),
        ))
    }
}

#[cfg(feature = "gpu-cuda")]
const QK_NORM_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry deltanet_qk_norm_f32(
    .param .u64 p_q, .param .u64 p_k, .param .u64 p_y,
    .param .u32 p_kd, .param .u32 p_rep, .param .u32 p_nvh,
    .param .u32 p_sq, .param .u32 p_oq,
    .param .u32 p_sk, .param .u32 p_ok,
    .param .f32 p_qscale
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<16>;
    .reg .f32 %f<6>;
    .reg .b64 %rd<8>;
    .shared .align 4 .f32 sdata[2];

    ld.param.u64 %rd1, [p_q];
    ld.param.u64 %rd2, [p_k];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_kd];
    ld.param.u32 %r2, [p_rep];
    ld.param.u32 %r5, [p_nvh];
    ld.param.u32 %r3, [p_sq];
    ld.param.u32 %r4, [p_oq];
    ld.param.u32 %r6, [p_sk];
    ld.param.u32 %r7, [p_ok];
    ld.param.f32 %f5, [p_qscale];

    mov.u32 %r8, %ctaid.x;
    div.u32 %r9, %r8, %r2;
    mov.u32 %r10, %tid.x;
    mad.lo.u32 %r11, %r9, %r1, 0;
    mad.lo.u32 %r11, %r11, %r3, %r4;
    mad.lo.u32 %r12, %r9, %r1, 0;
    mad.lo.u32 %r12, %r12, %r6, %r7;

    setp.ne.u32 %p1, %r10, 0;
    @%p1 bra reduce_done;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r13, 0;
sum_loop:
    setp.ge.u32 %p2, %r13, %r1;
    @%p2 bra sum_done;
    mad.lo.u32 %r14, %r13, %r3, %r11;
    mul.wide.u32 %rd4, %r14, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.f32 %f3, [%rd4];
    fma.rn.f32 %f1, %f3, %f3, %f1;
    mad.lo.u32 %r14, %r13, %r6, %r12;
    mul.wide.u32 %rd5, %r14, 4;
    add.u64 %rd5, %rd2, %rd5;
    ld.global.f32 %f4, [%rd5];
    fma.rn.f32 %f2, %f4, %f4, %f2;
    add.u32 %r13, %r13, 1;
    bra sum_loop;
sum_done:
    st.shared.f32 [sdata], %f1;
    st.shared.f32 [sdata+4], %f2;
reduce_done:
    bar.sync 0;
    ld.shared.f32 %f3, [sdata];
    ld.shared.f32 %f4, [sdata+4];
    add.f32 %f3, %f3, 1e-6;
    add.f32 %f4, %f4, 1e-6;
    rsqrt.approx.f32 %f3, %f3;
    rsqrt.approx.f32 %f4, %f4;
    mul.f32 %f3, %f3, %f5;

    mad.lo.u32 %r14, %r10, %r3, %r11;
    mul.wide.u32 %rd4, %r14, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.f32 %f1, [%rd4];
    mad.lo.u32 %r14, %r10, %r6, %r12;
    mul.wide.u32 %rd5, %r14, 4;
    add.u64 %rd5, %rd2, %rd5;
    ld.global.f32 %f2, [%rd5];
    mul.f32 %f1, %f1, %f3;
    mul.f32 %f2, %f2, %f4;

    mad.lo.u32 %r14, %r8, %r1, 0;
    mad.lo.u32 %r14, %r14, 1, %r10;
    mul.wide.u32 %rd6, %r14, 4;
    add.u64 %rd6, %rd3, %rd6;
    st.global.f32 [%rd6], %f1;
    mad.lo.u32 %r14, %r5, %r1, %r14;
    mul.wide.u32 %rd6, %r14, 4;
    add.u64 %rd6, %rd3, %rd6;
    st.global.f32 [%rd6], %f2;
    ret;
}
"#;

#[cfg(feature = "gpu-cuda")]
const GATED_NORM_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry gated_norm_f32(
    .param .u64 p_o, .param .u64 p_w, .param .u64 p_z, .param .u64 p_y,
    .param .u32 p_cols,
    .param .u32 p_so0, .param .u32 p_so1, .param .u32 p_oo,
    .param .u32 p_sw, .param .u32 p_wo,
    .param .u32 p_sz0, .param .u32 p_sz1, .param .u32 p_zo,
    .param .f32 p_eps
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<24>;
    .reg .f32 %f<9>;
    .reg .b64 %rd<12>;
    .shared .align 4 .f32 sdata[256];

    ld.param.u64 %rd1, [p_o];
    ld.param.u64 %rd2, [p_w];
    ld.param.u64 %rd3, [p_z];
    ld.param.u64 %rd4, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_so0];
    ld.param.u32 %r3, [p_so1];
    ld.param.u32 %r4, [p_oo];
    ld.param.u32 %r5, [p_sw];
    ld.param.u32 %r6, [p_wo];
    ld.param.u32 %r7, [p_sz0];
    ld.param.u32 %r8, [p_sz1];
    ld.param.u32 %r9, [p_zo];
    ld.param.f32 %f1, [p_eps];

    mov.u32 %r10, %tid.x;
    mov.u32 %r11, %ctaid.x;
    mad.lo.u32 %r12, %r11, %r2, %r4;
    mov.f32 %f2, 0f00000000;
    mov.u32 %r13, %r10;

sum_loop:
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra sum_done;
    mad.lo.u32 %r14, %r13, %r3, %r12;
    mul.wide.u32 %rd5, %r14, 4;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.f32 %f3, [%rd5];
    fma.rn.f32 %f2, %f3, %f3, %f2;
    add.u32 %r13, %r13, 256;
    bra sum_loop;

sum_done:
    mad.lo.u32 %r15, %r10, 4, 0;
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r15;
    st.shared.f32 [%r22], %f2;
    bar.sync 0;
    mov.u32 %r16, 128;
red_loop:
    setp.eq.u32 %p2, %r16, 0;
    @%p2 bra red_done;
    add.u32 %r19, %r10, %r16;
    min.u32 %r19, %r19, 255;
    mul.lo.u32 %r17, %r10, 4;
    mul.lo.u32 %r18, %r19, 4;
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r17;
    ld.shared.f32 %f4, [%r22];
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r18;
    ld.shared.f32 %f5, [%r22];
    add.f32 %f4, %f4, %f5;
    setp.lt.u32 %p3, %r10, %r16;
    mov.u32 %r22, sdata;
    add.u32 %r22, %r22, %r17;
    @%p3 st.shared.f32 [%r22], %f4;
    bar.sync 0;
    shr.u32 %r16, %r16, 1;
    bra red_loop;

red_done:
    ld.shared.f32 %f6, [sdata];
    cvt.rn.f32.u32 %f7, %r1;
    div.rn.f32 %f6, %f6, %f7;
    add.f32 %f6, %f6, %f1;
    rsqrt.approx.f32 %f6, %f6;
    setp.eq.u32 %p4, %r10, 0;
    @%p4 st.shared.f32 [sdata], %f6;
    bar.sync 0;
    ld.shared.f32 %f6, [sdata];

    mov.u32 %r13, %r10;
out_loop:
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra out_done;
    mad.lo.u32 %r14, %r13, %r3, %r12;
    mul.wide.u32 %rd5, %r14, 4;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.f32 %f3, [%rd5];
    mad.lo.u32 %r20, %r13, %r5, %r6;
    mul.wide.u32 %rd6, %r20, 4;
    add.u64 %rd6, %rd2, %rd6;
    ld.global.f32 %f7, [%rd6];
    // z index = row*s_z0 + c*s_z1 + z_offset
    mad.lo.u32 %r21, %r11, %r7, %r9;
    mad.lo.u32 %r21, %r13, %r8, %r21;
    mul.wide.u32 %rd7, %r21, 4;
    add.u64 %rd7, %rd3, %rd7;
    ld.global.f32 %f8, [%rd7];
    // silu(z) = z / (1 + exp(-z))
    neg.f32 %f4, %f8;
    mul.f32 %f4, %f4, 1.44269504;
    ex2.approx.f32 %f4, %f4;
    add.f32 %f4, %f4, 1.0;
    rcp.approx.f32 %f4, %f4;
    mul.f32 %f8, %f8, %f4;
    mul.f32 %f3, %f3, %f6;
    mul.f32 %f3, %f3, %f7;
    mul.f32 %f3, %f3, %f8;
    mad.lo.u32 %r23, %r11, %r1, %r13;
    mul.wide.u32 %rd8, %r23, 4;
    add.u64 %rd8, %rd4, %rd8;
    st.global.f32 [%rd8], %f3;
    add.u32 %r13, %r13, 256;
    bra out_loop;

out_done:
    ret;
}
"#;

#[cfg(feature = "gpu-cuda")]
const GATES_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry deltanet_gates_f32(
    .param .u64 p_ab, .param .u64 p_b, .param .u64 p_a, .param .u64 p_y,
    .param .u32 p_n,
    .param .u32 p_sab, .param .u32 p_oab,
    .param .u32 p_sb, .param .u32 p_ob,
    .param .u32 p_sa, .param .u32 p_oa
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<14>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [p_ab];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_a];
    ld.param.u64 %rd4, [p_y];
    ld.param.u32 %r1, [p_n];
    ld.param.u32 %r2, [p_sab];
    ld.param.u32 %r3, [p_oab];
    ld.param.u32 %r4, [p_sb];
    ld.param.u32 %r5, [p_ob];
    ld.param.u32 %r6, [p_sa];
    ld.param.u32 %r7, [p_oa];
    mov.u32 %r8, %tid.x;

loop:
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra done;
    mad.lo.u32 %r9, %r8, %r2, %r3;
    mul.wide.u32 %rd5, %r9, 4;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.f32 %f1, [%rd5];
    // softplus = max(x,0) + log(1 + exp(-|x|))
    abs.f32 %f2, %f1;
    neg.f32 %f2, %f2;
    mul.f32 %f2, %f2, 1.44269504;
    ex2.approx.f32 %f2, %f2;
    add.f32 %f2, %f2, 1.0;
    lg2.approx.f32 %f2, %f2;
    // lg2 -> ln: multiply by ln(2)
    mul.f32 %f2, %f2, 0.69314718;
    max.f32 %f3, %f1, 0f00000000;
    add.f32 %f2, %f2, %f3;
    mad.lo.u32 %r10, %r8, %r6, %r7;
    mul.wide.u32 %rd6, %r10, 4;
    add.u64 %rd6, %rd3, %rd6;
    ld.global.f32 %f4, [%rd6];
    mul.f32 %f4, %f4, 1.44269504;
    ex2.approx.f32 %f4, %f4;
    neg.f32 %f4, %f4;
    mul.f32 %f2, %f2, %f4;
    // g row
    mul.wide.u32 %rd7, %r8, 4;
    add.u64 %rd7, %rd4, %rd7;
    st.global.f32 [%rd7], %f2;
    // beta = 1/(1+exp(-b))
    mad.lo.u32 %r11, %r8, %r4, %r5;
    mul.wide.u32 %rd8, %r11, 4;
    add.u64 %rd8, %rd2, %rd8;
    ld.global.f32 %f5, [%rd8];
    neg.f32 %f5, %f5;
    mul.f32 %f5, %f5, 1.44269504;
    ex2.approx.f32 %f5, %f5;
    add.f32 %f5, %f5, 1.0;
    rcp.approx.f32 %f5, %f5;
    // out[n + i] = beta
    add.u32 %r12, %r1, %r8;
    mul.wide.u32 %rd9, %r12, 4;
    add.u64 %rd9, %rd4, %rd9;
    st.global.f32 [%rd9], %f5;
    add.u32 %r8, %r8, 256;
    bra loop;

done:
    ret;
}
"#;

#[cfg(feature = "gpu-cuda")]
const SOFTMAX_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry softmax_f32(
    .param .u64 p_x, .param .u64 p_y,
    .param .u32 p_tlen, .param .u32 p_groups,
    .param .u32 p_s0, .param .u32 p_s1, .param .u32 p_s2, .param .u32 p_off,
    .param .f32 p_scale
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<20>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<10>;
    .shared .align 4 .f32 sdata[512];

    ld.param.u64 %rd1, [p_x];
    ld.param.u64 %rd2, [p_y];
    ld.param.u32 %r1, [p_tlen];
    ld.param.u32 %r2, [p_groups];
    ld.param.u32 %r3, [p_s0];
    ld.param.u32 %r4, [p_s1];
    ld.param.u32 %r5, [p_s2];
    ld.param.u32 %r6, [p_off];
    ld.param.f32 %f1, [p_scale];
    mov.u32 %r7, %ctaid.x;
    // kh = block / groups, g = block % groups
    div.u32 %r8, %r7, %r2;
    mul.lo.u32 %r9, %r8, %r2;
    sub.u32 %r10, %r7, %r9;
    mov.u32 %r11, %tid.x;
    // base = off + kh*s0 + g*s2
    mad.lo.u32 %r12, %r8, %r3, %r6;
    mad.lo.u32 %r12, %r10, %r5, %r12;
    // first pass: max
    mov.f32 %f2, 0fFF800000;
    mov.u32 %r13, %r11;
max_loop:
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra max_done;
    mad.lo.u32 %r14, %r13, %r4, %r12;
    mul.wide.u32 %rd3, %r14, 4;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f3, [%rd3];
    mul.f32 %f3, %f3, %f1;
    max.f32 %f2, %f2, %f3;
    add.u32 %r13, %r13, 256;
    bra max_loop;
max_done:
    mad.lo.u32 %r15, %r11, 4, 0;
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r15;
    st.shared.f32 [%r16], %f2;
    bar.sync 0;
    mov.u32 %r17, 128;
red_max:
    setp.eq.u32 %p2, %r17, 0;
    @%p2 bra red_max_done;
    add.u32 %r18, %r11, %r17;
    min.u32 %r18, %r18, 255;
    mul.lo.u32 %r19, %r11, 4;
    mul.lo.u32 %r15, %r18, 4;
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r19;
    ld.shared.f32 %f4, [%r16];
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r15;
    ld.shared.f32 %f5, [%r16];
    max.f32 %f4, %f4, %f5;
    setp.lt.u32 %p3, %r11, %r17;
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r19;
    @%p3 st.shared.f32 [%r16], %f4;
    bar.sync 0;
    shr.u32 %r17, %r17, 1;
    bra red_max;
red_max_done:
    ld.shared.f32 %f6, [sdata];
    // second pass: exp and sum into upper half of shared
    mov.f32 %f7, 0f00000000;
    mov.u32 %r13, %r11;
exp_loop:
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra exp_done;
    mad.lo.u32 %r14, %r13, %r4, %r12;
    mul.wide.u32 %rd3, %r14, 4;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f3, [%rd3];
    mul.f32 %f3, %f3, %f1;
    sub.f32 %f3, %f3, %f6;
    mul.f32 %f3, %f3, 1.44269504;
    ex2.approx.f32 %f3, %f3;
    add.f32 %f7, %f7, %f3;
    add.u32 %r13, %r13, 256;
    bra exp_loop;
exp_done:
    // each thread's partial sum goes into sdata[tid] for the reduction
    mad.lo.u32 %r15, %r11, 4, 0;
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r15;
    st.shared.f32 [%r16], %f7;
    bar.sync 0;
    mov.u32 %r17, 128;
red_sum:
    setp.eq.u32 %p2, %r17, 0;
    @%p2 bra red_sum_done;
    add.u32 %r18, %r11, %r17;
    min.u32 %r18, %r18, 255;
    mul.lo.u32 %r19, %r11, 4;
    mul.lo.u32 %r15, %r18, 4;
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r19;
    ld.shared.f32 %f4, [%r16];
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r15;
    ld.shared.f32 %f5, [%r16];
    add.f32 %f4, %f4, %f5;
    setp.lt.u32 %p3, %r11, %r17;
    mov.u32 %r16, sdata;
    add.u32 %r16, %r16, %r19;
    @%p3 st.shared.f32 [%r16], %f4;
    bar.sync 0;
    shr.u32 %r17, %r17, 1;
    bra red_sum;
red_sum_done:
    ld.shared.f32 %f7, [sdata];
    rcp.approx.f32 %f7, %f7;
    // write pass: recompute exp and store normalized
    mov.u32 %r13, %r11;
write_loop:
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra done;
    mad.lo.u32 %r14, %r13, %r4, %r12;
    mul.wide.u32 %rd3, %r14, 4;
    add.u64 %rd3, %rd1, %rd3;
    ld.global.f32 %f3, [%rd3];
    mul.f32 %f3, %f3, %f1;
    sub.f32 %f3, %f3, %f6;
    mul.f32 %f3, %f3, 1.44269504;
    ex2.approx.f32 %f3, %f3;
    mul.f32 %f3, %f3, %f7;
    // output index = kh*tlen*groups + t*groups + g (block = kh*groups + g)
    mad.lo.u32 %r19, %r8, %r1, %r13;
    mad.lo.u32 %r19, %r19, %r2, %r10;
    mul.wide.u32 %rd4, %r19, 4;
    add.u64 %rd4, %rd2, %rd4;
    st.global.f32 [%rd4], %f3;
    add.u32 %r13, %r13, 256;
    bra write_loop;

done:
    ret;
}
"#;

#[cfg(feature = "gpu-cuda")]
const SIGMOID_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry sigmoid_gate_f32(
    .param .u64 p_a, .param .u64 p_b, .param .u64 p_y, .param .u32 p_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_y];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r4, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r4, %r1;
    @%p1 bra done;
    mul.wide.u32 %rd4, %r4, 4;
    add.u64 %rd4, %rd1, %rd4;
    ld.global.f32 %f1, [%rd4];
    mul.wide.u32 %rd5, %r4, 4;
    add.u64 %rd5, %rd2, %rd5;
    ld.global.f32 %f2, [%rd5];
    neg.f32 %f2, %f2;
    mul.f32 %f2, %f2, 1.44269504;
    ex2.approx.f32 %f2, %f2;
    add.f32 %f2, %f2, 1.0;
    rcp.approx.f32 %f2, %f2;
    mul.f32 %f1, %f1, %f2;
    mul.wide.u32 %rd5, %r4, 4;
    add.u64 %rd5, %rd3, %rd5;
    st.global.f32 [%rd5], %f1;
done:
    ret;
}
"#;

/// Write one fp16 KV row in place instead of copying the whole cache with
/// `slice_scatter` on every generated token.
pub(crate) fn write_kv_row(dst: &mut Tensor, row: &Tensor, pos: usize) -> Result<(), String> {
    if fused_enabled() && KV_OK.load(Ordering::Relaxed) {
        match dst.inplace_op2(row, &WriteKvRow { pos }) {
            Ok(()) => return Ok(()),
            Err(e) => {
                static REPORTED: OnceLock<()> = OnceLock::new();
                report_once(&REPORTED, "write_kv_row", &e.to_string());
                KV_OK.store(false, Ordering::Relaxed);
            }
        }
    }
    // Fallback: rebuild the cache with one row replaced.
    let cols = dst.dim(1).map_err(|e| e.to_string())?;
    let r = row.reshape((1, cols)).map_err(|e| e.to_string())?;
    let zeros = Tensor::zeros(dst.shape(), DType::F16, dst.device()).map_err(|e| e.to_string())?;
    *dst = zeros.slice_scatter(&r, 0, pos).map_err(|e| e.to_string())?;
    Ok(())
}

struct WriteKvRow {
    pos: usize,
}

impl InplaceOp2 for WriteKvRow {
    fn name(&self) -> &'static str {
        "exllama_fused_write_kv_row"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<()> {
        let CpuStorage::F16(dst) = s1 else {
            candle_core::bail!("write_kv_row: destination is not fp16")
        };
        let src = s2.as_slice::<f16>()?;
        let cols = l1.dim(1)?;
        for i in 0..cols {
            let doff = l1.start_offset() + self.pos * l1.stride()[0] + i * l1.stride()[1];
            let soff = l2.start_offset() + i * l2.stride()[1];
            dst[doff] = src[soff];
        }
        Ok(())
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &mut candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let dev = s1.device.clone();
        let dst = s1.as_cuda_slice_mut::<f16>()?;
        let src = s2.as_cuda_slice::<f16>()?;
        let cols = l1.dim(1)?;
        let func = dev.get_or_load_custom_func(
            "write_kv_row_f16",
            "exllama_fused_write_kv_row",
            WRITE_KV_PTX,
        )?;
        let cfg = LaunchConfig::for_num_elems(cols as u32);
        let (cols_u, pos_u) = (cols as u32, self.pos as u32);
        let (sd0, sd1, od) = (
            l1.stride()[0] as u32,
            l1.stride()[1] as u32,
            l1.start_offset() as u32,
        );
        let (ss, os) = (l2.stride()[1] as u32, l2.start_offset() as u32);
        let mut args = func.builder();
        args.arg(&mut *dst)
            .arg(src)
            .arg(&cols_u)
            .arg(&pos_u)
            .arg(&sd0)
            .arg(&sd1)
            .arg(&od)
            .arg(&ss)
            .arg(&os);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok(())
    }
}

#[cfg(feature = "gpu-cuda")]
const WRITE_KV_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry write_kv_row_f16(
    .param .u64 p_dst, .param .u64 p_src,
    .param .u32 p_cols, .param .u32 p_pos,
    .param .u32 p_sd0, .param .u32 p_sd1, .param .u32 p_od,
    .param .u32 p_ss, .param .u32 p_os
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<11>;
    .reg .b16 %rs<2>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [p_dst];
    ld.param.u64 %rd2, [p_src];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_pos];
    ld.param.u32 %r3, [p_sd0];
    ld.param.u32 %r4, [p_sd1];
    ld.param.u32 %r5, [p_od];
    ld.param.u32 %r6, [p_ss];
    ld.param.u32 %r7, [p_os];
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r0, %ntid.x;
    mov.u32 %r9, %tid.x;
    mad.lo.u32 %r8, %r8, %r0, %r9;
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra done;
    mad.lo.u32 %r2, %r2, %r3, %r5;
    mad.lo.u32 %r2, %r8, %r4, %r2;
    mul.lo.u32 %r2, %r2, 2;
    mul.wide.u32 %rd5, %r2, 1;
    add.u64 %rd3, %rd1, %rd5;
    mad.lo.u32 %r7, %r8, %r6, %r7;
    mul.lo.u32 %r7, %r7, 2;
    mul.wide.u32 %rd5, %r7, 1;
    add.u64 %rd4, %rd2, %rd5;
    ld.global.b16 %rs1, [%rd4];
    st.global.b16 [%rd3], %rs1;
done:
    ret;
}
"#;

/// Depthwise conv1d + circular window gather + SiLU in one kernel.
/// `conv` is `(hist, conv_dim)` where `hist = kernel - 1`, `cur` is the new
/// `(conv_dim,)` row, `cw` is `(conv_dim, kernel)`. Returns `(conv_dim,)`.
pub(crate) fn conv_step(
    conv: &Tensor,
    cur: &Tensor,
    cw: &Tensor,
    pos: usize,
    kernel: usize,
) -> Result<Tensor, String> {
    if !fused_enabled() || !CONV_OK.load(Ordering::Relaxed) {
        return conv_step_fallback(conv, cur, cw, pos, kernel);
    }
    let op = FusedConvStep { pos, kernel };
    match conv.apply_op3_no_bwd(cur, cw, &op) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "conv_step", &e.to_string());
            CONV_OK.store(false, Ordering::Relaxed);
            conv_step_fallback(conv, cur, cw, pos, kernel)
        }
    }
}

fn conv_step_fallback(
    conv: &Tensor,
    cur: &Tensor,
    cw: &Tensor,
    pos: usize,
    kernel: usize,
) -> Result<Tensor, String> {
    let hist = kernel - 1;
    let mut parts: Vec<Tensor> = Vec::with_capacity(kernel);
    for j in 0..hist {
        parts.push(
            conv.narrow(0, (pos + j) % hist, 1)
                .map_err(|e| e.to_string())?,
        );
    }
    parts.push(
        cur.reshape((1, cur.elem_count()))
            .map_err(|e| e.to_string())?,
    );
    let window = Tensor::cat(&parts, 0).map_err(|e| e.to_string())?;
    window
        .t()
        .map_err(|e| e.to_string())?
        .mul(cw)
        .map_err(|e| e.to_string())?
        .sum(1)
        .map_err(|e| e.to_string())?
        .silu()
        .map_err(|e| e.to_string())
}

struct FusedConvStep {
    pos: usize,
    kernel: usize,
}

impl CustomOp3 for FusedConvStep {
    fn name(&self) -> &'static str {
        "exllama_fused_conv_step"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let conv = s1.as_slice::<f32>()?;
        let cur = s2.as_slice::<f32>()?;
        let w = s3.as_slice::<f32>()?;
        let hist = self.kernel - 1;
        let dims = l1.dims();
        let cols = dims[1];
        let mut out = vec![0.0f32; cols];
        for c in 0..cols {
            let mut sum = 0.0f32;
            for j in 0..self.kernel {
                let v = if j < hist {
                    conv[l1.start_offset()
                        + ((self.pos + j) % hist) * l1.stride()[0]
                        + c * l1.stride()[1]]
                } else {
                    cur[l2.start_offset() + c * l2.stride()[0]]
                };
                sum += v * w[l3.start_offset() + c * l3.stride()[0] + j * l3.stride()[1]];
            }
            out[c] = silu(sum);
        }
        Ok((CpuStorage::F32(out), Shape::from(cols)))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let conv = s1.as_cuda_slice::<f32>()?;
        let cur = s2.as_cuda_slice::<f32>()?;
        let w = s3.as_cuda_slice::<f32>()?;
        let cols = l1.dim(1)?;
        let func = s1.device.get_or_load_custom_func(
            "conv_step_f32",
            "exllama_fused_conv_step",
            CONV_STEP_PTX,
        )?;
        let out = unsafe { s1.device.alloc::<f32>(cols)? };
        let cfg = LaunchConfig::for_num_elems(cols as u32);
        let (cols_u, kernel_u, hist_u, pos_u) = (
            cols as u32,
            self.kernel as u32,
            (self.kernel - 1) as u32,
            self.pos as u32,
        );
        let (sc0, sc1, oc) = (
            l1.stride()[0] as u32,
            l1.stride()[1] as u32,
            l1.start_offset() as u32,
        );
        let (su, ou) = (l2.stride()[0] as u32, l2.start_offset() as u32);
        let (sw0, sw1, ow) = (
            l3.stride()[0] as u32,
            l3.stride()[1] as u32,
            l3.start_offset() as u32,
        );
        let mut args = func.builder();
        args.arg(conv)
            .arg(cur)
            .arg(w)
            .arg(&out)
            .arg(&cols_u)
            .arg(&kernel_u)
            .arg(&hist_u)
            .arg(&pos_u)
            .arg(&sc0)
            .arg(&sc1)
            .arg(&oc)
            .arg(&su)
            .arg(&ou)
            .arg(&sw0)
            .arg(&sw1)
            .arg(&ow);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            Shape::from(cols),
        ))
    }
}

#[cfg(feature = "gpu-cuda")]
const CONV_STEP_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry conv_step_f32(
    .param .u64 p_conv, .param .u64 p_cur, .param .u64 p_w, .param .u64 p_y,
    .param .u32 p_cols, .param .u32 p_kernel, .param .u32 p_hist, .param .u32 p_pos,
    .param .u32 p_sc0, .param .u32 p_sc1, .param .u32 p_oc,
    .param .u32 p_su, .param .u32 p_ou,
    .param .u32 p_sw0, .param .u32 p_sw1, .param .u32 p_ow
)
{
    .reg .pred %p<5>;
    .reg .b32 %r<20>;
    .reg .f32 %f<5>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [p_conv];
    ld.param.u64 %rd2, [p_cur];
    ld.param.u64 %rd3, [p_w];
    ld.param.u64 %rd4, [p_y];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_kernel];
    ld.param.u32 %r3, [p_hist];
    ld.param.u32 %r4, [p_pos];
    ld.param.u32 %r5, [p_sc0];
    ld.param.u32 %r6, [p_sc1];
    ld.param.u32 %r7, [p_oc];
    ld.param.u32 %r8, [p_su];
    ld.param.u32 %r9, [p_ou];
    ld.param.u32 %r10, [p_sw0];
    ld.param.u32 %r11, [p_sw1];
    ld.param.u32 %r12, [p_ow];
    mov.u32 %r13, %ctaid.x;
    mov.u32 %r14, %ntid.x;
    mov.u32 %r15, %tid.x;
    mad.lo.u32 %r13, %r13, %r14, %r15;
    setp.ge.u32 %p1, %r13, %r1;
    @%p1 bra done;
    mov.f32 %f1, 0f00000000;
    mov.u32 %r16, 0;
    mov.u32 %r17, %r4;
loop:
    setp.ge.u32 %p2, %r16, %r2;
    @%p2 bra loop_done;
    setp.lt.u32 %p3, %r16, %r3;
    @%p3 bra from_hist;
    mad.lo.u32 %r18, %r13, %r8, %r9;
    mul.wide.u32 %rd5, %r18, 4;
    add.u64 %rd5, %rd2, %rd5;
    ld.global.f32 %f2, [%rd5];
    bra have_w;
from_hist:
    mad.lo.u32 %r18, %r17, %r5, %r7;
    mad.lo.u32 %r18, %r13, %r6, %r18;
    mul.wide.u32 %rd5, %r18, 4;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.f32 %f2, [%rd5];
have_w:
    mad.lo.u32 %r18, %r13, %r10, %r12;
    mad.lo.u32 %r18, %r16, %r11, %r18;
    mul.wide.u32 %rd6, %r18, 4;
    add.u64 %rd6, %rd3, %rd6;
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r16, %r16, 1;
    add.u32 %r17, %r17, 1;
    setp.eq.u32 %p4, %r17, %r3;
    @%p4 mov.u32 %r17, 0;
    bra loop;
loop_done:
    // silu: s / (1 + exp(-s))
    neg.f32 %f2, %f1;
    mul.f32 %f2, %f2, 1.44269504;
    ex2.approx.f32 %f2, %f2;
    add.f32 %f2, %f2, 1.0;
    rcp.approx.f32 %f2, %f2;
    mul.f32 %f1, %f1, %f2;
    mul.wide.u32 %rd7, %r13, 4;
    add.u64 %rd7, %rd4, %rd7;
    st.global.f32 [%rd7], %f1;
done:
    ret;
}
"#;

/// In-place fp32 row write for the DeltaNet conv circular buffer.
pub(crate) fn write_conv_row(dst: &mut Tensor, row: &Tensor, pos: usize) -> Result<(), String> {
    if fused_enabled() && KV_OK.load(Ordering::Relaxed) {
        match dst.inplace_op2(row, &WriteRowF32 { pos }) {
            Ok(()) => return Ok(()),
            Err(e) => {
                static REPORTED: OnceLock<()> = OnceLock::new();
                report_once(&REPORTED, "write_conv_row", &e.to_string());
                KV_OK.store(false, Ordering::Relaxed);
            }
        }
    }
    let cols = dst.dim(1).map_err(|e| e.to_string())?;
    let r = row.reshape((1, cols)).map_err(|e| e.to_string())?;
    let zeros = Tensor::zeros(dst.shape(), DType::F32, dst.device()).map_err(|e| e.to_string())?;
    *dst = zeros.slice_scatter(&r, 0, pos).map_err(|e| e.to_string())?;
    Ok(())
}

struct WriteRowF32 {
    pos: usize,
}

impl InplaceOp2 for WriteRowF32 {
    fn name(&self) -> &'static str {
        "exllama_fused_write_row_f32"
    }

    fn cpu_fwd(
        &self,
        s1: &mut CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<()> {
        let CpuStorage::F32(dst) = s1 else {
            candle_core::bail!("write_conv_row: destination is not fp32")
        };
        let src = s2.as_slice::<f32>()?;
        let cols = l1.dim(1)?;
        let srow = if l2.stride().len() == 2 {
            l2.stride()[1]
        } else {
            l2.stride()[0]
        };
        for i in 0..cols {
            let doff = l1.start_offset() + self.pos * l1.stride()[0] + i * l1.stride()[1];
            let soff = l2.start_offset() + i * srow;
            dst[doff] = src[soff];
        }
        Ok(())
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &mut candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let dev = s1.device.clone();
        let dst = s1.as_cuda_slice_mut::<f32>()?;
        let src = s2.as_cuda_slice::<f32>()?;
        let cols = l1.dim(1)?;
        let func = dev.get_or_load_custom_func(
            "write_row_f32",
            "exllama_fused_write_row_f32",
            WRITE_ROW_F32_PTX,
        )?;
        let cfg = LaunchConfig::for_num_elems(cols as u32);
        let (cols_u, pos_u) = (cols as u32, self.pos as u32);
        let (sd0, sd1, od) = (
            l1.stride()[0] as u32,
            l1.stride()[1] as u32,
            l1.start_offset() as u32,
        );
        let ss = if l2.stride().len() == 2 {
            l2.stride()[1]
        } else {
            l2.stride()[0]
        };
        let (ss_u, os) = (ss as u32, l2.start_offset() as u32);
        let mut args = func.builder();
        args.arg(&mut *dst)
            .arg(src)
            .arg(&cols_u)
            .arg(&pos_u)
            .arg(&sd0)
            .arg(&sd1)
            .arg(&od)
            .arg(&ss_u)
            .arg(&os);
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok(())
    }
}

#[cfg(feature = "gpu-cuda")]
const WRITE_ROW_F32_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry write_row_f32(
    .param .u64 p_dst, .param .u64 p_src,
    .param .u32 p_cols, .param .u32 p_pos,
    .param .u32 p_sd0, .param .u32 p_sd1, .param .u32 p_od,
    .param .u32 p_ss, .param .u32 p_os
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<11>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [p_dst];
    ld.param.u64 %rd2, [p_src];
    ld.param.u32 %r1, [p_cols];
    ld.param.u32 %r2, [p_pos];
    ld.param.u32 %r3, [p_sd0];
    ld.param.u32 %r4, [p_sd1];
    ld.param.u32 %r5, [p_od];
    ld.param.u32 %r6, [p_ss];
    ld.param.u32 %r7, [p_os];
    mov.u32 %r8, %ctaid.x;
    mov.u32 %r0, %ntid.x;
    mov.u32 %r9, %tid.x;
    mad.lo.u32 %r8, %r8, %r0, %r9;
    setp.ge.u32 %p1, %r8, %r1;
    @%p1 bra done;
    mad.lo.u32 %r2, %r2, %r3, %r5;
    mad.lo.u32 %r2, %r8, %r4, %r2;
    mul.lo.u32 %r2, %r2, 4;
    mul.wide.u32 %rd5, %r2, 1;
    add.u64 %rd3, %rd1, %rd5;
    mad.lo.u32 %r7, %r8, %r6, %r7;
    mul.lo.u32 %r7, %r7, 4;
    mul.wide.u32 %rd5, %r7, 1;
    add.u64 %rd4, %rd2, %rd5;
    ld.global.f32 %f1, [%rd4];
    st.global.f32 [%rd3], %f1;
done:
    ret;
}
"#;

// ---------------------------------------------------------------------------
// EXL3 packed trellis dequant-dot (device-resident packed weights)
// ---------------------------------------------------------------------------

use crate::exl3::PackedLayout;

static PACKED_OK: AtomicBool = AtomicBool::new(true);

/// One EXL3 quantized linear, entirely on device from the packed trellis:
/// `y = svh * H128(W_hat^T @ xr)`. The input Hadamard is `input_hadamard`;
/// the dequant kernel splits each output row's k-loop over a warp's lanes
/// (so the GPU is not limited to one block per SM), and a small output
/// butterfly applies H128 and the sign vector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn packed_linear(
    trellis: &Tensor,
    xr: &Tensor,
    lut: &Tensor,
    lut_i8: &Tensor,
    w_scale: f32,
    svh: &Tensor,
    layout: PackedLayout,
) -> Result<Tensor, String> {
    if svh.elem_count() != layout.n {
        return Err(format!(
            "packed linear: svh has {} values, expected n={}",
            svh.elem_count(),
            layout.n
        ));
    }
    if !fused_enabled() || !PACKED_OK.load(Ordering::Relaxed) {
        return packed_linear_fallback(trellis, xr, lut, svh, layout);
    }
    let t = if dp4a_enabled() {
        let op = FusedPackedDp4a {
            lut_i8: lut_i8.clone(),
            w_scale,
            layout,
        };
        trellis
            .apply_op2_no_bwd(xr, &op)
            .map_err(|e| e.to_string())?
    } else {
        packed_dot_split(trellis, xr, lut, layout)?
    };
    output_hadamard(&t, svh)
}

/// `t = W_hat^T @ xr`, decoded from the trellis. One warp per output row; the
/// 32 lanes split the `tiles_k` loop, so even a small linear fills the GPU.
fn packed_dot_split(
    trellis: &Tensor,
    xr: &Tensor,
    lut: &Tensor,
    layout: PackedLayout,
) -> Result<Tensor, String> {
    if !fused_enabled() || !PACKED_OK.load(Ordering::Relaxed) {
        return packed_dot_split_fallback(trellis, xr, lut, layout);
    }
    let op = FusedPackedDotSplit {
        lut: lut.clone(),
        layout,
    };
    match trellis.apply_op2_no_bwd(xr, &op) {
        Ok(t) => Ok(t),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "exl3_packed_single", &e.to_string());
            PACKED_OK.store(false, Ordering::Relaxed);
            packed_dot_split_fallback(trellis, xr, lut, layout)
        }
    }
}

fn packed_dot_split_fallback(
    trellis: &Tensor,
    xr: &Tensor,
    lut: &Tensor,
    layout: PackedLayout,
) -> Result<Tensor, String> {
    use candle_core::Storage;
    let read = |t: &Tensor| -> Result<Vec<f32>, String> {
        let (storage, l) = t.storage_and_layout();
        match &*storage {
            Storage::Cpu(c) => {
                let all = c.as_slice::<f32>().map_err(|e| e.to_string())?;
                Ok(all[l.start_offset()..l.start_offset() + t.elem_count()].to_vec())
            }
            _ => Err("packed_dot fallback needs CPU tensors".into()),
        }
    };
    let (ts, tl) = trellis.storage_and_layout();
    let Storage::Cpu(tc) = &*ts else {
        return Err("packed_dot fallback needs CPU tensors".into());
    };
    let all_words = tc.as_slice::<u32>().map_err(|e| e.to_string())?;
    let nw = layout.words_per_tile / 2;
    let nwords = layout.tiles_k * layout.tiles_n * nw;
    let start = tl.start_offset();
    if start + nwords > all_words.len() {
        return Err("packed_dot: trellis storage too short".into());
    }
    let t = crate::exl3::packed_dot_host_impl(
        &all_words[start..start + nwords],
        layout,
        &read(lut)?,
        &read(xr)?,
    )?;
    Tensor::from_vec(t, (layout.n,), trellis.device()).map_err(|e| e.to_string())
}

// Opt-in INT8 dp4a variant. On-the-fly decode approximation; the trellis is
// never requantized or stored as INT8. Enabled with EXLLAMA_PACKED_DP4A=1.
fn dp4a_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("EXLLAMA_PACKED_DP4A").is_some())
}

struct FusedPackedDp4a {
    lut_i8: Tensor,
    w_scale: f32,
    layout: PackedLayout,
}

impl CustomOp2 for FusedPackedDp4a {
    fn name(&self) -> &'static str {
        "exlama_fused_exl3_packed_dp4a"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        // CPU fallback stays exact: the INT8 path is device-only.
        let words = s1.as_slice::<u32>()?;
        let xvals = s2.as_slice::<f32>()?;
        let (ls, _) = self.lut_i8.storage_and_layout();
        let candle_core::Storage::Cpu(_) = &*ls else {
            return Err(candle_core::Error::Msg("packed LUT is not on CPU".into()));
        };
        let t = crate::exl3::packed_dot_host_impl(
            &words[l1.start_offset()..],
            self.layout,
            &vec![0.0f32; 65536],
            &xvals[l2.start_offset()..l2.start_offset() + self.layout.k],
        )
        .map_err(candle_core::Error::Msg)?;
        Ok((CpuStorage::F32(t), (self.layout.n,).into()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        _l1: &Layout,
        s2: &candle_core::CudaStorage,
        _l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;
        let words = s1.as_cuda_slice::<u32>()?;
        let xr = s2.as_cuda_slice::<f32>()?;
        let (ls, _) = self.lut_i8.storage_and_layout();
        let Storage::Cuda(lc) = &*ls else {
            return Err(candle_core::Error::Msg("INT8 LUT is not on CUDA".into()));
        };
        let lut_i8 = lc.as_cuda_slice::<u32>()?;
        let func = s1.device.get_or_load_custom_func(
            "exl3_packed_dp4a",
            "exllama_exl3_packed",
            PACKED_DP4A_PTX,
        )?;
        let out = unsafe { s1.device.alloc::<f32>(self.layout.n)? };
        let nw = (self.layout.words_per_tile / 2) as u32;
        let ring = nw * 32;
        let (n_u, tk_u, tn_u) = (
            self.layout.n as u32,
            self.layout.tiles_k as u32,
            self.layout.tiles_n as u32,
        );
        let (bits_u, half_u) = (self.layout.k_bits_int as u32, self.layout.k_half as u32);
        // dp4a dequantizes 4 INT8 products with scale `w_scale / 32`.
        let out_scale = self.w_scale / 32.0;
        let mut args = func.builder();
        args.arg(words)
            .arg(xr)
            .arg(lut_i8)
            .arg(&out)
            .arg(&n_u)
            .arg(&tk_u)
            .arg(&tn_u)
            .arg(&nw)
            .arg(&bits_u)
            .arg(&half_u)
            .arg(&ring)
            .arg(lut_i8)
            .arg(&out_scale);
        let cfg = LaunchConfig {
            grid_dim: (self.layout.n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            (self.layout.n,).into(),
        ))
    }
}

#[cfg(feature = "gpu-cuda")]
const PACKED_DP4A_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry exl3_packed_dp4a(
    .param .u64 p_trellis,
    .param .u64 p_xr,
    .param .u64 p_lut,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_tiles_k,
    .param .u32 p_tiles_n,
    .param .u32 p_nw,
    .param .u32 p_bits,
    .param .u32 p_half,
    .param .u32 p_ring,
    .param .u64 p_lut_i8,
    .param .f32 p_out_scale
)
{
    .reg .u32 %j, %tn, %tk, %rr, %c, %i, %end, %s, %wi, %o;
    .reg .u32 %w0, %w1, %idx, %base, %wo, %wi1, %sh, %sh2, %xi, %nwn;
    .reg .u32 %n, %ntk, %ntn, %nnw, %nbits, %nhalf, %nring, %mask;
    .reg .u32 %ctaidv, %ntidv, %tidv;
    .reg .u32 %a0, %a1, %a2, %a3, %a4, %a5, %g, %q, %tmp;
    .reg .u64 %rd1, %rd2, %rd3, %rd4, %rd5, %rd6, %rd7, %rd8, %rd9;
    .reg .f32 %acc, %fv, %fx, %accF;
    .reg .s32 %xi8;
    .reg .u32 %accI, %packW, %packX, %wByte, %xByte, %shft, %grp;
    .reg .pred %p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7, %pG;

    ld.param.u64 %rd1, [p_trellis];
    ld.param.u64 %rd2, [p_xr];
    ld.param.u64 %rd3, [p_lut];
    ld.param.u64 %rd4, [p_out];
    ld.param.u32 %n, [p_n];
    ld.param.u32 %ntk, [p_tiles_k];
    ld.param.u32 %ntn, [p_tiles_n];
    ld.param.u32 %nnw, [p_nw];
    ld.param.u32 %nbits, [p_bits];
    ld.param.u32 %nhalf, [p_half];
    ld.param.u32 %nring, [p_ring];
    ld.param.u64 %rd7, [p_lut_i8];

    mov.u32 %ctaidv, %ctaid.x;
    mov.u32 %ntidv, %ntid.x;
    mov.u32 %tidv, %tid.x;
    mad.lo.u32 %j, %ctaidv, %ntidv, %tidv;
    setp.ge.u32 %p0, %j, %n;
    @%p0 bra done;

    and.b32 %c, %j, 15;
    shr.u32 %tn, %j, 4;
    mov.u32 %accI, 0;
    mov.u32 %packW, 0;
    mov.u32 %packX, 0;
    mov.u32 %grp, 0;
    mov.u32 %tk, 0;

tk_loop:
    setp.ge.u32 %p1, %tk, %ntk;
    @%p1 bra tk_end;

    mad.lo.u32 %tmp, %tk, %ntn, %tn;
    mul.lo.u32 %base, %tmp, %nnw;
    mov.u32 %rr, 0;

r_loop:
    setp.ge.u32 %p2, %rr, 16;
    @%p2 bra r_end;

    // inverse of tile_perm: (r,c) -> ring position i
    and.b32 %a0, %c, 7;
    shl.b32 %a0, %a0, 2;
    and.b32 %a1, %rr, 7;
    shr.u32 %a1, %a1, 1;
    or.b32 %a0, %a0, %a1;
    shl.b32 %i, %a0, 3;
    and.b32 %a2, %rr, 1;
    add.u32 %i, %i, %a2;
    and.b32 %a3, %rr, 8;
    shr.u32 %a3, %a3, 2;
    add.u32 %i, %i, %a3;
    and.b32 %a4, %c, 8;
    shr.u32 %a4, %a4, 1;
    add.u32 %i, %i, %a4;

    // bit index of the end of this 16-bit window (may be negative)
    setp.eq.u32 %p3, %nhalf, 0;
    @%p3 bra int_end;
    shr.u32 %g, %i, 4;
    and.b32 %q, %i, 15;
    mul.lo.u32 %nwn, %nnw, 2; // words_per_tile = nw * 2
    mad.lo.u32 %end, %g, %nwn, 0;
    mad.lo.u32 %end, %q, %nbits, %end;
    add.u32 %a5, %q, 1;
    shr.u32 %a5, %a5, 1;
    add.u32 %end, %end, %a5;
    bra calc_window;

int_end:
    add.u32 %a5, %i, 1;
    mul.lo.u32 %end, %a5, %nbits;

calc_window:
    add.u32 %s, %end, 0xfffffff0;
    sub.u32 %mask, %nring, 1;
    and.b32 %tmp, %nring, %mask;
    setp.eq.u32 %p6, %tmp, 0;
    @%p6 bra wrap_pow2;
    rem.s32 %s, %s, %nring;
    setp.lt.s32 %p4, %s, 0;
    @%p4 add.u32 %s, %s, %nring;
    bra wrap_done;
wrap_pow2:
    and.b32 %s, %s, %mask;
wrap_done:
    shr.u32 %wi, %s, 5;
    and.b32 %o, %s, 31;

    add.u32 %wi1, %wi, 1;
    setp.eq.u32 %p5, %wi1, %nnw;
    @%p5 mov.u32 %wi1, 0;

    add.u32 %wo, %base, %wi;
    mul.wide.u32 %rd5, %wo, 4;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.u32 %w0, [%rd5];

    add.u32 %wo, %base, %wi1;
    mul.wide.u32 %rd6, %wo, 4;
    add.u64 %rd6, %rd1, %rd6;
    ld.global.u32 %w1, [%rd6];

    sub.u32 %sh, 48, %o;
    setp.gt.u32 %p7, %sh, 31;
    @%p7 bra wide_window;
    shf.r.wrap.b32 %idx, %w1, %w0, %sh;
    and.b32 %idx, %idx, 0x0000ffff;
    bra window_done;
wide_window:
    sub.u32 %sh2, %sh, 32;
    shr.u32 %idx, %w0, %sh2;
    and.b32 %idx, %idx, 0x0000ffff;
window_done:

    mul.wide.u32 %rd5, %idx, 4;
    add.u64 %rd5, %rd7, %rd5;
    ld.global.u32 %wByte, [%rd5];
    and.b32 %wByte, %wByte, 0xff;

    mad.lo.u32 %xi, %tk, 16, %rr;
    mul.wide.u32 %rd6, %xi, 4;
    add.u64 %rd6, %rd2, %rd6;
    ld.global.f32 %fx, [%rd6];
    mul.f32 %fx, %fx, 32.0;
    cvt.rni.s32.f32 %xi8, %fx;
    max.s32 %xi8, %xi8, -127;
    min.s32 %xi8, %xi8, 127;
    and.b32 %xByte, %xi8, 0xff;

    shl.b32 %shft, %grp, 3;
    shl.b32 %wByte, %wByte, %shft;
    shl.b32 %xByte, %xByte, %shft;
    or.b32 %packW, %packW, %wByte;
    or.b32 %packX, %packX, %xByte;
    add.u32 %grp, %grp, 1;
    setp.eq.u32 %pG, %grp, 4;
    @%pG dp4a.s32.s32 %accI, %packW, %packX, %accI;
    @%pG mov.u32 %packW, 0;
    @%pG mov.u32 %packX, 0;
    @%pG mov.u32 %grp, 0;
    add.u32 %rr, %rr, 1;
    bra r_loop;

r_end:
    add.u32 %tk, %tk, 1;
    bra tk_loop;

tk_end:
    cvt.rn.f32.s32 %accF, %accI;
    ld.param.f32 %fv, [p_out_scale];
    mul.f32 %accF, %accF, %fv;
    mul.wide.u32 %rd5, %j, 4;
    add.u64 %rd5, %rd4, %rd5;
    st.global.f32 [%rd5], %accF;

done:
    ret;
}
"#;

/// `y = svh * H128(t)`, one launch per layer.
pub(crate) fn output_hadamard(t: &Tensor, svh: &Tensor) -> Result<Tensor, String> {
    if t.elem_count() != svh.elem_count() || !t.elem_count().is_multiple_of(128) {
        return Err(format!(
            "output hadamard: t={} svh={}, want equal non-zero multiples of 128",
            t.elem_count(),
            svh.elem_count()
        ));
    }
    static OUTPUT_H_OK: AtomicBool = AtomicBool::new(true);
    if !fused_enabled() || !OUTPUT_H_OK.load(Ordering::Relaxed) {
        return output_hadamard_fallback(t, svh);
    }
    match t.apply_op2_no_bwd(svh, &FusedOutputH) {
        Ok(y) => Ok(y),
        Err(e) => {
            static REPORTED: OnceLock<()> = OnceLock::new();
            report_once(&REPORTED, "output_h128", &e.to_string());
            OUTPUT_H_OK.store(false, Ordering::Relaxed);
            output_hadamard_fallback(t, svh)
        }
    }
}

fn output_hadamard_fallback(t: &Tensor, svh: &Tensor) -> Result<Tensor, String> {
    use candle_core::Storage;
    let read = |x: &Tensor| -> Result<Vec<f32>, String> {
        let (storage, l) = x.storage_and_layout();
        match &*storage {
            Storage::Cpu(c) => {
                let all = c.as_slice::<f32>().map_err(|e| e.to_string())?;
                Ok(all[l.start_offset()..l.start_offset() + x.elem_count()].to_vec())
            }
            _ => Err("output_h128 fallback needs CPU tensors".into()),
        }
    };
    let y = output_hadamard_scale_f32(&read(t)?, &read(svh)?)?;
    Tensor::from_vec(y, (t.elem_count(),), t.device()).map_err(|e| e.to_string())
}

fn packed_linear_fallback(
    trellis: &Tensor,
    xr: &Tensor,
    lut: &Tensor,
    svh: &Tensor,
    layout: PackedLayout,
) -> Result<Tensor, String> {
    use candle_core::Storage;
    let read_f32 = |t: &Tensor| -> Result<Vec<f32>, String> {
        let (storage, l) = t.storage_and_layout();
        match &*storage {
            Storage::Cpu(c) => {
                let all = c.as_slice::<f32>().map_err(|e| e.to_string())?;
                Ok(all[l.start_offset()..l.start_offset() + t.elem_count()].to_vec())
            }
            _ => Err("packed_linear fallback needs CPU tensors".into()),
        }
    };
    let (ts, tl) = trellis.storage_and_layout();
    let Storage::Cpu(tc) = &*ts else {
        return Err("packed_linear fallback needs CPU tensors".into());
    };
    let all_words = tc.as_slice::<u32>().map_err(|e| e.to_string())?;
    let nw = layout.words_per_tile / 2;
    let nwords = layout.tiles_k * layout.tiles_n * nw;
    let start = tl.start_offset();
    if start + nwords > all_words.len() {
        return Err("packed_linear: trellis storage too short".into());
    }
    let t = crate::exl3::packed_dot_host_impl(
        &all_words[start..start + nwords],
        layout,
        &read_f32(lut)?,
        &read_f32(xr)?,
    )?;
    let y = output_hadamard_scale_f32(&t, &read_f32(svh)?)?;
    Tensor::from_vec(y, (layout.n,), trellis.device()).map_err(|e| e.to_string())
}

struct FusedPackedDotSplit {
    lut: Tensor,
    layout: PackedLayout,
}

impl CustomOp2 for FusedPackedDotSplit {
    fn name(&self) -> &'static str {
        "exlama_fused_exl3_packed_single"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let words = s1.as_slice::<u32>()?;
        let xvals = s2.as_slice::<f32>()?;
        let (ls, _) = self.lut.storage_and_layout();
        let candle_core::Storage::Cpu(lc) = &*ls else {
            return Err(candle_core::Error::Msg("packed LUT is not on CPU".into()));
        };
        let t = crate::exl3::packed_dot_host_impl(
            &words[l1.start_offset()..],
            self.layout,
            lc.as_slice::<f32>()?,
            &xvals[l2.start_offset()..l2.start_offset() + self.layout.k],
        )
        .map_err(candle_core::Error::Msg)?;
        Ok((CpuStorage::F32(t), (self.layout.n,).into()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::Storage;

        let words = s1.as_cuda_slice::<u32>()?;
        let xr = s2.as_cuda_slice::<f32>()?;
        let (ls, _) = self.lut.storage_and_layout();
        let Storage::Cuda(lc) = &*ls else {
            return Err(candle_core::Error::Msg("packed LUT is not on CUDA".into()));
        };
        let lut = lc.as_cuda_slice::<f32>()?;
        let func = s1.device.get_or_load_custom_func(
            "exl3_packed_single",
            "exllama_exl3_packed",
            PACKED_SINGLE_PTX,
        )?;
        let out = unsafe { s1.device.alloc::<f32>(self.layout.n)? };
        let nw = (self.layout.words_per_tile / 2) as u32;
        let ring = nw * 32;
        let (n_u, tk_u, tn_u) = (
            self.layout.n as u32,
            self.layout.tiles_k as u32,
            self.layout.tiles_n as u32,
        );
        let (bits_u, half_u) = (self.layout.k_bits_int as u32, self.layout.k_half as u32);
        let (st, sx) = (l1.start_offset() as u32, l2.start_offset() as u32);
        let mut args = func.builder();
        args.arg(words)
            .arg(xr)
            .arg(lut)
            .arg(&out)
            .arg(&n_u)
            .arg(&tk_u)
            .arg(&tn_u)
            .arg(&nw)
            .arg(&bits_u)
            .arg(&half_u)
            .arg(&ring)
            .arg(&st)
            .arg(&sx);
        let cfg = LaunchConfig {
            grid_dim: (self.layout.n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            (self.layout.n,).into(),
        ))
    }
}

/// Output-side H128 + scale vector, one 128-thread block per output block.
struct FusedOutputH;

impl CustomOp2 for FusedOutputH {
    fn name(&self) -> &'static str {
        "exlama_fused_output_h128"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let t = s1.as_slice::<f32>()?;
        let sv = s2.as_slice::<f32>()?;
        let y = output_hadamard_scale_f32(
            &t[l1.start_offset()..l1.start_offset() + l1.shape().elem_count()],
            &sv[l2.start_offset()..l2.start_offset() + l2.shape().elem_count()],
        )
        .map_err(candle_core::Error::Msg)?;
        Ok((CpuStorage::F32(y), l1.shape().clone()))
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        let t = s1.as_cuda_slice::<f32>()?;
        let sv = s2.as_cuda_slice::<f32>()?;
        let len = l1.shape().elem_count();
        let out = unsafe { s1.device.alloc::<f32>(len)? };
        // Same butterfly kernel as the input side; `after = 1` applies the
        // sign vector after H128 instead of before it.
        let func = s1.device.get_or_load_custom_func(
            "exl3_input_h128",
            "exllama_exl3_h128",
            INPUT_H_PTX,
        )?;
        let blocks = (len / 128) as u32;
        let scale = 1.0f32 / (128.0f32).sqrt();
        let (st, ss) = (l1.start_offset() as u32, l2.start_offset() as u32);
        let after = 1u32;
        let mut args = func.builder();
        args.arg(t)
            .arg(sv)
            .arg(&out)
            .arg(&blocks)
            .arg(&scale)
            .arg(&st)
            .arg(&ss)
            .arg(&after);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * 4,
        };
        unsafe {
            args.launch(cfg)
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, s1.device.clone()),
            l1.shape().clone(),
        ))
    }
}

#[cfg(feature = "gpu-cuda")]
const PACKED_SINGLE_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry exl3_packed_single(
    .param .u64 p_trellis,
    .param .u64 p_xr,
    .param .u64 p_lut,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_tiles_k,
    .param .u32 p_tiles_n,
    .param .u32 p_nw,
    .param .u32 p_bits,
    .param .u32 p_half,
    .param .u32 p_ring
)
{
    .reg .u32 %j, %tn, %tk, %rr, %c, %i, %end, %s, %wi, %o;
    .reg .u32 %w0, %w1, %idx, %base, %wo, %wi1, %sh, %sh2, %xi, %nwn;
    .reg .u32 %n, %ntk, %ntn, %nnw, %nbits, %nhalf, %nring, %mask;
    .reg .u32 %ctaidv, %ntidv, %tidv;
    .reg .u32 %a0, %a1, %a2, %a3, %a4, %a5, %g, %q, %tmp;
    .reg .u64 %rd1, %rd2, %rd3, %rd4, %rd5, %rd6, %rd7, %rd8, %rd9;
    .reg .f32 %acc, %fv, %fx;
    .reg .pred %p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7;

    ld.param.u64 %rd1, [p_trellis];
    ld.param.u64 %rd2, [p_xr];
    ld.param.u64 %rd3, [p_lut];
    ld.param.u64 %rd4, [p_out];
    ld.param.u32 %n, [p_n];
    ld.param.u32 %ntk, [p_tiles_k];
    ld.param.u32 %ntn, [p_tiles_n];
    ld.param.u32 %nnw, [p_nw];
    ld.param.u32 %nbits, [p_bits];
    ld.param.u32 %nhalf, [p_half];
    ld.param.u32 %nring, [p_ring];

    mov.u32 %ctaidv, %ctaid.x;
    mov.u32 %ntidv, %ntid.x;
    mov.u32 %tidv, %tid.x;
    mad.lo.u32 %j, %ctaidv, %ntidv, %tidv;
    setp.ge.u32 %p0, %j, %n;
    @%p0 bra done;

    and.b32 %c, %j, 15;
    shr.u32 %tn, %j, 4;
    mov.f32 %acc, 0f00000000;
    mov.u32 %tk, 0;

tk_loop:
    setp.ge.u32 %p1, %tk, %ntk;
    @%p1 bra tk_end;

    mad.lo.u32 %tmp, %tk, %ntn, %tn;
    mul.lo.u32 %base, %tmp, %nnw;
    mov.u32 %rr, 0;

r_loop:
    setp.ge.u32 %p2, %rr, 16;
    @%p2 bra r_end;

    // inverse of tile_perm: (r,c) -> ring position i
    and.b32 %a0, %c, 7;
    shl.b32 %a0, %a0, 2;
    and.b32 %a1, %rr, 7;
    shr.u32 %a1, %a1, 1;
    or.b32 %a0, %a0, %a1;
    shl.b32 %i, %a0, 3;
    and.b32 %a2, %rr, 1;
    add.u32 %i, %i, %a2;
    and.b32 %a3, %rr, 8;
    shr.u32 %a3, %a3, 2;
    add.u32 %i, %i, %a3;
    and.b32 %a4, %c, 8;
    shr.u32 %a4, %a4, 1;
    add.u32 %i, %i, %a4;

    // bit index of the end of this 16-bit window (may be negative)
    setp.eq.u32 %p3, %nhalf, 0;
    @%p3 bra int_end;
    shr.u32 %g, %i, 4;
    and.b32 %q, %i, 15;
    mul.lo.u32 %nwn, %nnw, 2; // words_per_tile = nw * 2
    mad.lo.u32 %end, %g, %nwn, 0;
    mad.lo.u32 %end, %q, %nbits, %end;
    add.u32 %a5, %q, 1;
    shr.u32 %a5, %a5, 1;
    add.u32 %end, %end, %a5;
    bra calc_window;

int_end:
    add.u32 %a5, %i, 1;
    mul.lo.u32 %end, %a5, %nbits;

calc_window:
    add.u32 %s, %end, 0xfffffff0;
    sub.u32 %mask, %nring, 1;
    and.b32 %tmp, %nring, %mask;
    setp.eq.u32 %p6, %tmp, 0;
    @%p6 bra wrap_pow2;
    rem.s32 %s, %s, %nring;
    setp.lt.s32 %p4, %s, 0;
    @%p4 add.u32 %s, %s, %nring;
    bra wrap_done;
wrap_pow2:
    and.b32 %s, %s, %mask;
wrap_done:
    shr.u32 %wi, %s, 5;
    and.b32 %o, %s, 31;

    add.u32 %wi1, %wi, 1;
    setp.eq.u32 %p5, %wi1, %nnw;
    @%p5 mov.u32 %wi1, 0;

    add.u32 %wo, %base, %wi;
    mul.wide.u32 %rd5, %wo, 4;
    add.u64 %rd5, %rd1, %rd5;
    ld.global.u32 %w0, [%rd5];

    add.u32 %wo, %base, %wi1;
    mul.wide.u32 %rd6, %wo, 4;
    add.u64 %rd6, %rd1, %rd6;
    ld.global.u32 %w1, [%rd6];

    sub.u32 %sh, 48, %o;
    setp.gt.u32 %p7, %sh, 31;
    @%p7 bra wide_window;
    shf.r.wrap.b32 %idx, %w1, %w0, %sh;
    and.b32 %idx, %idx, 0x0000ffff;
    bra window_done;
wide_window:
    sub.u32 %sh2, %sh, 32;
    shr.u32 %idx, %w0, %sh2;
    and.b32 %idx, %idx, 0x0000ffff;
window_done:

    mul.wide.u32 %rd5, %idx, 4;
    add.u64 %rd5, %rd3, %rd5;
    ld.global.f32 %fv, [%rd5];

    mad.lo.u32 %xi, %tk, 16, %rr;
    mul.wide.u32 %rd6, %xi, 4;
    add.u64 %rd6, %rd2, %rd6;
    ld.global.f32 %fx, [%rd6];

    fma.rn.f32 %acc, %fv, %fx, %acc;
    add.u32 %rr, %rr, 1;
    bra r_loop;

r_end:
    add.u32 %tk, %tk, 1;
    bra tk_loop;

tk_end:
    mul.wide.u32 %rd5, %j, 4;
    add.u64 %rd5, %rd4, %rd5;
    st.global.f32 [%rd5], %acc;

done:
    ret;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn butterfly_h128_matches_exl3_reference() {
        let vals: Vec<f32> = (0..256).map(|i| (i as f32 % 13.0) * 0.07 - 0.4).collect();
        let mut got = vals.clone();
        let mut want = vals.clone();
        let scale = 1.0 / (128.0f32).sqrt();
        for blk in 0..2 {
            butterfly_h128(&mut got[blk * 128..(blk + 1) * 128]);
            crate::exl3::hadamard128(&mut want, blk * 128, 1, scale);
        }
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        let suh: Vec<f32> = (0..256).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
        let svh: Vec<f32> = (0..256).map(|i| 0.5 - (i % 3) as f32 * 0.05).collect();
        let xr = input_hadamard_f32(&vals, &suh).unwrap();
        let mut ref_xr = vals
            .iter()
            .zip(&suh)
            .map(|(&a, &b)| a * b)
            .collect::<Vec<_>>();
        for blk in 0..2 {
            crate::exl3::hadamard128(&mut ref_xr, blk * 128, 1, scale);
        }
        for (a, b) in xr.iter().zip(&ref_xr) {
            assert!((a - b).abs() < 1e-6, "input: {a} vs {b}");
        }
        let y = output_hadamard_scale_f32(&vals, &svh).unwrap();
        let mut ref_y = vals.clone();
        for blk in 0..2 {
            crate::exl3::hadamard128(&mut ref_y, blk * 128, 1, scale);
        }
        for (v, &s) in ref_y.iter_mut().zip(&svh) {
            *v *= s;
        }
        for (a, b) in y.iter().zip(&ref_y) {
            assert!((a - b).abs() < 1e-6, "output: {a} vs {b}");
        }
    }

    #[test]
    fn fused_rms_norm_matches_fallback_1d_and_2d() {
        let dev = Device::Cpu;
        let eps = 1e-6;
        // 1-D (hidden,) — the post/final norm case.
        let x = Tensor::from_slice(
            &[
                0.1f32, -0.7, 2.3, 0.0, 1.1, -0.4, 0.8, 1.7, -1.2, 0.3, 0.6, -2.0, 0.9, -0.1, 1.4,
                0.2,
            ],
            (16,),
            &dev,
        )
        .unwrap();
        let w1 = Tensor::from_slice(
            &[
                1.0f32, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.0, 1.1, 0.95, 1.05, 0.85, 1.15, 0.9, 1.25,
                1.0,
            ],
            (16,),
            &dev,
        )
        .unwrap();
        let got = rms_norm(&x, &w1, eps).unwrap().to_vec1::<f32>().unwrap();
        let want = rms_norm_fallback(&x, &w1, eps)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "1-D: {} vs {}", a, b);
        }

        // 2-D (heads, head_dim) — q/k norms normalize each head independently.
        let vals: Vec<f32> = (0..24).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let x2 = Tensor::from_slice(&vals, (4, 6), &dev).unwrap();
        let w2 = Tensor::from_slice(&[1.0f32, 1.1, 0.9, 1.2, 0.8, 1.05], (6,), &dev).unwrap();
        let got = rms_norm(&x2, &w2, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = rms_norm_fallback(&x2, &w2, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "2-D: {} vs {}", a, b);
        }
    }

    #[test]
    fn fused_rope_matches_fallback_with_unrotated_tail() {
        let dev = Device::Cpu;
        let (heads, head_len, half) = (3usize, 16usize, 2usize);
        let vals: Vec<f32> = (0..heads * head_len)
            .map(|i| (i as f32 % 11.0) / 11.0 - 0.5)
            .collect();
        let x = Tensor::from_slice(&vals, (heads * head_len,), &dev).unwrap();
        let c: Vec<f32> = (0..half).map(|i| 0.6 + i as f32 * 0.1).collect();
        let s: Vec<f32> = (0..half).map(|i| 0.8 - i as f32 * 0.1).collect();
        let cos = Tensor::from_slice(&c, (1, half), &dev).unwrap();
        let sin = Tensor::from_slice(&s, (1, half), &dev).unwrap();
        let got = rope(&x, &cos, &sin, heads, head_len, half)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = rope_fallback(&x, &cos, &sin, heads, head_len, half)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "{} vs {}", a, b);
        }
        // The tail (dims 2*half..head_len) must pass through byte-for-byte.
        for h in 0..heads {
            for d in half * 2..head_len {
                assert_eq!(got[h * head_len + d], vals[h * head_len + d]);
            }
        }
    }
}

/// Real-GPU numeric check for the fused EXL3 packed-linear kernel.
#[cfg(all(test, feature = "gpu-cuda"))]
#[test]
fn cuda_packed_linear_matches_cpu_reference() {
    use crate::exl3::{codebook_lut_f32, packed_dot_host_impl, Codebook, PackedLayout};
    if !candle_core::utils::cuda_is_available() {
        eprintln!("[fused] CUDA unavailable, skipping packed-linear check");
        return;
    }
    let dev = candle_core::Device::new_cuda(0).unwrap();
    let layout = PackedLayout {
        k: 128,
        n: 128,
        tiles_k: 8,
        tiles_n: 8,
        words_per_tile: 128,
        k_bits_int: 8,
        k_half: false,
    };
    let nw = layout.words_per_tile / 2;
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let words: Vec<u32> = (0..layout.tiles_k * layout.tiles_n * nw)
        .map(|_| next() as u32)
        .collect();
    let xr: Vec<f32> = (0..layout.k)
        .map(|i| ((i % 11) as f32) * 0.03 - 0.15)
        .collect();
    let lut = codebook_lut_f32(Codebook::Mcg);
    let svh: Vec<f32> = (0..layout.n).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
    let t = packed_dot_host_impl(&words, layout, &lut, &xr).unwrap();
    let expect = output_hadamard_scale_f32(&t, &svh).unwrap();

    let wt = candle_core::Tensor::from_slice(&words, (layout.tiles_k, layout.tiles_n, nw), &dev)
        .unwrap();
    let xt = candle_core::Tensor::from_slice(&xr, (layout.k,), &dev).unwrap();
    let lt = candle_core::Tensor::from_slice(&lut, (65536,), &dev).unwrap();
    let (lut8, ws) = crate::exl3::codebook_lut_i8(Codebook::Mcg);
    let lt_i8 = candle_core::Tensor::from_slice(&lut8, (65536,), &dev).unwrap();
    let st = candle_core::Tensor::from_slice(&svh, (layout.n,), &dev).unwrap();
    let got = packed_linear(&wt, &xt, &lt, &lt_i8, ws, &st, layout)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
        // The opt-in INT8 path is an approximation; keep a looser but still
        // meaningful bound there, and the exact bound for the default path.
        let tol = if dp4a_enabled() {
            0.12 + 0.12 * e.abs()
        } else {
            5e-3 + 5e-3 * e.abs()
        };
        assert!(
            (g - e).abs() < tol,
            "packed linear row {i}: got {g}, expected {e}"
        );
    }
    eprintln!("[fused] CUDA packed-linear check passed");
}

/// Real-GPU numeric check: the PTX kernels must match the candle-op fallback.
/// Only runs with `--features gpu-cuda` on a machine with CUDA available.
#[cfg(all(test, feature = "gpu-cuda"))]
#[test]
fn cuda_fused_kernels_match_fallback() {
    if !candle_core::utils::cuda_is_available() {
        eprintln!("[fused] CUDA unavailable, skipping real-GPU numeric check");
        return;
    }
    let dev = candle_core::Device::new_cuda(0).unwrap();

    // RMSNorm, 1-D and 2-D (per-head).
    for (shape, vals) in [
        (
            vec![512usize],
            (0..512).map(|i| i as f32 / 100.0 - 2.0).collect::<Vec<_>>(),
        ),
        (
            vec![8usize, 256usize],
            (0..2048)
                .map(|i| i as f32 / 500.0 - 2.0)
                .collect::<Vec<_>>(),
        ),
    ] {
        let x = Tensor::from_slice(&vals, shape.as_slice(), &dev).unwrap();
        let cols = *shape.last().unwrap();
        let w1v: Vec<f32> = (0..cols).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
        let w1 = Tensor::from_slice(&w1v, (cols,), &dev).unwrap();
        let got = x
            .apply_op2_no_bwd(&w1, &FusedRmsNorm { eps: 1e-6 })
            .expect("RMSNorm CUDA kernel must load")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = rms_norm_fallback(&x, &w1, 1e-6)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (a, b) in got.iter().zip(&want) {
            assert!(
                (a - b).abs() < 2e-3 + 2e-3 * scale,
                "RMSNorm mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    // Partial RoPE with an unrotated tail.
    let (heads, head_len, half) = (8usize, 64usize, 8usize);
    let vals: Vec<f32> = (0..heads * head_len)
        .map(|i| i as f32 / 100.0 - 3.0)
        .collect();
    let x = Tensor::from_slice(&vals, (heads * head_len,), &dev).unwrap();
    let c: Vec<f32> = (0..half).map(|i| 0.9 - i as f32 / 20.0).collect();
    let s: Vec<f32> = (0..half).map(|i| 0.1 + i as f32 / 20.0).collect();
    let cos = Tensor::from_slice(&c, (1, half), &dev).unwrap();
    let sin = Tensor::from_slice(&s, (1, half), &dev).unwrap();
    let got = x
        .apply_op3_no_bwd(
            &cos,
            &sin,
            &FusedRope {
                heads,
                head_len,
                half,
            },
        )
        .expect("RoPE CUDA kernel must load")
        .to_vec1::<f32>()
        .unwrap();
    let want = rope_fallback(&x, &cos, &sin, heads, head_len, half)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut bad = 0;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        if (a - b).abs() >= 2e-3 + 2e-3 * scale {
            if bad < 12 {
                eprintln!(
                    "RoPE mismatch idx={} head={} d={} got={} want={}",
                    i,
                    i / head_len,
                    i % head_len,
                    a,
                    b
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{} RoPE values diverge from the fallback", bad);
    for h in 0..heads {
        for d in half * 2..head_len {
            assert_eq!(got[h * head_len + d], vals[h * head_len + d]);
        }
    }
    eprintln!("[fused] real-GPU numeric check passed");
}

#[test]
fn fused_phase2b_matches_fallback_on_cpu() {
    let dev = candle_core::Device::Cpu;

    // Gated norm: per-row RMS scale * raw w * silu(z).
    let (rows, cols) = (4usize, 6usize);
    let ov: Vec<f32> = (0..rows * cols).map(|i| i as f32 / 7.0 - 1.5).collect();
    let wv: Vec<f32> = (0..cols).map(|i| 0.8 + i as f32 / 10.0).collect();
    let zv: Vec<f32> = (0..rows * cols).map(|i| i as f32 / 9.0 - 2.0).collect();
    let o = Tensor::from_slice(&ov, (rows, cols), &dev).unwrap();
    let w = Tensor::from_slice(&wv, (1, cols), &dev).unwrap();
    let z = Tensor::from_slice(&zv, (rows, cols), &dev).unwrap();
    let got = gated_norm(&o, &w, &z, 1e-6)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = gated_norm_fallback(&o, &w, &z, 1e-6)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-6, "gated norm: {} vs {}", a, b);
    }

    // DeltaNet gates: softplus, decay and write gate.
    let abv: Vec<f32> = (0..5).map(|i| i as f32 / 2.0 - 2.0).collect();
    let bv: Vec<f32> = (0..5).map(|i| i as f32 / 3.0 - 1.0).collect();
    let av: Vec<f32> = (0..5).map(|i| -1.0 - i as f32 / 10.0).collect();
    let ab = Tensor::from_slice(&abv, (5,), &dev).unwrap();
    let b = Tensor::from_slice(&bv, (5,), &dev).unwrap();
    let alog = Tensor::from_slice(&av, (5,), &dev).unwrap();
    let got = deltanet_gates(&ab, &b, &alog)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = deltanet_gates_fallback(&ab, &b, &alog)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, bb) in got.iter().zip(&want) {
        assert!((a - bb).abs() < 1e-6, "gates: {} vs {}", a, bb);
    }

    // Softmax over dim 1 of (nkv, tlen, groups), with scaling.
    let (nkv, tlen, groups) = (2usize, 7usize, 3usize);
    let sv: Vec<f32> = (0..nkv * tlen * groups)
        .map(|i| i as f32 / 5.0 - 4.0)
        .collect();
    let scores = Tensor::from_slice(&sv, (nkv, tlen, groups), &dev).unwrap();
    let got = softmax(&scores, 0.5)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = softmax_fallback(&scores, 0.5)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, bb) in got.iter().zip(&want) {
        assert!((a - bb).abs() < 1e-6, "softmax: {} vs {}", a, bb);
    }
    for kh in 0..nkv {
        for g in 0..groups {
            let sum: f32 = (0..tlen)
                .map(|t| got[kh * tlen * groups + t * groups + g])
                .sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax row sums to {}", sum);
        }
    }

    // Sigmoid gate on contiguous tensors.
    let gv: Vec<f32> = (0..24).map(|i| i as f32 / 4.0 - 3.0).collect();
    let a = Tensor::from_slice(&gv, (2, 3, 4), &dev).unwrap();
    let g = Tensor::from_slice(&gv, (2, 3, 4), &dev).unwrap();
    let got = sigmoid_gate(&a, &g)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = sigmoid_gate_fallback(&a, &g)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (x, y) in got.iter().zip(&want) {
        assert!((x - y).abs() < 1e-6, "sigmoid gate: {} vs {}", x, y);
    }
}

#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_phase2b_kernels_match_fallback() {
    if !candle_core::utils::cuda_is_available() {
        eprintln!("[fused] CUDA unavailable, skipping phase 2b GPU check");
        return;
    }
    let dev = candle_core::Device::new_cuda(0).unwrap();

    // Gated norm.
    let (rows, cols) = (16usize, 128usize);
    let ov: Vec<f32> = (0..rows * cols).map(|i| i as f32 / 50.0 - 1.0).collect();
    let wv: Vec<f32> = (0..cols).map(|i| 0.9 + i as f32 / 500.0).collect();
    let zv: Vec<f32> = (0..rows * cols).map(|i| i as f32 / 200.0 - 2.0).collect();
    let o = Tensor::from_slice(&ov, (rows, cols), &dev).unwrap();
    let w = Tensor::from_slice(&wv, (1, cols), &dev).unwrap();
    let z = Tensor::from_slice(&zv, (rows, cols), &dev).unwrap();
    let got = o
        .apply_op3_no_bwd(&w, &z, &FusedGatedNorm { eps: 1e-6 })
        .expect("gated_norm CUDA kernel must load")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = gated_norm_fallback(&o, &w, &z, 1e-6)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    for (a, b) in got.iter().zip(&want) {
        assert!(
            (a - b).abs() < 2e-3 + 2e-3 * scale,
            "gated norm: {} vs {}",
            a,
            b
        );
    }

    // DeltaNet gates.
    let n = 32usize;
    let abv: Vec<f32> = (0..n).map(|i| i as f32 / 4.0 - 4.0).collect();
    let bv: Vec<f32> = (0..n).map(|i| i as f32 / 5.0 - 3.0).collect();
    let av: Vec<f32> = (0..n).map(|i| -1.0 - i as f32 / 20.0).collect();
    let ab = Tensor::from_slice(&abv, (n,), &dev).unwrap();
    let b = Tensor::from_slice(&bv, (n,), &dev).unwrap();
    let alog = Tensor::from_slice(&av, (n,), &dev).unwrap();
    let got = ab
        .apply_op3_no_bwd(&b, &alog, &FusedDeltaNetGates)
        .expect("gates CUDA kernel must load")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = deltanet_gates_fallback(&ab, &b, &alog)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, bb) in got.iter().zip(&want) {
        assert!(
            (a - bb).abs() < 2e-4 + 2e-4 * bb.abs(),
            "gates: {} vs {}",
            a,
            bb
        );
    }

    // Softmax.
    let (nkv, tlen, groups) = (2usize, 256usize, 4usize);
    let sv: Vec<f32> = (0..nkv * tlen * groups)
        .map(|i| i as f32 / 100.0 - 5.0)
        .collect();
    let scores = Tensor::from_slice(&sv, (nkv, tlen, groups), &dev).unwrap();
    let got = scores
        .apply_op1_no_bwd(&FusedSoftmax { scale: 0.5 })
        .expect("softmax CUDA kernel must load")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = softmax_fallback(&scores, 0.5)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, bb) in got.iter().zip(&want) {
        assert!((a - bb).abs() < 1e-5, "softmax: {} vs {}", a, bb);
    }

    // Sigmoid gate.
    let gv: Vec<f32> = (0..2048).map(|i| i as f32 / 100.0 - 10.0).collect();
    let a = Tensor::from_slice(&gv, (2, 4, 256), &dev).unwrap();
    let g = Tensor::from_slice(&gv, (2, 4, 256), &dev).unwrap();
    let got = a
        .apply_op2_no_bwd(&g, &FusedSigmoidGate)
        .expect("sigmoid gate CUDA kernel must load")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = sigmoid_gate_fallback(&a, &g)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (x, y) in got.iter().zip(&want) {
        assert!(
            (x - y).abs() < 2e-5 + 2e-5 * y.abs(),
            "sigmoid: {} vs {}",
            x,
            y
        );
    }
    eprintln!("[fused] phase 2b real-GPU numeric check passed");
}

#[test]
fn fused_deltanet_qk_norm_matches_fallback_on_cpu() {
    let dev = candle_core::Device::Cpu;
    let (nkh, nvh, kd) = (4usize, 8usize, 16usize);
    let qv: Vec<f32> = (0..nkh * kd).map(|i| i as f32 / 13.0 - 2.0).collect();
    let kv: Vec<f32> = (0..nkh * kd).map(|i| i as f32 / 17.0 + 1.0).collect();
    let q = Tensor::from_slice(&qv, (nkh * kd,), &dev).unwrap();
    let k = Tensor::from_slice(&kv, (nkh * kd,), &dev).unwrap();
    let got = deltanet_qk_norm(&q, &k, nkh, nvh, kd)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = deltanet_qk_norm_fallback(&q, &k, nkh, nvh, kd)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-6, "qk norm: {} vs {}", a, b);
    }
    // Repetition order: value heads 2 and 3 must share key head 1.
    for d in 0..kd {
        assert_eq!(got[2 * kd + d], got[3 * kd + d]);
    }
}

#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_deltanet_qk_norm_matches_fallback() {
    if !candle_core::utils::cuda_is_available() {
        eprintln!("[fused] CUDA unavailable, skipping qk norm GPU check");
        return;
    }
    let dev = candle_core::Device::new_cuda(0).unwrap();
    let (nkh, nvh, kd) = (8usize, 16usize, 128usize);
    let qv: Vec<f32> = (0..nkh * kd).map(|i| i as f32 / 100.0 - 4.0).collect();
    let kv: Vec<f32> = (0..nkh * kd).map(|i| i as f32 / 70.0 + 1.0).collect();
    let q = Tensor::from_slice(&qv, (nkh * kd,), &dev).unwrap();
    let k = Tensor::from_slice(&kv, (nkh * kd,), &dev).unwrap();
    let got = q
        .apply_op2_no_bwd(&k, &FusedDeltanetQkNorm { nkh, nvh, kd })
        .expect("qk norm CUDA kernel must load")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = deltanet_qk_norm_fallback(&q, &k, nkh, nvh, kd)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut bad = 0;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        if (a - b).abs() >= 1e-4 + 2e-4 * scale {
            if bad < 10 {
                eprintln!("qk norm mismatch idx={} got={} want={}", i, a, b);
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{} qk norm values diverge", bad);
    eprintln!("[fused] qk norm real-GPU numeric check passed");
}

#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_write_kv_row_matches_fallback() {
    if !candle_core::utils::cuda_is_available() {
        eprintln!("[fused] CUDA unavailable, skipping KV row GPU check");
        return;
    }
    let dev = candle_core::Device::new_cuda(0).unwrap();
    let (ctx, cols, pos) = (8usize, 512usize, 3usize);
    let cache = Tensor::zeros((ctx, cols), DType::F16, &dev).unwrap();
    let rowv: Vec<f16> = (0..cols)
        .map(|i| f16::from_f32(i as f32 / 100.0 - 2.0))
        .collect();
    let row = Tensor::from_slice(&rowv, (1, cols), &dev).unwrap();
    let got = cache;
    got.inplace_op2(&row, &WriteKvRow { pos })
        .expect("write_kv_row CUDA kernel must load");
    let want = Tensor::zeros((ctx, cols), DType::F16, &dev)
        .unwrap()
        .slice_scatter(&row, 0, pos)
        .unwrap();
    let a = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = want
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(a, b);
    eprintln!("[fused] KV row real-GPU check passed");
}

#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_conv_step_matches_fallback() {
    if !candle_core::utils::cuda_is_available() {
        eprintln!("[fused] CUDA unavailable, skipping conv GPU check");
        return;
    }
    let dev = candle_core::Device::new_cuda(0).unwrap();
    let (hist, cols, kernel, pos) = (3usize, 256usize, 4usize, 2usize);
    let cv: Vec<f32> = (0..hist * cols).map(|i| i as f32 / 50.0 - 2.0).collect();
    let cur: Vec<f32> = (0..cols).map(|i| i as f32 / 30.0 + 1.0).collect();
    let wv: Vec<f32> = (0..cols * kernel).map(|i| i as f32 / 100.0 - 1.0).collect();
    let conv = Tensor::from_slice(&cv, (hist, cols), &dev).unwrap();
    let cur_t = Tensor::from_slice(&cur, (cols,), &dev).unwrap();
    let w = Tensor::from_slice(&wv, (cols, kernel), &dev).unwrap();
    let got = conv
        .apply_op3_no_bwd(&cur_t, &w, &FusedConvStep { pos, kernel })
        .expect("conv_step CUDA kernel must load")
        .to_vec1::<f32>()
        .unwrap();
    let want = conv_step_fallback(&conv, &cur_t, &w, pos, kernel)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let mut bad = 0;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        if (a - b).abs() >= 2e-4 + 2e-4 * scale {
            if bad < 12 {
                eprintln!(
                    "conv mismatch c={} got={} want={} cur={} w0..3={:?}",
                    i,
                    a,
                    b,
                    cur[i],
                    &wv[i * kernel..i * kernel + kernel]
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{} conv values diverge", bad);
    // In-place row write.
    let c2 = conv;
    c2.inplace_op2(&cur_t, &WriteRowF32 { pos })
        .expect("write_row_f32 CUDA kernel must load");
    let row = c2
        .narrow(0, pos, 1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(row, cur);
    eprintln!("[fused] conv real-GPU check passed");
}
