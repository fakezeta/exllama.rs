//! Compute backend selection: CPU BLAS (matrixmultiply sgemm), GPU (candle),
//! or scalar fallback. The best available backend is selected at startup and
//! used transparently by `tensor::matmul_t`.

use std::sync::OnceLock;
#[cfg(feature = "gpu")]
use std::{collections::HashMap, sync::Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar, // hand-written dot loops
    Blas,   // matrixmultiply sgemm (SIMD; the kernel ndarray uses)
    Gpu,    // candle (CUDA / Metal) when compiled with a gpu feature
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Blas => "cpu-blas (matrixmultiply sgemm)",
            Backend::Gpu => "gpu (candle)",
        }
    }
}

/// Global compute state: chosen backend + optional GPU device.
struct Compute {
    backend: Backend,
    #[cfg(feature = "gpu")]
    device: Option<candle_core::Device>,
}

static COMPUTE: OnceLock<Compute> = OnceLock::new();

/// Device-resident weights. A decode matmul used to re-upload the entire
/// weight matrix on every token, which made the GPU backend slower than the
/// CPU paths. Weight buffers live for the process lifetime, so they can be
/// uploaded once and reused, keyed by buffer address and shape.
#[cfg(feature = "gpu")]
struct GpuCache {
    tensors: Mutex<HashMap<(usize, usize, usize, u32), candle_core::Tensor>>,
}

#[cfg(feature = "gpu")]
static GPU_CACHE: OnceLock<GpuCache> = OnceLock::new();

/// Detect the best backend. Preferences: "auto" (default), "scalar", "blas",
/// "gpu". Falls back to BLAS if GPU is requested but unavailable.
pub fn init(preference: &str, nthreads: usize) -> Backend {
    COMPUTE
        .get_or_init(|| {
            let gpu = detect_gpu();
            let gpu_present = gpu.is_some();

            let backend = match preference {
                "scalar" => Backend::Scalar,
                "blas" | "cpu" => Backend::Blas,
                "gpu" if gpu_present => Backend::Gpu,
                "gpu" => Backend::Blas,
                "auto" => {
                    if gpu_present {
                        // GPU is dramatically faster for large matmuls.
                        Backend::Gpu
                    } else {
                        // No GPU: benchmark both CPU paths once for this machine.
                        pick_fastest_cpu(nthreads)
                    }
                }
                _ => Backend::Blas,
            };
            eprintln!(
                "[compute] backend={}  (available: cpu-blas{} )",
                backend.name(),
                if gpu_present { ", gpu" } else { "" }
            );
            if backend == Backend::Gpu {
                eprintln!(
                    "[compute] GPU weights are cached on device (first matmul per matrix uploads once)"
                );
            }
            Compute {
                backend,
                #[cfg(feature = "gpu")]
                device: gpu,
            }
        })
        .backend
}

pub fn backend() -> Backend {
    COMPUTE.get().map(|c| c.backend).unwrap_or(Backend::Blas)
}

/// The selected device, for the device-resident forward path.
#[cfg(feature = "gpu")]
pub fn device() -> Option<candle_core::Device> {
    COMPUTE.get().and_then(|c| c.device.clone())
}

/// Score the two CPU backends from the probe timings. Decode runs once per
/// generated token while prefill runs once per request, so the GEMV timing is
/// weighted 4x: a 4x GEMV win (observed on 12th-gen cores) must not be masked
/// by a 2x prefill win, especially since f16/packed decode does not even use
/// this backend — only f32 quantized linears and plain matrices do.
fn pick_by_score(
    gemv_scalar: f64,
    gemv_blas: f64,
    prefill_scalar: f64,
    prefill_blas: f64,
) -> Backend {
    const DECODE_WEIGHT: f64 = 4.0;
    let scalar = DECODE_WEIGHT * gemv_scalar + prefill_scalar;
    let blas = DECODE_WEIGHT * gemv_blas + prefill_blas;
    if blas < scalar {
        Backend::Blas
    } else {
        Backend::Scalar
    }
}

/// Micro-benchmark a decode GEMV and a prefill-shaped matrix-matrix multiply
/// to pick the faster CPU path on this machine. `nthreads` comes from the CLI
/// so the probe measures the same configuration used for inference.
fn pick_fastest_cpu(nthreads: usize) -> Backend {
    let n = nthreads.max(1);
    let reps = 3u32;

    // Decode shape: one input row, large weight matrix (memory-bound GEMV).
    let (m, k) = (4096usize, 1024usize);
    let w: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32) * 0.1).collect();
    let mut y = vec![0.0f32; m];
    crate::tensor::matmul_t_scalar(&x, 1, k, &w, m, &mut y, n);
    blas_matmul(&x, 1, k, &w, m, &mut y);

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        crate::tensor::matmul_t_scalar(&x, 1, k, &w, m, &mut y, n);
    }
    let gemv_scalar = t0.elapsed().as_secs_f64() / reps as f64;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        blas_matmul(&x, 1, k, &w, m, &mut y);
    }
    let gemv_blas = t0.elapsed().as_secs_f64() / reps as f64;

    // Prefill shape: a batch of input rows (compute-bound matrix-matrix).
    let (rows, k2, out) = (32usize, 1024usize, 4096usize);
    let wx: Vec<f32> = (0..out * k2)
        .map(|i| ((i % 89) as f32) * 0.01 - 0.4)
        .collect();
    let xx: Vec<f32> = (0..rows * k2).map(|i| ((i % 19) as f32) * 0.05).collect();
    let mut yy = vec![0.0f32; rows * out];
    crate::tensor::matmul_t_scalar(&xx, rows, k2, &wx, out, &mut yy, n);
    blas_matmul(&xx, rows, k2, &wx, out, &mut yy);

    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        crate::tensor::matmul_t_scalar(&xx, rows, k2, &wx, out, &mut yy, n);
    }
    let prefill_scalar = t0.elapsed().as_secs_f64() / reps as f64;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        blas_matmul(&xx, rows, k2, &wx, out, &mut yy);
    }
    let prefill_blas = t0.elapsed().as_secs_f64() / reps as f64;

    let chosen = pick_by_score(gemv_scalar, gemv_blas, prefill_scalar, prefill_blas);
    eprintln!(
        "[compute] CPU probe ({} threads): GEMV scalar={:.2}ms blas={:.2}ms | prefill scalar={:.2}ms blas={:.2}ms -> using {}",
        n,
        gemv_scalar * 1e3,
        gemv_blas * 1e3,
        prefill_scalar * 1e3,
        prefill_blas * 1e3,
        chosen.name()
    );
    chosen
}

#[cfg(not(feature = "gpu"))]
fn detect_gpu() -> Option<()> {
    None
}

#[cfg(feature = "gpu")]
fn detect_gpu() -> Option<candle_core::Device> {
    #[cfg(feature = "gpu-cuda")]
    if candle_core::utils::cuda_is_available() {
        if let Ok(d) = candle_core::Device::new_cuda(0) {
            return Some(d);
        }
    }
    #[cfg(feature = "gpu-metal")]
    if candle_core::utils::metal_is_available() {
        if let Ok(d) = candle_core::Device::new_metal(0) {
            return Some(d);
        }
    }
    None
}

/// Compute y = x @ W^T. `x` is (x_rows, x_cols) row-major, `w` is
/// (w_rows=out, x_cols) row-major, `y` is (x_rows, w_rows) row-major.
/// Dispatches to the selected backend; `nthreads` is used by the scalar path.
pub fn matmul_t(
    x: &[f32],
    x_rows: usize,
    x_cols: usize,
    w: &[f32],
    w_rows: usize,
    y: &mut [f32],
    nthreads: usize,
) {
    let be = backend();
    match be {
        Backend::Scalar => scalar_matmul(x, x_rows, x_cols, w, w_rows, y, nthreads),
        Backend::Blas => blas_matmul(x, x_rows, x_cols, w, w_rows, y),
        Backend::Gpu => {
            #[cfg(feature = "gpu")]
            if gpu_matmul(x, x_rows, x_cols, w, w_rows, y) {
                return;
            }
            blas_matmul(x, x_rows, x_cols, w, w_rows, y)
        }
    }
}

/// Scalar path: the original hand-written threaded dot-product loops.
fn scalar_matmul(
    x: &[f32],
    x_rows: usize,
    x_cols: usize,
    w: &[f32],
    w_rows: usize,
    y: &mut [f32],
    nthreads: usize,
) {
    // Delegate to the existing implementation in tensor.rs via a helper that
    // uses its own scoped-thread writer.
    crate::tensor::matmul_t_scalar(x, x_rows, x_cols, w, w_rows, y, nthreads);
}

/// CPU BLAS path using matrixmultiply's SIMD sgemm kernel.
/// We compute Y = X @ W^T with row-major X (x_rows x x_cols), W (w_rows x x_cols).
/// sgemm computes C = alpha*A*B + beta*C. Let A = X (m=x_rows, k=x_cols),
/// B = W^T (k=x_cols, n=w_rows). W is row-major (w_rows, x_cols) so W^T has
/// row stride 1 and col stride x_cols. C = Y (m=x_rows, n=w_rows) row-major.
fn blas_matmul(x: &[f32], x_rows: usize, x_cols: usize, w: &[f32], w_rows: usize, y: &mut [f32]) {
    if x_rows == 0 || w_rows == 0 || x_cols == 0 {
        return;
    }
    // `sgemm` takes raw pointers and cannot check slice lengths for us. A
    // mismatch here used to be an out-of-bounds read/write instead of a clear
    // error; the higher-level `tensor::matmul_t` has always asserted this.
    let w_needed = w_rows
        .checked_mul(x_cols)
        .expect("matmul weight size overflow");
    let y_needed = x_rows
        .checked_mul(w_rows)
        .expect("matmul output size overflow");
    let x_needed = x_rows
        .checked_mul(x_cols)
        .expect("matmul input size overflow");
    assert!(x.len() >= x_needed, "matmul input buffer too small");
    assert!(w.len() >= w_needed, "matmul weight buffer too small");
    assert!(y.len() >= y_needed, "matmul output buffer too small");
    unsafe {
        matrixmultiply::sgemm(
            x_rows,          // m
            x_cols,          // k
            w_rows,          // n
            1.0,             // alpha
            x.as_ptr(),      // A
            x_cols as isize, // rsa (row stride of X)
            1,               // csa
            w.as_ptr(),      // B = W^T
            1,               // rsb (row stride of W^T == 1)
            x_cols as isize, // csb (col stride of W^T == x_cols)
            0.0,             // beta
            y.as_mut_ptr(),  // C
            w_rows as isize, // rsc
            1,               // csc
        );
    }
}

#[cfg(feature = "gpu")]
fn weight_fingerprint(w: &[f32]) -> u32 {
    // FNV-1a over sampled positions (including the last). Cheap relative to
    // an upload, and catches recycled buffers holding different weights.
    let mut h = 2166136261u32;
    let n = w.len();
    for i in [0usize, n / 4, n / 2, (n * 3) / 4, n.saturating_sub(1)] {
        if i < n {
            h = (h ^ w[i].to_bits()).wrapping_mul(16777619);
        }
    }
    h
}

#[cfg(feature = "gpu")]
fn gpu_matmul(
    x: &[f32],
    x_rows: usize,
    x_cols: usize,
    w: &[f32],
    w_rows: usize,
    y: &mut [f32],
) -> bool {
    use candle_core::Tensor;
    let device = match COMPUTE.get().and_then(|c| c.device.as_ref()) {
        Some(d) => d,
        None => return false,
    };
    if x_rows != 1 {
        return false; // decode is GEMV; fall back for multi-row
    }
    // Pointer alone is unsafe as a cache key: if an address is recycled for
    // a different weight matrix, a stale device tensor would be reused.
    // A cheap fingerprint of a few sampled elements disambiguates reuse.
    let key = (w.as_ptr() as usize, w_rows, x_cols, weight_fingerprint(w));
    let cache = GPU_CACHE.get_or_init(|| GpuCache {
        tensors: Mutex::new(HashMap::new()),
    });
    let ws = {
        let mut tensors = cache.tensors.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = tensors.get(&key) {
            t.clone()
        } else {
            let t = match Tensor::from_slice(w, (w_rows, x_cols), device) {
                Ok(t) => t,
                Err(_) => return false,
            };
            // Bound the cache in case a caller passes transient buffers.
            if tensors.len() >= 512 {
                tensors.clear();
            }
            tensors.insert(key, t.clone());
            t
        }
    };
    let xs = match Tensor::from_slice(x, (x_cols,), device) {
        Ok(t) => t,
        Err(_) => return false,
    };
    // CUDA matmul requires contiguous operands; `t()` returns a strided view.
    let wt = match ws.t().and_then(|t| t.contiguous()) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let res = xs
        .reshape((1usize, x_cols))
        .and_then(|x2| x2.matmul(&wt))
        .and_then(|o| o.reshape((w_rows,)))
        .and_then(|o| o.to_vec1::<f32>());
    match res {
        Ok(v) if v.len() == y.len() => {
            y.copy_from_slice(&v);
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dot;

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_cache_fingerprint_detects_recycled_buffer() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let mut b = a.clone();
        assert_eq!(weight_fingerprint(&a), weight_fingerprint(&b));
        b[512] += 1.0; // sampled position changed -> new cache key
        b[1023] = -7.0; // last element changed too
        assert_ne!(weight_fingerprint(&a), weight_fingerprint(&b));
    }

    #[test]
    fn decode_is_weighted_more_than_prefill_when_picking_backend() {
        // Measured on a 12th-gen Windows host (ms): scalar wins GEMV by ~4x,
        // BLAS wins prefill by ~2.3x. Decode must win this trade-off.
        assert_eq!(
            pick_by_score(0.53e-3, 2.17e-3, 8.76e-3, 3.86e-3),
            Backend::Scalar
        );
        // If one backend wins both shapes there is no trade-off.
        assert_eq!(
            pick_by_score(1.0e-3, 3.0e-3, 2.0e-3, 5.0e-3),
            Backend::Scalar
        );
        assert_eq!(pick_by_score(3.0e-3, 1.0e-3, 9.0e-3, 2.0e-3), Backend::Blas);
    }

    #[test]
    fn blas_rejects_undersized_buffers() {
        let w = vec![1.0f32; 12]; // (3, 4)
        let x = vec![1.0f32; 4];
        let mut short = vec![0.0f32; 2];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            blas_matmul(&x, 1, 4, &w, 3, &mut short);
        }));
        assert!(result.is_err(), "undersized output must not reach sgemm");
    }

    #[test]
    fn blas_matches_dot() {
        // w = (3, 4), x = (1, 4) -> y = W @ x (3 outputs)
        let w: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let x: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5];
        let mut yb = vec![0.0f32; 3];
        blas_matmul(&x, 1, 4, &w, 3, &mut yb);
        for j in 0..3 {
            let expect = dot(&x, &w[j * 4..j * 4 + 4]);
            assert!(
                (yb[j] - expect).abs() < 1e-4,
                "blas[{}]={} dot={}",
                j,
                yb[j],
                expect
            );
        }
    }

    #[test]
    fn blas_matches_dot_multirow() {
        let w: Vec<f32> = (0..6).map(|i| i as f32).collect(); // (2,3)
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2,3)
        let mut yb = vec![0.0f32; 4];
        blas_matmul(&x, 2, 3, &w, 2, &mut yb);
        for r in 0..2 {
            for j in 0..2 {
                let expect = dot(&x[r * 3..r * 3 + 3], &w[j * 3..j * 3 + 3]);
                assert!((yb[r * 2 + j] - expect).abs() < 1e-4);
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_blas_vs_scalar() {
        // Representative GEMV: out=6144, in=1024 (a large qkv projection).
        let (rows, cols) = (6144usize, 1024usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i % 251) as f32 * 0.01).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 * 0.1).collect();
        let mut y = vec![0.0f32; rows];
        // warm up
        blas_matmul(&x, 1, cols, &w, rows, &mut y);
        crate::tensor::matmul_t_scalar(&x, 1, cols, &w, rows, &mut y, 1);
        crate::tensor::matmul_t_scalar(&x, 1, cols, &w, rows, &mut y, 4);

        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            crate::tensor::matmul_t_scalar(&x, 1, cols, &w, rows, &mut y, 1);
        }
        let scalar1 = t0.elapsed().as_secs_f64() / 20.0;
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            crate::tensor::matmul_t_scalar(&x, 1, cols, &w, rows, &mut y, 4);
        }
        let scalar = t0.elapsed().as_secs_f64() / 20.0;
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            blas_matmul(&x, 1, cols, &w, rows, &mut y);
        }
        let blas = t0.elapsed().as_secs_f64() / 20.0;
        eprintln!(
            "BENCH GEMV {}x{}: scalar(1thr)={:.5}s scalar(4thr)={:.5}s blas={:.5}s",
            rows, cols, scalar1, scalar, blas
        );
    }
}
