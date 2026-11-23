//! Compute backend selection: CPU BLAS (matrixmultiply sgemm), GPU (candle),
//! or scalar fallback. The best available backend is selected at startup and
//! used transparently by `tensor::matmul_t`.

use std::sync::OnceLock;

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

/// Detect the best backend. Preferences: "auto" (default), "scalar", "blas",
/// "gpu". Falls back to BLAS if GPU is requested but unavailable.
pub fn init(preference: &str) -> Backend {
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
                        // Decode is memory-bound GEMV: benchmark scalar vs
                        // sgemm once and pick the faster one for this machine.
                        pick_fastest_cpu()
                    }
                }
                _ => Backend::Blas,
            };
            eprintln!(
                "[compute] backend={}  (available: cpu-blas{} )",
                backend.name(),
                if gpu_present { ", gpu" } else { "" }
            );
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

/// Micro-benchmark a representative GEMV to decide whether the hand-written
/// threaded dot or the BLAS sgemm kernel is faster on this machine. The result
/// is cached with the backend selection.
fn pick_fastest_cpu() -> Backend {
    let (m, k) = (4096usize, 1024usize);
    let w: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
    let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32) * 0.1).collect();
    let mut y = vec![0.0f32; m];
    // Warm up both paths.
    crate::tensor::matmul_t_scalar(&x, 1, k, &w, m, &mut y, 4);
    blas_matmul(&x, 1, k, &w, m, &mut y);

    let reps = 3;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        crate::tensor::matmul_t_scalar(&x, 1, k, &w, m, &mut y, 4);
    }
    let scalar = t0.elapsed();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        blas_matmul(&x, 1, k, &w, m, &mut y);
    }
    let blas = t0.elapsed();
    let chosen = if blas < scalar {
        Backend::Blas
    } else {
        Backend::Scalar
    };
    eprintln!(
        "[compute] CPU probe: scalar={:?} blas={:?} -> using {}",
        scalar,
        blas,
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
    let xs = match Tensor::from_slice(x, (x_cols,), device) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let ws = match Tensor::from_slice(w, (w_rows, x_cols), device) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let res = xs
        .reshape((1usize, x_cols))
        .and_then(|x2| x2.matmul(&ws.t()?))
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
