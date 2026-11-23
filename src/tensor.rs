//! Row-major f32 matrix plus matmul helpers (no external deps).
//!
//! Weights follow the PyTorch convention: a linear layer with
//! `out_features` and `in_features` stores its weight as a matrix of shape
//! `[out_features, in_features]`, and computes `y = x @ W^T`.

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0f32; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix size mismatch");
        Matrix { rows, cols, data }
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Dot product of two equal-length slices.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    for k in 0..n {
        acc += a[k] * b[k];
    }
    acc
}

/// Compute `y = x @ W^T`.
///
/// - `x` is `x_rows * x_cols` (row-major).
/// - `w` is `out x in` with `in == x_cols`.
/// - `y` must be `x_rows * out` (row-major).
///
/// Uses `nthreads` OS threads when beneficial (single-token decode splits over
/// output neurons; multi-token prefill splits over rows).
pub fn matmul_t(
    x: &[f32],
    x_rows: usize,
    x_cols: usize,
    w: &Matrix,
    y: &mut [f32],
    nthreads: usize,
) {
    assert_eq!(x_cols, w.cols, "matmul inner dim mismatch");
    assert_eq!(y.len(), x_rows * w.rows, "matmul output size mismatch");
    crate::compute::matmul_t(x, x_rows, x_cols, &w.data, w.rows, y, nthreads);
}

/// Original hand-written threaded scalar matmul (the "scalar" backend).
pub fn matmul_t_scalar(
    x: &[f32],
    x_rows: usize,
    x_cols: usize,
    w: &[f32],
    w_rows: usize,
    y: &mut [f32],
    nthreads: usize,
) {
    assert_eq!(y.len(), x_rows * w_rows, "matmul output size mismatch");
    if x_rows == 0 || w_rows == 0 || x_cols == 0 {
        return;
    }
    let n = nthreads.max(1);
    if n == 1 || (x_rows == 1 && w_rows < 256) {
        for i in 0..x_rows {
            let xrow = &x[i * x_cols..(i + 1) * x_cols];
            for j in 0..w_rows {
                y[i * w_rows + j] = dot(xrow, &w[j * x_cols..(j + 1) * x_cols]);
            }
        }
        return;
    }
    // Each worker writes to a disjoint region of `y`, so a raw base pointer
    // (wrapped in a Send newtype) is shared across the spawned closures.
    let base = SharedPtr(y.as_mut_ptr());
    std::thread::scope(|s| {
        if x_rows >= w_rows {
            // Split over input rows.
            let n = n.min(x_rows);
            let chunk = x_rows.div_ceil(n);
            for t in 0..n {
                let i0 = t * chunk;
                if i0 >= x_rows {
                    break;
                }
                let i1 = (i0 + chunk).min(x_rows);
                s.spawn(move || {
                    for i in i0..i1 {
                        let xrow = &x[i * x_cols..(i + 1) * x_cols];
                        for j in 0..w_rows {
                            base.write_at(
                                i * w_rows + j,
                                dot(xrow, &w[j * x_cols..(j + 1) * x_cols]),
                            );
                        }
                    }
                });
            }
        } else {
            // Split over output neurons.
            let n = n.min(w_rows);
            let chunk = w_rows.div_ceil(n);
            for t in 0..n {
                let j0 = t * chunk;
                if j0 >= w_rows {
                    break;
                }
                let j1 = (j0 + chunk).min(w_rows);
                s.spawn(move || {
                    for i in 0..x_rows {
                        let xrow = &x[i * x_cols..(i + 1) * x_cols];
                        for j in j0..j1 {
                            base.write_at(
                                i * w_rows + j,
                                dot(xrow, &w[j * x_cols..(j + 1) * x_cols]),
                            );
                        }
                    }
                });
            }
        }
    });
}

/// A raw mutable pointer that is explicitly `Send`, for sharing a base
/// address across scoped threads that each write to disjoint regions.
#[derive(Clone, Copy)]
struct SharedPtr(*mut f32);
unsafe impl Send for SharedPtr {}

impl SharedPtr {
    /// Write `v` at element offset `off` from the base address.
    ///
    /// # Safety
    /// Caller must ensure `off` indexes a live element of the original
    /// allocation and that no other thread writes the same element.
    #[inline(always)]
    fn write_at(&self, off: usize, v: f32) {
        unsafe {
            *self.0.add(off) = v;
        }
    }
}
