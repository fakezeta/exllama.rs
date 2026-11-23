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

/// Row-major matrix with fp16-encoded weights (2 bytes/weight). Used for the
/// token embedding table: a single row is decoded to f32 on lookup, which
/// halves the largest remaining allocation for untied models.
pub struct F16Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Raw fp16 bit patterns, row-major.
    pub data: Vec<u16>,
}

impl F16Matrix {
    /// Decode one row into f32 values.
    pub fn row_f32(&self, row: usize) -> Vec<f32> {
        (0..self.cols)
            .map(|j| crate::exl3::half_to_f32(self.data[row * self.cols + j]))
            .collect()
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_matrix_row_decodes_to_f32() {
        let enc = crate::exl3::f32_to_half;
        let m = F16Matrix {
            rows: 2,
            cols: 3,
            data: vec![enc(1.0), enc(2.0), enc(3.0), enc(-1.5), enc(0.0), enc(0.25)],
        };
        assert_eq!(m.row_f32(0), vec![1.0, 2.0, 3.0]);
        assert_eq!(m.row_f32(1), vec![-1.5, 0.0, 0.25]);
    }

    #[test]
    fn dot_requires_equal_lengths() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        let result = std::panic::catch_unwind(|| dot(&[1.0, 2.0], &[3.0]));
        assert!(
            result.is_err(),
            "a length mismatch must not be silently truncated"
        );
    }
}

/// Dot product of two equal-length slices.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Silently truncating to the shorter slice hid shape bugs (a wrong-sized
    // projection looked like a plausible answer). All callers pass equal
    // lengths; enforce that invariant in every build, not just debug.
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut acc = 0.0f32;
    for k in 0..a.len() {
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
    // Each worker writes to a disjoint region of `y`. The pool joins all jobs
    // before returning, so raw pointers to the borrowed slices stay valid.
    let base = SharedPtr(y.as_mut_ptr());
    let xp = crate::threadpool::SendSlice::new(x);
    let wp = crate::threadpool::SendSlice::new(w);
    let mut jobs: Vec<Box<dyn FnOnce() + Send + 'static>> = Vec::new();
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
            let (base, xp, wp) = (base, xp, wp);
            jobs.push(Box::new(move || unsafe {
                let x = xp.get(x_rows * x_cols);
                let w = wp.get(w_rows * x_cols);
                for i in i0..i1 {
                    let xrow = &x[i * x_cols..(i + 1) * x_cols];
                    for j in 0..w_rows {
                        base.write_at(i * w_rows + j, dot(xrow, &w[j * x_cols..(j + 1) * x_cols]));
                    }
                }
            }));
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
            let (base, xp, wp) = (base, xp, wp);
            jobs.push(Box::new(move || unsafe {
                let x = xp.get(x_rows * x_cols);
                let w = wp.get(w_rows * x_cols);
                for i in 0..x_rows {
                    let xrow = &x[i * x_cols..(i + 1) * x_cols];
                    for j in j0..j1 {
                        base.write_at(i * w_rows + j, dot(xrow, &w[j * x_cols..(j + 1) * x_cols]));
                    }
                }
            }));
        }
    }
    crate::threadpool::run(jobs);
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
