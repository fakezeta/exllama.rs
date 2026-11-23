//! Tiny fixed-size worker pool. Matmuls used to spawn scoped threads on every
//! call (hundreds of times per token); workers here live for the process
//! lifetime and receive jobs through std channels.
//!
//! No external crates. Jobs finish before `run` returns, so borrowed data can
//! be shared safely via `SendSlice` (the pointee outlives every job).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Barrier, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct Pool {
    senders: Vec<Sender<Job>>,
    next: AtomicUsize,
}

static POOL: OnceLock<Pool> = OnceLock::new();

/// Initialise the pool at startup. Called once from `main` with the configured
/// thread count; later calls are no-ops.
pub fn init(threads: usize) {
    let n = threads.max(1);
    POOL.get_or_init(|| {
        let mut senders = Vec::with_capacity(n);
        for i in 0..n {
            let (tx, rx) = channel::<Job>();
            std::thread::Builder::new()
                .name(format!("exllama-worker-{}", i))
                .spawn(move || worker(rx))
                .expect("failed to spawn worker thread");
            senders.push(tx);
        }
        Pool {
            senders,
            next: AtomicUsize::new(0),
        }
    });
}

fn worker(rx: Receiver<Job>) {
    for job in rx {
        // A panicking job must not kill the worker; the barrier guard inside
        // the job still releases the caller.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
    }
}

/// An immutable slice pointer that may cross thread boundaries.
#[derive(Clone, Copy)]
pub struct SendSlice<T>(*const T);
unsafe impl<T> Send for SendSlice<T> {}

impl<T> SendSlice<T> {
    pub fn new(slice: &[T]) -> Self {
        SendSlice(slice.as_ptr())
    }

    /// # Safety
    /// The original slice must still be alive when this is called. All jobs
    /// are joined by `run` before it returns, so this holds for jobs sent here.
    pub unsafe fn get(&self, len: usize) -> &'static [T] {
        std::slice::from_raw_parts(self.0, len)
    }
}

/// Releases the barrier even if the job panics.
struct BarrierGuard(Arc<Barrier>);
impl Drop for BarrierGuard {
    fn drop(&mut self) {
        self.0.wait();
    }
}

/// Run at most one job per worker, wait for all of them, then return.
/// Without `init`, jobs run sequentially (useful in unit tests).
pub fn run(jobs: Vec<Job>) {
    if jobs.is_empty() {
        return;
    }
    let Some(pool) = POOL.get() else {
        for job in jobs {
            job();
        }
        return;
    };
    let width = pool.senders.len();
    let mut jobs = jobs.into_iter();
    // Run every job; if there are more than workers, process them in batches.
    loop {
        let batch: Vec<Job> = jobs.by_ref().take(width).collect();
        if batch.is_empty() {
            break;
        }
        if batch.len() == 1 {
            batch.into_iter().next().unwrap()();
            continue;
        }
        let n = batch.len();
        let barrier = Arc::new(Barrier::new(n + 1));
        let start = pool.next.fetch_add(n, Ordering::Relaxed) % width;
        for (i, job) in batch.into_iter().enumerate() {
            let tx = &pool.senders[(start + i) % width];
            let b = barrier.clone();
            tx.send(Box::new(move || {
                let _guard = BarrierGuard(b);
                job();
            }))
            .expect("worker thread gone");
        }
        barrier.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_executes_all_jobs() {
        init(4);
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut jobs: Vec<Job> = Vec::new();
        for _ in 0..10 {
            let c = counter.clone();
            jobs.push(Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }
        run(jobs);
        assert_eq!(counter.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn run_without_init_executes_all_jobs() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut jobs: Vec<Job> = Vec::new();
        for _ in 0..4 {
            let c = counter.clone();
            jobs.push(Box::new(move || {
                c.fetch_add(1, Ordering::Relaxed);
            }));
        }
        run(jobs);
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }
}
