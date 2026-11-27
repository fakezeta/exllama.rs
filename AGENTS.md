# AGENT.md — Developer Guide for exllama.rs

This document is a complete orienteering guide for AI agents (or humans) continuing development of this codebase. Read it fully before making changes. It captures the *why* behind the code, not just the *what*, so you don't have to re-derive the research.

**Before planning or prioritizing any non-trivial work, read `IMPROVEMENT_PLAN.md`.** It is the source of truth for open priorities and acceptance criteria; update it when priorities or measured results change.

---

## 1. Project Overview

**Goal:** A minimal Rust inference server for ExLlamaV3 (EXL3) quantized models, focused on the **Qwen3.5 text decoder** (`Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM`). It exposes an OpenAI-compatible HTTP API.

**Dependency policy:** JSON, HTTP, tokenizer, regex, PRNG and dequantizer are hand-written in `src/` with the standard library only. The only (non-optional) crate is `matrixmultiply` for the SIMD sgemm kernel; `candle-core` and `half` are opt-in for GPU. Do not add general-purpose crates (serde, tokio, serde_json, etc.) without explicit approval.

**Model scope:** Qwen3.5 only. The vision tower of multimodal checkpoints is ignored; text-only inference works. Other architectures are out of scope.

**Runtime:** CPU by default; optional GPU builds (`gpu-metal`/`gpu-cuda`) run a device-resident forward. No Python needed at runtime.

---

## 2. Quick Reference

### Build & test
```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build                      # debug
cargo build --release            # optimized binary at target/release/exllama-server
cargo test                       # 131 unit tests (11 ignored: real-model / /tmp fixtures + benchmarks)
cargo test --features gpu        # 161 tests (adds device-op + packed-math tests on candle's CPU device)
cargo test --features gpu-cuda   # 167 tests on a CUDA host (adds real-GPU kernel parity; use -- --test-threads=1)
cargo fmt --all
cargo clippy --all-targets -- -D warnings   # must pass (CI enforces)
```

### Run
```bash
./target/release/exllama-server \
  --model /path/to/model-dir \
  --tokenizer /path/to/model-dir/tokenizer.json \
  --context-size 4096 --port 8080
```
Defaults: context 4096, port 8080, host 0.0.0.0, max-tokens 512, threads = all CPUs.
`--tokenizer` defaults to `tokenizer.json` next to the model file.

### Endpoints
| Method | Path | Handler |
|--------|------|---------|
| GET | `/v1/models` / `/models` | `models_json()` in main.rs |
| GET | `/health` / `/` | static `{"status":"ok"}` |
| POST | `/v1/chat/completions` | `server::handle_chat` |
| POST | `/v1/completions` | `server::handle_completions` |
| OPTIONS | any | 204 (CORS preflight) |

### API parameters (chat & completions)
`temperature` (default 0.7), `top_p` (1.0), `top_k` (0 = off), `max_tokens` / `max_completion_tokens`, `stop` (string or array), `stop_token_ids`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias` (token-id → bias), `stream` (bool), `stream_options.include_usage`, `n` (1..=128), `user` (validated, no-op), `response_format` (`text` / `json_object`; best-effort hint, no constrained decoding), `logprobs` + `top_logprobs` (chat: bool + 0..=20; completions: integer with parallel-array response), and for chat: `enable_thinking` (bool, default **true** for 9B/4B). `model` is validated against the server name. Invalid parameters are rejected with a 400 (`repetition_penalty` must be finite and `> 0`). `n` choices are generated sequentially (per-choice seeds derived from `seed`); with greedy decoding they are identical. Logprobs use the penalty-adjusted distribution before temperature/top-k/top-p and exclude reasoning delimiters and stop tokens. A single generation has a 300-second wall-clock deadline (see `MAX_GENERATION_SECONDS`) so one stalled request cannot hold the model mutex forever.

---

## 3. Module Map

### `src/main.rs` — CLI + routing
- `Cli` struct + `parse_cli()` — hand-rolled arg parser (`-m/--model`, `-c/--context-size`, `-p/--port`, `--host`, `--tokenizer`, `--max-tokens`, `--threads`, `--name`, `--weight-mode`, `--gpu-weight-mode` (`f16`|`packed`), `--compute`, `--instances`). A single iterator owns argument advancement (no manual index increments); `--context-size` is clamped to `1..=1_000_000`, `--threads` to `1..=1024`, and `--instances` to `1..=64`.
- `main()` — loads `Model`, `Tokenizer`, builds `Engine`, wraps in `Arc`, binds `TcpListener`, calls `http::serve`.
- `route()` — dispatches HTTP requests (see table above).
- `models_json()` — builds `/v1/models` response.
- Has `#![allow(dead_code)]` at crate root (intentional unused helpers).

### `src/http.rs` — Minimal HTTP/1.1 server
- `Request { method, path, headers, body }` — parsed by `read_request()` (64 KiB header cap, 8 MiB body cap, rejects duplicate/invalid Content-Length).
- `respond()` — writes a complete response with Content-Length and CORS headers; 429 maps to "Too Many Requests" for admission backpressure.
- `Streamer` — chunked SSE writer with `Cache-Control: no-cache`; `write_chunk()` returns false on a broken connection so generation aborts; `end(&mut self)` writes `0\r\n\r\n` (takes `&mut self` so callers sharing the stream with generation callbacks can still finish it).
- `serve(listener, handler)` — accept loop; one OS thread per connection (capped at 64), 600 s read/write timeouts. Handler signature: `Fn(TcpStream, Request)` (takes ownership of the stream).
- `read_request()` reads the request line and headers through `Read::take(MAX_HEADER_BYTES + 1)` so a newline-free flood is rejected instead of buffered without limit. Body is capped at 8 MiB.

### `src/json.rs` — JSON parser/serializer
- `Json` enum: `Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>)` (no separate Float variant).
- Free function `json::parse(&str) -> Result<Json, String>` (NOT a method). Recursive-descent `Parser` handles strings with `\u` escapes, numbers, arrays, objects.
- `Json` has `get`, `as_str`, `as_i64`, `as_usize`, `as_f64`, `as_bool`, `as_arr`, `as_obj`, `to_string()` (via `Display`).
- `Json::Obj` preserves insertion order (used for response field ordering).
- `obj(&[(&str, Json)])` convenience constructor.

### `src/tensor.rs` — Matrices + matmul
- `Matrix { rows, cols, data: Vec<f32> }` — row-major. **Weights are stored `(out_features, in_features)`**; a linear computes `y = x @ W^T` via `matmul_t`.
- `F16Matrix { rows, cols, data: Vec<u16> }` — fp16 embedding storage for **all** plain embeddings (tied and untied); `row_f32(row)` decodes one row on lookup, and `exl3::matmul_f16` decodes rows in parallel for a tied `lm_head`.
- `matmul_t(x, x_rows, x_cols, w, y, nthreads)` — splits work over the persistent pool (over rows if many input rows, over output neurons otherwise).
- `dot(a, b)` asserts equal lengths in every build (previously `min(a.len(), b.len())` silently hid shape bugs); `exl3::dot_f16` does the same.
- **Rust 1.100 quirk:** `NonNull<f32>` and `*mut f32` are NOT `Send`. Workaround: `SharedPtr(*mut f32)` newtype with `unsafe impl Send`, `#[derive(Clone, Copy)]`, and a safe `write_at(offset, value)` method. Closures capture `SharedPtr` **by value** (Copy). Do not revert to raw pointers. Immutable inputs cross threads via `threadpool::SendSlice<T>`.

### `src/threadpool.rs` — Persistent worker pool
- `init(threads)` spawns a fixed set of workers (called from `main` with `--threads`); `run(jobs)` dispatches `Box<dyn FnOnce() + Send>` jobs through std channels and waits on a `Barrier` before returning.
- Jobs execute sequentially if `init` was never called (unit tests). A `BarrierGuard` releases the barrier even if a job panics, and the worker catches panics so the pool survives.
- A global `RUN_LOCK` serializes complete `run()` calls: concurrent callers used to interleave jobs on the same workers and could deadlock across per-call barriers (a worker blocked in one call's guard cannot run another call's queued job). Generation is already serialized by `Engine.model`, so this costs nothing in normal operation.
- Used by `tensor::matmul_t_scalar` and `QuantLinear::matmul` (f16/packed paths) instead of spawning scoped threads per call.

### `src/compute.rs` — Compute backend selection
- `Backend::{Scalar, Blas, Gpu}`; `init(preference)` picks at startup. `auto` (default): if GPU compiled in (`--features gpu-metal`/`gpu-cuda`) and a device is present → Gpu; else micro-benchmarks a 4096x1024 GEMV (`pick_fastest_cpu()`) and picks the faster of scalar vs `matrixmultiply::sgemm`.
- `matmul_t(x, x_rows, x_cols, w: &[f32], w_rows, y, nthreads)` dispatches to sgemm (Blas), candle `Tensor::matmul` (Gpu, x_rows==1), or `tensor::matmul_t_scalar`.
- CLI `--compute auto|blas|gpu|scalar`. Global `OnceLock<Compute>` holds backend + optional candle device.
- **NOTE:** for CPU decode GEMV (memory-bound), the hand-written threaded scalar dot is typically 3-4x *faster* than sgemm — so `auto` often picks `scalar`. That is expected, not a regression. sgemm wins for matrix-matrix (prefill batches).
- `blas-src`/`openblas-src` were rejected: netlib needs a Fortran compiler, OpenBLAS needs a native lib — both break portable CI. `matrixmultiply` is pure Rust with SIMD kernels (the sgemm core used by ndarray).
- Cargo features: `gpu` (candle-core + half), `gpu-metal` (Apple), `gpu-cuda` (NVIDIA); default build has no GPU deps.
- GPU weight cache keys include a cheap fingerprint of sampled weight elements alongside `(pointer, rows, cols)`, so a recycled address cannot reuse a stale device tensor. `weight_fingerprint` has a GPU-only unit test. **Note:** the device-resident path in `gpu.rs` supersedes `compute::gpu_matmul` for quantized linears; the latter remains for plain matrices on the CPU fallback and for the old host-forward path.

### `src/fused.rs` — Fused decode ops (feature `gpu`)
- Hot-path ops are candle `CustomOp1/2/3` with numerically exact CPU implementations (also the fallback) and single-kernel CUDA/PTX implementations via `CudaDevice::get_or_load_custom_func`: RMSNorm, partial RoPE (Phase 2a), then gated RMSNorm, DeltaNet `g`/`beta`, GQA softmax, the sigmoid gate multiply (Phase 2b), DeltaNet q/k L2-normalization + head repetition, depthwise conv1d + SiLU, and in-place KV/conv row writes via `InplaceOp2`. `deltanet_gates` returns `(g, beta)` as a `(2, nvh)` tensor — keep that shape in the fallback too, or the caller's `narrow`/`reshape` breaks. The sigmoid gate makes its operands contiguous first (the attention output is a transposed view). **Each kernel needs a distinct module name** — candle caches a module by name, so reusing one name makes the second `load_function` fail with `CUDA_ERROR_NOT_FOUND`. A kernel-load/launch failure transparently falls back (and disables that kernel for the process, so a bad PTX is not recompiled every call); `EXLLAMA_DISABLE_FUSED=1` forces the fallback for A/B testing. PTX must be validated with `ptxas -arch=sm_80` before shipping — JIT failures surface only as `CUDA_ERROR_INVALID_PTX` at runtime, and `ex2.approx.f32` needs an explicit `log2(e)` multiply. `cargo test --features gpu-cuda cuda_ -- --test-threads=1` checks all kernels numerically against the fallback on a real GPU (five `cuda_*_match*` tests covering RMSNorm/RoPE, phase 2b, q/k norm, KV row write, conv step); run them serially — concurrent device creation in one test binary can confuse candle's per-device module cache. CI only exercises the CPU path (4 focused tests plus all whole-forward/sampling GPU-feature tests). Packed-mode ops live here too: `input_hadamard` (`x⊙suh` + H128 butterfly), `packed_linear` (trellis dequant-dot with exact f32 LUT, one thread per output) and `output_hadamard` (H128 + `svh`); the opt-in `dp4a` variant loads an INT8 decode LUT uploaded by `GpuWeights`. A k-split warp-parallel dequant was tried and reverted: shared/shuffle reductions were racy on the RTX 5060 Ti driver — do not reintroduce it without a deterministic reduction. `cargo test --features gpu-cuda cuda_ -- --test-threads=1` now runs **six** real-GPU tests (the extra one is `cuda_packed_linear_matches_cpu_reference`); its tolerance is looser under `EXLLAMA_PACKED_DP4A=1` because INT8 is approximate.

### `src/gpu.rs` — Device-resident decoder (feature `gpu`)
- Built only with a `gpu` feature. `GpuWeights::build()` decodes all EXL3 W_hat tiles **once at load** and uploads them as fp16 device tensors; plain matrices and embeddings are uploaded as fp16, vectors/RoPE/H128 as fp32. No weight crosses the bus during inference.
- `GpuRunner` owns shared `Arc<GpuWeights>` plus per-instance `GpuState`; `Model` delegates `forward()` to it when the GPU backend is active. Activations, fp16 KV cache and DeltaNet conv/recurrent state all stay on device for the whole request. Only the input token ids and the final logits cross host/device — one logits copy per `Model::forward` call (for prefill with many tokens: one copy for the whole prompt).
- All forward math is reimplemented with candle tensor ops: RMSNorm, partial RoPE, per-128-block Hadamard, GQA attention with sigmoid gate, depthwise conv1d, Gated DeltaNet recurrence, per-head gated RMSNorm and the MLP. The CPU path in `model.rs` is unchanged and remains the fallback.
- **Candle shape gotcha (fixed):** `Tensor::mul()` requires *exactly* equal shapes — only `broadcast_mul()` broadcasts. Four bugs came from this: (1) `rms_norm` multiplied a 2-D `(heads, head_dim)` q/k tensor by a 1-D weight with `.mul()`; (2) the per-head gated norm multiplied `(nvh, vd)` by `(1, vd)`; (3) q_proj was reshaped to `(nh, 2, hd)` instead of `(nh, hd*2)` (query|gate are contiguous per head, see `split_q_gate`); (4) the RoPE tables `(context, rope_half)` were indexed with `narrow(0, pos, half)` (half *rows*) instead of one row. The full-attention KV gather also used `narrow(1, kh, hd)` on a `(tlen, nkv, hd)` tensor — it must select one KV head (`narrow(1, kh, 1)` + `squeeze(1)`).
- **Partial RoPE gotcha (fixed):** `rope_half = head_dim * partial_rotary_factor / 2` (e.g. 16 for `head_dim=128`, factor 0.25), so only the first `2*rope_half` dims of each head rotate. The GPU `rope()` originally returned only those dims and dropped the tail, so `k` had 128 elements where the KV cache row needed `nkv*head_dim=512` — the real-model error was `Shape mismatch, got buffer of size 128 which is compatible with shape [1, 512]`. It now concatenates the unrotated tail back (matching CPU `Model::apply_rope`), and `gpu::tests::rope_preserves_unrotated_tail` locks the numeric behavior. `device_forward_runs_end_to_end_on_cpu` now uses `rope_half=1 < head_dim/2` so this case is exercised in the full forward.
- **Two more GPU correctness bugs (fixed):** (1) `rms_norm()` reduced a 2-D `(heads, head_dim)` q/k norm over dim 0 (across heads) instead of the last dim, silently pooling statistics across heads — it now reduces over the last dim, matching CPU per-head norms (`gpu::tests::rms_norm_2d_normalizes_each_head`). (2) `mat()` resolved untied `lm_head.weight` to the embedding table *before* `q()` was tried; both have shape `(vocab, hidden)` so no error was raised and logits came from the wrong weights. The embedding is now only a last-resort tied-checkpoint fallback in `linear()` (`gpu::tests::linear_prefers_quantized_lm_head_over_tied_embedding`). Weight-upload, head-divisibility, projection-length and Hadamard-divisibility checks now fail with named diagnostics instead of opaque candle errors.
- With the GPU backend, `Model::load()` loads EXL3 with the `packed` host mode and ignores `--weight-mode`. `--gpu-weight-mode f16|packed` selects the **device** representation: `f16` (default) decodes/folds on the host and uploads `(n,k)` f16 matrices (`qhat: HashMap<String, QuantWeight>` with `QuantWeight::Dense`); `packed` uploads the trellis verbatim as U32 words plus `suh`/`svh` and exact codebook LUTs (`QuantWeight::Packed`) and decodes it in a CUDA kernel during the forward. No requantization: codebook LUTs are decode tables, not new weights. The packed path computes `xr = H128(x ⊙ suh)`, `t = W_hatᵀ @ xr` (fused PTX kernel in `fused.rs`, with an exact CPU fallback), `H128`, then `svh`, so no full-size device intermediates exist. It supports all three codebooks and integer/half bitrates (8/6/4/2 bpw). On the 2B 8-bpw model it uses ~1.6 GiB for quantized linears vs ~3.5 GiB for `f16`, at ~7 vs ~60 tok/s (speed optimization deliberately deferred). Logs `[gpu] weights uploaded (mode=...)` and `[gpu] device-resident forward active`.
- **GPU decode performance (fixed in v0.5.3):** a real-client run of the 0.8B model reported ~6.9 tok/s on GPU, *slower* than CPU (~11.9 tok/s), with GPU-Z showing only ~20% GPU load, ~33 W board power and ~50 °C — i.e. latency-bound, not compute/thermally limited. The cause was host-side: the Gated DeltaNet recurrence looped over all 32 value heads and pulled `g[vh]`/`beta[vh]` to the host with `to_vec0()` on every head (~576 full device synchronizations **per token**) plus ~20 tiny kernels per head (~11k launches/token); full attention similarly looped per query head. Both are now batched device ops: `deltanet_step_all()` advances every value head with one set of matmuls and no host round-trip, and `gqa_attention()` groups q/gate and the KV cache as `(nkv, groups, hd)` / `(tlen, nkv, hd)`. Do **not** reintroduce per-head loops or `to_vec0()` in the forward path. `gpu::tests::deltanet_all_matches_per_head` and `gqa_batched_matches_per_head` pin both rewrites against the old per-head math numerically. A third fix folds both 128-block Hadamards into every W_hat at load, so `q_matmul` is **one GEMV** (`svh ⊙ (B @ (x ⊙ suh))`) instead of two Hadamard GEMMs plus the W_hat GEMV; `folded_q_matmul_matches_unfolded` checks it numerically.
- **Load-time OOM from folding on device (fixed in v0.5.6):** v0.5.3–v0.5.5 folded W_hat with device matmuls; a vocab-sized `lm_head` (248320x1024 fp16 = 508 MB) needed several full-size device intermediates at once (~1.5 GB peak) and OOM'd on cards that already had ~2.9 GB in use by other processes. The fold now runs **on the host** in `exl3::fold_hadamards_host()` (using the existing f32 `hadamard_rows`/`hadamard_cols` butterflies) before `upload_device` uploads a single final `(n,k)` tensor — device transient memory for the fold is now zero. `gpu::fold_hadamards` remains as a device-side numeric reference (test-only now); `folded_q_matmul_matches_unfolded` checks both host and device folds against the unfolded path.
- **Hadamard-fold dimension bug (fixed in v0.5.4):** v0.5.3's `fold_hadamards()` read W_hat's `(n, k)` dims backwards (`dim(0)` is `n`) and returned the transposed matrix. Square synthetic weights hid it, but every non-square quantized linear failed at load with `matmul: weight input dim 6144 does not match input length 1024` (e.g. a `(6144, 1024)` gate/up proj folded to `(1024, 6144)`). The fold is now `B = H_n @ W_hat @ H_k` with the `(n, k)` shape explicitly preserved; `folded_q_matmul_matches_unfolded` uses a **non-square** fixture and `fold_preserves_shape_for_non_square_weights` locks the shape for both `n > k` and `k > n`.
- **CUDA matmul contiguity (fixed in v0.5.5):** candle's CUDA backend rejects `matmul` on strided views with `matmul is only supported for contiguous tensors ...`; the CPU backend happily accepts them, so CPU-device tests cannot catch it. The failing real-model case was `gqa_attention`'s `kmat.permute((1,0,2))` (shape `[2,2,256]`, stride `[256,512,1]`) against `qg.transpose(1,2)`. All runtime matmuls now go through `matmul_c()` in `gpu.rs`, which calls `.contiguous()` on both operands (a no-op for already-contiguous tensors); this includes batched GQA and both DeltaNet recurrences, plus `compute::gpu_matmul`'s `ws.t()` (the Hadamard fold no longer uses device matmul). Do **not** call `.matmul()` directly on the output of `permute`/`transpose`/`t` in GPU code — wrap it in `matmul_c` (or call `.contiguous()` explicitly).
- **On-device sampling (fixed in v0.5.3):** the device path no longer copies the full vocabulary of logits to the host each token. `GpuRunner::sample()` applies logit processors (repetition/presence/frequency penalties, bias) on device from a small history tensor, then does greedy/temperature/top-k/top-p sampling and logprob extraction with candle ops. top-k uses repeated argmax and top-p a 32-step logit-threshold search (candle 0.11 has no `topk`/`sort`); its CDF is walked in token-id order — distributionally identical to the host's score-order walk, but a *fixed* uniform can pick a different token. One f32 uniform from the request RNG goes up per step; one tiny tensor (token id, its logprob, up to `top_logprobs` pairs) comes back. `GpuSnapshot` carries device logits so `n > 1` choices still share one prefill. `server::prepare_logits`/`sample` are `pub(crate)` so `device_sample_matches_host_greedy_and_softmax`, `device_sample_matches_host_top_k_and_top_p` and `device_logprobs_match_host` compare device vs host directly.
- **Real OOM root cause — candle `cumsum` (fixed in v0.5.7):** the first-decode OOM on a 16 GB card came from `GpuRunner::sample()` calling `Tensor::cumsum(0)` on a `(vocab,)` tensor. Candle 0.11 implements `cumsum` as a matmul against an **upper-triangular `(n, n)` matrix** — at `vocab = 248320` that single temporary is ~246 GB. It failed instantly after prefill (no `[done]` line), with the GPU idle and flat VRAM, which is why earlier theories (load-time fold, top_k, context size) didn't fit. The inverse-CDF draw now runs on the host via `inverse_cdf()`: one ~1 MB D2H copy of the already top-k/top-p-filtered weights per token, no giant device allocation. `inverse_cdf_handles_full_vocab_size` locks it at the real vocab size. **Never call `Tensor::cumsum` on a vocabulary-sized tensor in candle 0.11.**
- **OOM triage follow-ups (fixed in v0.5.6):** a deep allocation audit confirmed the steady-state device footprint of the 0.8B model is ~2.8 GB (matching the flat GPU-Z reading), so a 16 GB OOM cannot come from the model itself. A real-client log proved the failure came on the first decode/sample step (no `[done]`) from a stale `v0.5.5` binary (its `[gpu] weights uploaded` line lacks the MiB/config fields). Real risks addressed: (1) device-side fold at load — now folded on host; (2) `--context-size` is only clamped to 1e6, and full-context KV caches scale linearly (~11.7 GB at 1e6) — a warning is now printed above 8192; (3) `top_k` had no upper bound, so the device sampler's repeated-argmax loop could allocate one vocab-sized tensor per candidate — now capped at 1024 in `server::parse_params` and defensively in `GpuRunner::sample`. Freed CUDA memory stays in cudarc's stream-ordered pool (no trim/synchronize), so per-thread streams across HTTP connections can also strand reserved VRAM.
- **Launch-count reduction (unreleased):** `suh`/`svh` are now baked into the folded weight on the host (`exl3::fold_hadamards_host`, in f32 before the single f16 rounding), so `q_matmul` is a bare GEMV and `qhat` stores one tensor per linear instead of `(B, suh, svh)`. All `*norm.weight` vectors get a precomputed `1 + w` at load (`GpuWeights.norm1`), removing one hidden-sized affine per norm call. Sampling skips the `ones`/`w * sel` work when `top_k == 0 && top_p == 1` and reuses one `-inf` and one `-40` device scalar instead of an H2D copy per loop iteration. All hot-path pointwise work now goes through `src/fused.rs` (one kernel each on CUDA): RMSNorm, RoPE, gated norm, DeltaNet gates and q/k normalization, GQA softmax + sigmoid gate, depthwise conv1d + SiLU, plus in-place KV/conv row writes that replaced full-cache `slice_scatter` copies. `q_matmul` is kept as a thin test wrapper around `matmul_w`.
- **Future work:** the remaining per-token overhead is the DeltaNet recurrence epilogues (`exp(g)`, `sub`/`add` around the batched matmuls in `deltanet_step_all`), the vocab-sized sampling kernels (repeated argmax / top-p search / 1 MB D2H), and the fp16→f32 conversion of the used KV prefix each step. CUDA graph capture is higher risk (sampling's D2H, variable `tlen`, per-thread streams, `--instances N`) and should follow another A/B measurement. Sampling still consumes one host RNG float per step. Real-GPU throughput is not exercised on CI.

### `src/rng.rs` — SplitMix64 PRNG
- `Rng::new(seed)`, `next_f64() -> f64`, `next_u32()`, private `next_u64()`. Used for sampling.

### `src/regex.rs` — Minimal backtracking regex engine
- `Regex::compile(pattern)` → `Regex` with `match_at` and `pre_tokenize` (isolated matches; used by tokenizer).
- Supports: alternation (`|`), concat, groups `(?:...)`, lookaheads `(?=...)`, `(?!...)`, inline case-insensitive `(?i:...)`, greedy and lazy quantifiers (`* + ? {n} {n,m}`, each with optional trailing `?`), character classes with ranges, escapes, `\s \d \w \S \D \W`, Unicode property classes `\p{L} \p{M} \p{N} \p{P} \p{S} \p{Z} \p{C}` (including inside `[...]`).
- Untrusted patterns are bounded: unbounded quantifiers cap at `MAX_QUANT_REPEATS = 4096` and each `match_at` gets a `MAX_MATCH_WORK = 1_000_000` step budget threaded through `possible_ends`/`all_seq_ends`. This prevents the old O(input-length) `Vec<Vec<usize>>` allocation and nested-quantifier blowups.
- `pre_tokenize()` implements **isolated** split behavior: matches are kept, gaps are dropped. This is exactly what the Qwen3.5 `Split` pre-tokenizer needs.
- Grouped subexpressions backtrack fully (`all_seq_ends`), so patterns like `(?:ab?|c)bb` match correctly.
- 10 tests, including `test_qwen_style` with the exact Qwen3.5 pattern and expected word lists, and `unbounded_quantifier_is_bounded` for the work caps.

### `src/tokenizer.rs` — Byte-level BPE tokenizer
- `Tokenizer::load(path)` — reads `tokenizer.json`: vocab, merges, byte_encoder, added_tokens, pre-tokenizer pattern. Unknown pre-tokenizer shapes are rejected (no silent `\s+` fallback); added tokens from the sibling `tokenizer_config.json` (`added_tokens_decoder`) are merged; `id_to_token` is built in sorted-id order for deterministic decode.
- `encode(&str) -> Vec<u32>`:
  1. **Added-token protection:** scans raw input for added tokens (plain substring, longest-first) and emits their ids directly, bypassing pre-tokenization.
  2. `encode_text()` on each remaining span: pre-tokenizer regex → `bpe_encode_word()` per word.
- `bpe_encode_word()` — converts word bytes to byte-level chars via GPT-2 byte map, then repeatedly merges the lowest-rank adjacent pair.
- `decode(&[u32], skip_special)` — maps token chars back to bytes, UTF-8 lossy decode.
- **EOS fallback:** tokenizer.json has `eos_token: null`; code falls back to `<|im_end|>` (id 248046), then `<|endoftext|>` (248044). `config.json`'s `eos_token_id` (scalar or first array element, top-level or `text_config`) overrides the fallback at startup. Critical for stopping generation.
- Qwen3.5 specifics: base vocab 248,044 (ids 0..248043), padded to 248,320; no BOS, no UNK; `add_prefix_space=false`.
- Added tokens 248044..248069 (26 in tokenizer.json). `<think>`=248068, `</think>`=248069, `<|im_start|>`=248045, `<|im_end|>`=248046, `<tool_call>`=248058.
- Tests verify encode + added-token protection against the real `/tmp/qwen35/tokenizer.json` (skipped if file absent).

### `src/chat.rs` — Qwen3.5 chat template
- `Message { role, content, reasoning_content, tool_calls, name }` plus `ToolCall { id, name, arguments }`.
- `apply_chat_template(messages, add_generation_prompt, enable_thinking) -> String`.
- Renders text-only subset of the Jinja template:
  - System first: `<|im_start|>system\n{trim}<|im_end|>\n`
  - User: `<|im_start|>user\n{trim}<|im_end|>\n`
  - Assistant: prefers an explicit `reasoning_content`; otherwise splits reasoning from content if `</think>` present. If `idx > last_query_index` emits `<think>\n{reasoning}\n</think>\n\n{body}`. Assistant `tool_calls` render as Qwen3.5 XML (`<tool_call><function=...><parameter=...>...`).
  - Tool messages: grouped into one user turn with `<tool_response>` blocks.
  - `tools_system_prompt()` builds the system preamble for OpenAI `tools`; `parse_tool_calls()` extracts generated `<tool_call>` blocks into response fields. Parameter names terminate at `>` **or** a line break — some clients inject object-valued pseudo-parameters (e.g. `__labels__`) that the model emits as `<parameter=name\n{...}` without `>`, which previously swallowed the next argument. Incomplete blocks are dropped, not left in content. Streaming uses `ToolCallStreamer` in `server.rs` to strip the XML from content and emit indexed `tool_calls` deltas instead.
  - Generation prompt: `<|im_start|>assistant\n` + `<think>\n` (thinking on) or `<think>\n\n</think>\n\n` (off).
- `enable_thinking` default is model-size-aware: `hidden_size >= 2560` → true (9B/4B), otherwise false (0.8B/2B); an explicit config value wins. `Params` seeds its default from `Engine.default_enable_thinking`.

### `src/server.rs` — Inference engine + API handlers
- `Engine { models: Vec<Mutex<Model>>, available, ready, tokenizer, model_name, default_max_tokens, threads, default_enable_thinking }`.
- `GenResult { text, reasoning_content, prompt_tokens, completion_tokens, finish_reason, logprobs, prefill_seconds, decode_seconds }`. The two timing fields let callers report a true decode rate (completion tokens / decode time) instead of dividing by total request time. A restored `PrefillState` reports `prefill_seconds = 0.0` because its cost is amortized over the choices that shared it.
- `PrefillState` + `prepare_prefill()` / `prepare_chat_prefill()` / `prepare_completion_prefill()` — one prompt forward pass captured as a `Model::snapshot()` plus final logits; `generate_inner` takes `Option<&PrefillState>` and restores it for each choice. Both endpoints prepare once when `n > 1`.
- Bounded admission: `Engine::try_admit()` reserves one of `MAX_QUEUED_REQUESTS = 16` slots (RAII `AdmissionGuard`); excess requests get a 429 instead of piling up. `Engine` owns `models: Vec<Mutex<Model>>` with an available-slot `Vec` + `Condvar`; `ModelLease` hands out one instance for the whole request. `--instances N` (default 1) creates N instances sharing the Arc model core; extra instances cost state, not weights.
- `Params` (Clone) adds `n`, `user`, `response_format`, `logprobs`, `top_logprobs` and an internal `deadline: Option<Instant>` override (never set by the API; used by tests). `parse_params_inner(body, engine, chat)` handles the different chat/completions shapes (`logprobs` bool vs integer). `ResponseFormat::JsonObject` adds a best-effort prompt hint (`apply_response_format_chat/text`); no constrained decoding.
- `generate_inner()` is the core loop; `generate()` wraps it with a no-op logprob callback. `run_chat_inner()` / `run_completion_inner()` likewise wrap `run_chat()` / `run_completion()` and thread an `on_logprob` callback through for SSE deltas. `prepare_logits()` (penalties/bias) is shared by `sample()` and `logprob_info()` so reported probabilities match the sampled distribution; `choice_seeds()` derives one RNG stream per choice for `n > 1`.
- `generate()` — core loop:
  1. `model.reset()`, `logits = model.forward(&prompt_ids)` (prefill).
  2. Each step: check the 300 s deadline → `sample()` → check stop token → decode incrementally → `on_token(chunk, is_reasoning)` → emit the token's `on_logprob` delta (after the content chunk) → `logits = model.forward(&[tok])`.
  3. Stops on EOS or matching `stop` strings; strips the matched stop string from output.
  4. `flush_utf8()` buffers incomplete multibyte chars across token boundaries (avoids char-boundary panics).
  5. When `split_reasoning=true`, tokens 248068/248069 (`<think>`/`</think>`) toggle `in_reasoning` and are never emitted; they **are still fed through the model** so state stays aligned. `OutputRouter` handles toggling and buffered text; reasoning text accumulates in `GenResult.reasoning_content`, answer text in `GenResult.text`.
- `run_chat()` / `run_completion()` — render prompt, tokenize, call `generate()`. **Chat uses `add_generation_prompt=true`** (critical: model needs `<think>\n` to start) and `split_reasoning=true`; completions use `split_reasoning=false`.
- `sample()` — repetition penalty (divides logits of already-generated tokens by `repetition_penalty` > 1, HashSet dedup), presence/frequency penalties, `logit_bias`, then temperature (≤0 = argmax), top-k, top-p nucleus sampling. The default path (no top-k, top_p=1) uses an allocation-free two-pass categorical draw; other paths use one candidate vector with partial selection. `generate()` passes completion history into `sample()`.
- Progress logging to stderr: `[load]` per 16 linears + totals in `exl3.rs::load`; `[prompt] processed/total (tok/s)` per 32-token prefill batch; `[decode] N tokens generated (tok/s)` every 16 tokens; `[done]` final summary. Constants `PROMPT_BATCH=32`, `DECODE_BATCH=16` in `server.rs`.
- `handle_chat()` / `handle_completions()` — parse JSON body, build SSE chunks or JSON responses. SSE uses `chat_chunk()` / completion chunk builders; terminates with `data: [DONE]`. With `stream_options.include_usage`, both endpoints emit a final `choices: []` chunk carrying aggregate usage. `ToolCallStreamer` incrementally strips `<tool_call>` XML from the answer stream and `tool_call_delta()` emits indexed OpenAI tool-call deltas (name + argument fragments). On a truncated generation (`finish_reason=length`) its `finish()` drops any unfinished block rather than leaking raw XML into `content`; `trailing_prefix_len()` (in `chat.rs`) is shared with the non-streaming `parse_tool_calls()` path.
- `Params` includes `enable_thinking`, penalties, `logit_bias`, stop token ids and `include_usage`. `parse_params()` validates all fields and returns 400 on bad input.
- `OutputRouter` owns UTF-8 assembly, stream toggling and stop-string tail buffering; `on_token` returns false to abort generation when the client disconnects. RoPE tables and DeltaNet head scratch are precomputed/reused for throughput.

### `src/exl3.rs` — EXL3 safetensors reader + dequantizer (2214 lines, most research-heavy)
See §5 for the full format. Key items:
- `half_to_f32`, `f32_to_half`, `half_add`, `half_fma` — hand-written fp16 emulation (round-to-nearest-even).
- `decode_3inst`, `decode_mcg`, `decode_mul1` — the three codebooks (16-bit index → fp16).
- `window()` — extracts a 16-bit codeword from the packed MSB-first bitstream with wraparound.
- `tile_perm()` — maps ring position → row-major 16×16 tile offset.
- `unpack_tile_bits()` — unpacks one tile into 256 raw fp16 codebook bits; `unpack_tile()` converts those to f32 via the `half_f32` LUT; `unpack_tile_f32()` is the fast hot-path variant that looks up the codebook's precomputed f32 table directly (bit-identical result, no per-codeword arithmetic). The f16 cache stores bits directly. `matmul_f16()` is the threaded lm_head matmul over an fp16 embedding table.
- `hadamard128()`, `hadamard_rows()`, `hadamard_cols()` — normalized Sylvester H128 butterflies. `hadamard_rows` uses a stack buffer per 128-column slice (cache-friendly) instead of strided butterflies.
- `read_safetensors()`, `load_all_shards()` — safetensors header parsing, sharding via index JSON or glob. Header arithmetic is checked and malformed metadata (dtype/shape/data_offsets) is rejected; `validate_tile_width()` bounds trellis widths before unpacking. `Shards::compact_to()` drops trellis byte ranges after the linear pass.
- `dequant_linear()` — reference full dequant pipeline for one quantized linear: unpack tiles → H_k → ×suh → H_n → ×svh → transpose to `(n, k)` → `Matrix`. **Used only by tests** as ground truth.
- `QuantLinear` — a quantized linear (trellis + suh/svh) with three storage modes set by `--weight-mode`:
  - `packed`: `matmul(x)` decodes one 16×16 tile at a time into a small scratch (1× on-disk RAM; ~1.7 tok/s / 0.6 s per token on a 12th-gen x86 host — not interactive). O(1) `window()` (64-bit funnel shift), `DECODE_*` LUTs (65536 f32 entries per codebook, avoids log2/powi per codeword), `TILE_PERM` table, and stack buffers keep it from being worse.
  - `f16`: `build_decode_cache()` decodes W_hat once into fp16 (2 bytes/weight, lossless) stored as `(n,k)`; `matmul` uses `half_f32` LUT for a fast dot. Its cache build now uses the `decode_bits` LUT too (bit-identical to `decode()`), so it is no longer ~10x slower to load than the f32 path.
  - `f32` (default): `build_decode_cache_f32()` decodes W_hat once into f32 `(n,k)`; `matmul` is a direct f32 dot. ~0.02 s for a 1024×6144 layer.
  - All modes use the derivation `y[j] = svh[j] * (H_n @ (W_hat^T @ (H_k @ (x ⊙ suh))))[j]` and split output rows across the persistent worker pool (`SharedPtr` newtype / `SendSlice`, ≥65536-element threshold). Verified equivalent to full dequant by the fresh-clone-safe `synthetic_tests::matmul_matches_dequant_*` tests (3INST, MCG, MUL1 integer + half-bitrate). The real-model `quantized_matmul_matches_f32_dequant` is `#[ignore]`d because it needs `/tmp/unstable_*.bin`.
  - `linear()` in model.rs prefers a plain `Matrix`, then `QuantLinear`, then (for tied `lm_head`) the fp16 embedding table via `matmul_f16`, skipping the largest decode and avoiding an f32 copy.
- `Exl3Model::load(path, weight_mode)` — loads config.json + all shards, detects prefix, keeps quantized linears in `linears` (skipping vision) with optional f32/f16 decode cache per `weight_mode`, loads plain tensors as flat vectors, builds plain 2-D matrices (embeddings etc.) in `weights`.
- `Exl3Model { config, prefix: String, weights: HashMap<String, Matrix>, embeddings_f16: HashMap<String, F16Matrix>, linears: HashMap<String, QuantLinear>, vectors: HashMap<String, Vec<f32>> }`. `prefix` is detected from shard tensor names (`model.` or `model.language_model.`) and reused by `Model::load`. All plain embeddings are read straight from the F16 shard into `embeddings_f16`; tied checkpoints use the same table for `lm_head` via `matmul_f16` (no f32 vocab x hidden matrix).
- `read_signs()` — reads suh/svh (F16) with legacy packed su/sv fallback.

### `src/model.rs` — Qwen3.5 decoder forward (1356 lines)
See §6 for the full math. Key items:
- `ModelConfig` — extracted from `config.json` (handles nested `text_config`).
- `ModelCore` (Arc-shared: config, prefix, weights, linears, vectors, threads, context_size, rope tables, zeroed state templates) + `Model` (one instance: `core`, `position`, `full_states`, `linear_states`). `Model::spawn()` creates another instance sharing weights but with fresh state; this is what `--instances N` uses.
- `w(name)` / `ql(name)` / `v(name)` — resolve unprefixed names by trying `{prefix}{name}`, `{prefix}{name}.weight`, bare `name`, `name.weight`; `ql` also aliases `lm_head` → the quantized head. `embedding_row(token)` looks up the fp16 table under `{prefix}embed_tokens.weight` first (or the f32 fallback), so untied multimodal checkpoints resolve correctly.
- `linear(x, name)` — universal linear: plain `Matrix` → `QuantLinear` → fp16 embedding (`matmul_f16`, tied `lm_head`). All projections (q/k/v/o, in_proj_*, out_proj, MLP gate/up/down, lm_head) go through it; only the embedding lookup bypasses it.
- `FullAttnState { k, v }` — **fp16** KV cache `(max_ctx, num_kv_heads, head_dim)` stored as `Vec<u16>` bits (halves context RAM). Writes use `f32_to_half`; the used prefix is decoded through the `half_f32` LUT into f32 scratch before attention.
- `ModelState` / `snapshot()` / `restore()` — a compact post-prefill snapshot (only the used KV prefix plus DeltaNet state) so `n > 1` choices share one prompt forward pass.
- `LinearAttnState { conv, conv_pos, state }` — conv circular buffer + recurrent state.
- Testable math helpers (extracted from the forward paths without changing the math): `split_q_gate()` (per-head `[query|gate]` interleaving), `gqa_attention()` (`AttnShape` carries nh/nkv/hd/tlen; GQA gather + scale + softmax + sigmoid gate), and `deltanet_step()` (one value-head recurrence; `DeltaNetScratch` carries kd/vd and reusable buffers). Each has a hand-computed golden test in `model::tests`.
- `rms_norm()` — zero-centered: `x * rsqrt(mean(x²)+eps) * (1+w)`.
- `rms_norm_gated()` — `norm(x) * w * silu(gate)`, applied **per value head** (groups of `linear_value_head_dim`); normalizing over the whole concatenated vector was a bug that broke linear attention.
- `forward_full_attn()`, `forward_linear_attn()`, `forward_token()`, `forward(&[u32])`.

### `src/fuzz.rs` — deterministic fuzz smoke tests (`#[cfg(test)]` only)
- Seeded LCG drives random inputs through `json::parse` (with parse→serialize→parse roundtrip), `http::read_request` (real loopback sockets), `regex::Regex::compile`/`match_at`/`pre_tokenize`, `server::parse_params` / `parse_completion_params` / `parse_messages`, `apply_chat_template`, and the tokenizer. No external crate; every `cargo test` run includes it. Any panic is a bug; parse errors are expected.

### `tests/` — manual API test scripts
Shell scripts for end-to-end checks against a running server (not `cargo test`):
- `test_api.sh` — full API smoke test (health/models + 3 greedy chat requests).
- `test_diag.sh` — raw `/v1/completions` prompts, bypassing the chat template.
- `test_sample.sh` — sampled chat generation (`temperature=0.8`, `top_p=0.9`).
- `test_prompts.sh` — probes different assistant-prefix formats via completions.
- See `tests/README.md` for usage and when to use each.

---

## 4. Data Flow (end to end)

```
HTTP POST /v1/chat/completions
  → main::route() → server::handle_chat()
  → parse JSON body → Params + messages
  → Engine::run_chat()
      → apply_chat_template(messages, true, enable_thinking)   [chat.rs]
      → Tokenizer::encode(prompt)                              [tokenizer.rs]
      → Engine::generate(prompt_ids, params, on_token)
          → Model::reset()
          → Model::forward(prompt_ids)   (prefill, one token at a time)
              → for each token: embedding lookup, N layers, final norm, lm_head → logits
          → loop: sample(logits) → on_token(chunk) → Model::forward(&[tok])
  → SSE chunks or JSON response → http::respond / Streamer
```

**Concurrency:** one thread per HTTP connection; `Engine.model` is a `Mutex<Model>`, so only one request generates at a time (serialized). This is intentional for simplicity.

---

## 5. EXL3 Format & Dequantization (research notes — do not lose)

### Container
- Standard **safetensors**: `u64 LE header_len` + JSON header + raw tensor bytes (no magic bytes, no padding).
- Sharded: `model-00001-of-0000N.safetensors` + `model.safetensors.index.json` (`weight_map` dict), or single `model.safetensors`.

### Per quantized linear `<name>`
| Tensor | Dtype | Shape | Meaning |
|--------|-------|-------|---------|
| `<name>.trellis` | I16 | `(k/16, n/16, 16*K)` | Packed tail-biting trellis |
| `<name>.suh` | F16 | `(k,)` | Input sign/scale vector |
| `<name>.svh` | F16 | `(n,)` | Output sign/scale vector |
| `<name>.bias` | F16 | `(n,)` | Optional bias |
| `<name>.mcg` / `<name>.mul1` | I32 scalar | — | Codebook marker (presence only) |
| `<name>.su` / `<name>.sv` | I16 | legacy | Packed ±1 signs (fallback) |

`K = trellis.shape[-1] / 16` (integer, or half-integer for MUL1 only). `k = 16*tiles_k`, `n = 16*tiles_n`.

### Codebook selection
- Neither `mcg` nor `mul1` → **3INST** (cb=0)
- `mcg` present → **MCG** (cb=1)
- `mul1` present → **MUL1** (cb=2)

### Codebook decode (16-bit idx → fp16 bits)
- **3INST:** `x = (idx*89226354 + 64248484) & 0xFFFFFFFF; x = (x & 0x8FFF8FFF) ^ 0x3B603B60; value = half(lo16(x)) + half(hi16(x))`
- **MCG:** `x = (idx*0xCBAC1FED) & 0xFFFFFFFF;` then same LOP3 + sum.
- **MUL1:** `x = (idx*0x83DCD12D) & 0xFFFFFFFF; bsum = byte0+byte1+byte2+byte3(x); h = 1024+bsum; value = fma(half(h), half(0x1EEE), half(0xC931))` (range ≈ ±3.45).
- The LOP3 and FMA are emulated in f32 then rounded once to fp16 (`half_add`, `half_fma`).

### Trellis packing/unpacking
- A 16×16 tile = ring of 256 positions. Position `i` advances the bitstream by `D(i)` bits: `D(i)=K` (integer) or `KA + ((0xAAAA >> (i&15)) & 1)` (half-integer, MASK=0xAAAA). `S(i) = prefix sum of D`. Ring holds `R = 256*K` bits.
- Position `i`'s 16-bit codeword = ring bits `[S(i)-16, S(i))`, MSB-first, wrapping mod R. Integer case: `S(i) = (i+1)*K`. Half case: `S(i) = (i>>4)*bpb + (i&15)*KA + ((i&15)+1)/2` with `bpb = 16*KA+8`.
- Packed: ring as 32-bit words, bit 0 at MSB of word 0, stored LE as pairs of uint16 (`word = packed[2w+1]<<16 | packed[2w]`).
- Tile element order: `tile_perm(i)` for `t=i>>3, j=i&7`: `r = (t&3)*2 + (j&1) + (8 if j&2) + ...`, `c = (t>>2) + (8 if j&4)`. Verified against reference.

### Dequant pipeline (`dequant_linear`)
1. Unpack all tiles → `W_hat` `(k, n)` in rotated basis: `W_hat[16*tk+r, 16*tn+c] = decode(window(tile_perm⁻¹...))`.
2. `W1 = H_k @ W_hat` (Hadamard on each 128-row block, scaled `1/sqrt(128)`).
3. `W2 = W1 ⊙ suh` (broadcast over columns, i.e. scale each row by suh[i]).
4. `W3 = W2 @ H_n` (Hadamard on each 128-column block).
5. `W4 = W3 ⊙ svh` (scale each column by svh[j]).
6. Transpose `(k,n) → (n,k)` and store as `Matrix` for `matmul_t`.
- **Note:** `(A ⊙ suh) @ H_n == (A @ H_n) ⊙ suh` (row scaling commutes with right-multiply), so the order of suh vs H_n is not a bug.
- Reference: `tests/testlib/trellis.py` in turboderp-org/exllamav3. Ground-truth codebook values tested in `exl3.rs::tests`.

### Tensor name conventions
- Text-only: `model.embed_tokens.weight`, `model.layers.{i}.*`, `lm_head.weight`, `model.norm.weight`.
- Multimodal (`Qwen3_5ForConditionalGeneration`): text model nested under `model.language_model.*`; **vision under `model.visual.*` (skipped)**; `lm_head.weight` still top-level.
- `Model::w()`/`v()` handles both via prefix-first-then-bare resolution.

---

## 6. Qwen3.5 Decoder Math (from HF `modeling_qwen3_5.py`)

### Config (9B)
`hidden_size=4096`, `num_layers=32`, `num_attention_heads=16`, `num_key_value_heads=4`, `head_dim=256`, `intermediate_size=12288`, `full_attention_interval=4`, `attn_output_gate=true`, `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=32`, `linear_value_head_dim=128`, `rms_norm_eps=1e-6`, `vocab_size=248320`, `rope_theta=1e7`, `partial_rotary_factor=0.25`, `tie_word_embeddings=false`.

Layer types from `config.layer_types` (or `(i+1)%4==0`). Full attention at layers 3,7,11,15,19,23,27,31.

### Per layer
`x = x + Attn(RMSNorm(x))`; `x = x + MLP(RMSNorm(x))`. No residual scaling. All linears bias-free.
Final: `x = RMSNorm(x)`; `logits = lm_head(x)`.

### RMSNorm (zero-centered, Qwen3 style)
`out = x * rsqrt(mean(x²)+eps) * (1 + weight)` — weight initialized to 0. Used for input/post/final norms and q_norm/k_norm (per head). Computed in fp32.

### Full attention (GQA)
- `q_proj: 4096→8192`. Output is **interleaved per head**: head h occupies `[h*512 .. h*512+512)`, with **query in first 256 dims and gate in last 256 dims**. (The earlier assumption "all queries then all gates" was WRONG — verified from `torch.chunk(q_proj(...).view(..., -1, head_dim*2), 2, dim=-1)`).
- `k_proj: 4096→1024`, `v_proj: 4096→1024`, `o_proj: 4096→4096`.
- q_norm/k_norm applied to each 256-dim head **before RoPE**.
- Partial RoPE: rotary dim = `int(256*0.25) = 64`; rotate-half layout on first 64 dims; theta=1e7. Text-only mrope is a no-op (all position IDs equal).
- Scale = `1/sqrt(256)`. GQA grouping: `nh/nkv = 4` q-heads per kv-head.
- `attn_output *= sigmoid(gate)` after attention, before o_proj.

### Linear attention (Gated DeltaNet)
- Separate projections: `in_proj_qkv: 4096→8192` (=2048+2048+4096, order **query, key, value**), `in_proj_z: 4096→4096`, `in_proj_b: 4096→32`, `in_proj_a: 4096→32`.
- Depthwise `conv1d`: kernel 4, left-pad 3, SiLU, on the full 8192-dim qkv before splitting. Weight shape `(conv_dim, 1, 4)` → flattened channel-major (`cw[c*4+j]`).
- Per head (fp32): `q̂ = l2norm(q)`, `k̂ = l2norm(k)` (with `+1e-6`), `q̂ /= sqrt(128)`. Repeat k heads 2× (16→32) to match v heads.
- `g = -exp(A_log) * softplus(a + dt_bias)`; `beta = sigmoid(b)`.
- State `S` shape `(32, 128, 128)`, zero-init. Per token: `S *= exp(g)`; `mem = S @ k̂`; `S += k̂ ⊗ ((v - mem)*beta)`; `o = S @ q̂` (uses UPDATED state).
- Output: `RMSNormGated(o, z, norm.weight)` = `norm(o) * w * silu(z)` (weight init ones), then `out_proj: 4096→4096`.
- MTP tensors skipped.

---

## 7. Common Tasks & How-To

### Add a new API parameter
1. Add field to `Params` in `server.rs` and its `Default`.
2. Parse it in `parse_params()` (use `Json` accessors; note `as_usize`/`as_f64`/`as_bool`).
3. Use it in `generate()` or handlers.

### Add a new endpoint
1. Add a match arm in `route()` in `main.rs`.
2. Implement handler in `server.rs` (or inline). Use `json_error()` for errors, `respond()` for JSON, `Streamer` for SSE.

### Support a new architecture
1. Extend `ModelConfig` and config parsing in `Model::load()`.
2. Add new state structs and forward logic in `model.rs`.
3. Update tensor-name resolution if prefixes differ.
4. Update `Exl3Model::load()` tensor filtering if needed.
5. Update `AGENT.md` and tests.

### Fix dequantization issues
1. Check `exl3.rs` codebook/tile functions against `tests/testlib/trellis.py` (fetch from turboderp-org/exllamav3).
2. Add/extend unit tests with reference values (see existing `exl3::tests`).
3. Verify codebook decode values against a pure-numpy reference (torch is NOT installed here).

### Fix tokenizer issues
1. Compare against `/tmp/qwen35/tokenizer.json` and the reference values in §6 of the original research (tokenizer section).
2. The regex engine tests (`test_qwen_style`) lock down the pre-tokenizer behavior.
3. Added-token protection is in `Tokenizer::encode` — keep longest-first matching.

### Debug generation quality
- Check chat template output first (`apply_chat_template` tests).
- Verify prompt tokenization (`Tokenizer::encode` on the rendered prompt) — special tokens must be single ids.
- Check stop token: `eos_id()` must return 248046.
- Verify dequantization with a tiny synthetic EXL3 model if possible.
- Sampling: `temperature <= 0` → argmax.

### Benchmark a real model
```bash
EXLLAMA_BENCH_MODEL=/path/to/model cargo test --release \
  bench_generation_throughput -- --ignored --nocapture
# optional: EXLLAMA_BENCH_THREADS=N (default: all CPUs)
#           EXLLAMA_BENCH_WEIGHT_MODE=f16|f32|packed (default: f32)
#           EXLLAMA_BENCH_COMPUTE=auto|scalar|blas (default: auto)
```
The opt-in benchmarks initialize the global threadpool and compute backend
**exactly like `main()` does** — do not remove those calls. Without them,
`threadpool::run()` executes jobs sequentially and `compute::backend()` falls
back to single-threaded sgemm, so the reported tok/s is a single-core number
that ignores `--threads`. The report splits prefill and decode time; use the
`decode tok/s` figure, not completion-tokens-over-total-time. Load time is
dominated by decode-cache construction; the f32 cache build is LUT-backed and
fast, and f16 now uses a matching `decode_bits` LUT. Measured on a 12th-gen x86
host (10 threads, 4 bpw 0.8B, 159 linears): f32 loads in ~10.6 s and decodes at
~11.9 tok/s; f16 ~10.3 s / ~8.6 tok/s *after* the `decode_bits` LUT fix (54.6 s
before it); packed ~6.8 s / ~1.6 tok/s. So f32 is the fastest decode mode on
modern CPUs — do not assume f16 is "near-f32". Note the auto backend picker weights the
GEMV probe 4x over prefill (`compute::pick_by_score`): decode repeats per token
and prefill runs once, so a large GEMV win must not be masked by a modest
prefill win. With f16/packed weights the choice barely affects decode (those
paths use hand-written threaded kernels, not `compute::matmul_t`), but with f32
it matters — use `EXLLAMA_BENCH_COMPUTE` to A/B it on a given host.

---

## 8. Known Limitations & Gotchas

- **Memory vs speed:** `--weight-mode` trades RAM for tokens/s, but the ranking is host-dependent. Measured on a 12th-gen x86 host (10 threads, 4 bpw 0.8B): `f32` (default) loads in ~10.6 s and decodes at **~11.9 tok/s** — the fastest decode mode there; `f16` (~2× on-disk) loads in ~10.3 s after the `decode_bits` LUT fix (54.6 s before it — its cache build used per-codeword arithmetic while f32 used a LUT) and decodes at ~8.6 tok/s; `packed` (~1× on-disk) loads in ~6.8 s but decodes at ~1.6 tok/s — not interactive. On CPU, SIMD on `dot()` would not help the packed path; its bottleneck is codebook decode, not the MACs. Tied-embedding models skip the lm_head decode (plain embedding matrix used directly). Peak RAM during load includes raw shard bytes plus decoded caches (in `f32` mode, several times on-disk); the old "≈ 2× on-disk" claim was inaccurate. Plain 2-D tensors are decoded once (previously stored in both `vectors` and `weights`), trellis byte ranges are compacted away after the linear pass, and untied embedding tables are stored as fp16 with per-row decode. Tied models no longer keep a ~1 GB f32 embedding: all plain embeddings are fp16 and tied `lm_head` uses the threaded `matmul_f16`. The full-attention KV cache is fp16 (2 bytes/value, LUT-decoded per step), halving context RAM. The packed decode hot path now uses `unpack_tile_f32()` (direct codebook f32 LUT) instead of per-codeword arithmetic + fp16 conversion; true SIMD gather remains unavailable on stable std-only Rust (no portable `simd` API), but the arithmetic bottleneck is gone.
- **GPU fidelity limits:** the device-resident path (feature `gpu`, backend `gpu`) keeps all weights/activations/state on device, but uses fp16 weights and fp16 activation casts around each GEMM (the CPU path uses f32), so outputs can differ slightly. On the GPU backend sampling now also runs on device (logit penalties, temperature, top-k/top-p, greedy, logprobs); only a token id, its logprob and optional top pairs cross back per step — see §3. Real-GPU correctness/throughput is not exercised on CI; only device-op unit tests on candle's CPU device run there. The host keeps the packed trellis (~1x disk) as a fallback source; no large host decode cache is built on GPU builds. The whole device forward is now exercised end-to-end on candle's CPU device with a tiny multi-head model (`gpu::tests::device_forward_runs_end_to_end_on_cpu`), which caught eight shape/indexing bugs that individual op tests missed (the fifth was the dropped partial-RoPE tail; two more were per-head RMSNorm reducing across heads and an untied quantized `lm_head` being shadowed by the tied-embedding fallback, see §3); a real-GPU A/B is still advisable.
- **Concurrency:** `n > 1` choices share one prefill via `PrefillState` (see §3). Independent requests can run in parallel with `--instances N` (default 1): N model instances share one Arc model core (weights/config) and only duplicate KV/recurrent state, so memory scales with instances' state rather than full weights. Bounded admission returns 429 past 16 queued requests. Matmuls are **not** batched across requests (that needs batch-aware attention/DeltaNet), and `threadpool::RUN_LOCK` still serializes matmul dispatch globally — instances overlap non-matmul work (tokenize, sampling, I/O, state handling) rather than two GEMVs at once.
- **API fidelity limits:** `response_format: json_object` only adds a prompt hint — there is no grammar/constrained decoding or JSON validation. Streaming tool calls now produce indexed `tool_calls` deltas (name + argument fragments) with the XML withheld from content, but tool-call arguments are still assembled with the simple `<parameter=...>` parser, not full JSON streaming. That parser tolerates a missing `>` in the parameter header (name ends at a newline) and drops incomplete blocks from content; deeply malformed values still fall back to strings. Chat `top_logprobs` is capped at 20 (completions accept up to 20 as an extension; OpenAI allows 5). Logprobs exclude reasoning delimiters and stop tokens, so their count matches `completion_tokens`, not every token sampled internally; `<think>`/`</think>` also consume a forward step but are not counted in `completion_tokens`.
- **No full Jinja template engine:** chat template covers text-only common cases; tool-call arguments and advanced Jinja features (e.g. `tojson` of tool definitions) are not implemented.
- **No Unicode NFC normalization:** tokenizer assumes input is already NFC (chat content typically is).
- **No vision inference:** vision tensors are dequantized-skipped and ignored.
- **Rust Send quirk:** see `SharedPtr` note in §3 (tensor.rs). Do not use raw pointers in scoped threads.
- **Bitrate detection gotcha (fixed):** `k_half` must be derived with integer math (`words_per_tile % 16 != 0`). A previous float check used `!k_float.fract().is_normal()` which is ALWAYS true for integer bitrates because `0.0f32.is_normal()` is false — this broke every integer-bpw model (e.g. 8.00 bpw) with a bogus "half bitrate requires mul1 codebook" error. See `bitrate_from_tile_width()` in exl3.rs.
- **tokenizer.json format variants (fixed):** newer `tokenizers` versions omit `model.byte_encoder` (derive with `build_byte_level_map()`), store merges as space-separated strings (`"Ġ Ġ"`) instead of 2-element arrays, rename the sequence key `pre_tokenizers` → `pretokenizers`, wrap the regex as `{"Regex": "..."}` instead of `{"String": "..."}`, and serialize the Split behavior enum as `"Isolated"` (capital I) instead of `"isolated"`. The UnstableLlama EXL3 repos use these newer forms; `Tokenizer::load()` now handles both old and new spellings. The behavior check is case-insensitive (`eq_ignore_ascii_case`) — a strict lowercase check caused a v0.4.0 startup regression. Do not revert these compatibility paths.
- **Prefix must come from the shards (fixed in v0.4.2):** `Exl3Model::load` detects `model.` vs `model.language_model.` from the shard tensor names and now stores it as `Exl3Model.prefix`; `Model::load` uses it verbatim. Re-detecting it from `weights` broke untied multimodal checkpoints because their embedding lives in `embeddings_f16`, causing "missing embedding tensor `embed_tokens.weight`" on the first request. Never re-derive the prefix from loaded tensors.
- **EXL3 tensor naming / dtypes (fixed):** quantized linears in the UnstableLlama repos use base names **without** `.weight` (e.g. `lm_head.trellis` → matrix key `lm_head`, `...mlp.gate_proj.trellis` → key `...gate_proj`). `Model::w()` now resolves both with and without `.weight`, with an `lm_head` alias and tied-embedding fallback to `embed_tokens.weight`. Many tensors (norms, conv1d, dt_bias) are **BF16**, which `read_f32()` now converts (top 16 bits of f32). Non-quantized 2-D linears (`in_proj_a`/`in_proj_b`, shape `[16, hidden]`) are now loaded as `Matrix` entries — previously only embed/lm_head were, leaving `a`/`b` empty (g=beta=0) and breaking the Gated DeltaNet recurrence. The conv circular buffer holds `kernel-1` history slots (not `kernel`). Without these fixes generation degenerated into token loops.
- **README claims CI covers fmt/clippy** — keep those checks passing (`cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`).
- **Reference files:** `/tmp/qwen35/tokenizer.json` and `/tmp/unstable_tok.json` are used by real-tokenizer tests; `/tmp/unstable_prefix.bin` + `/tmp/unstable_layer0.bin` by real-dequant tests; `/tmp/exl3src/` has exllamav3 source; `/tmp/trellis_ref.py` is the trellis reference. These are test-time-only and may not exist in a fresh clone. All `/tmp`-dependent tests are explicitly `#[ignore]`d with a reason; run them with `cargo test -- --ignored` when the fixtures exist. The committed synthetic EXL3 fixtures and decoder golden tests always run.

---

## 9. Tests Quick Map

| Test file/module | What it validates |
|------------------|-------------------|
| `json::tests` | JSON parse/serialize (if present) |
| `regex::tests::test_qwen_style` | Pre-tokenizer regex against ground-truth word lists |
| `tokenizer::tests` | Encode + added-token protection vs real Qwen3.5 tokenizer |
| `chat::tests` | Chat template rendering (4 cases), tool-call parsing incl. malformed `<parameter=` headers and incomplete blocks |
| `exl3::tests::codebook_decode_matches_reference` | All 3 codebooks vs numpy reference |
| `exl3::tests::half_roundtrip` | fp16 conversion correctness |
| `exl3::tests::tile_perm_matches_reference` | Tile permutation vs reference |
| `exl3::synthetic_tests::matmul_matches_dequant_*` | Fresh-clone-safe equivalence of all three `--weight-mode` paths vs full dequant, for 3INST / MCG / MUL1 (integer + half-bitrate); no `/tmp` fixtures |
| `exl3::real_model_tests` | Real layer-0 dequant + matmul checks; `#[ignore]`d (require `/tmp/unstable_*.bin`) |
| `model::tests` | Golden tests for RMSNorm, per-head RMSNormGated, partial RoPE, q_proj interleaving, GQA grouping/gate, DeltaNet recurrence, plus fp16 embedding resolution |
| `server::tests` | Sampling equivalence, logprobs vs log-softmax, parameter validation, `n` choices, SSE framing/usage/logprob ordering, tool-call streaming deltas (incl. malformed headers and truncation), deadline, HTTP responses |
| `http::tests` | Bounded request line and normal request parsing through the bounded reader |
| `threadpool::tests` | Concurrent `run()` calls complete without deadlock |
| `tensor::tests` | `dot()` rejects length mismatches |
| `regex::tests::unbounded_quantifier_is_bounded` | Quantifier repeat/work caps on long and nested-pattern inputs |
| `server::tests::prefill_reuse_matches_fresh_prefill` | Post-prefill snapshot/restore produces the same output as a fresh prefill |
| `server::tests::admission_bounds_queued_requests` | Bounded request admission slots |
| `model::tests::kv_cache_stores_fp16_bits` | fp16 KV cache layout and LUT decode |
| `compute::tests::gpu_cache_fingerprint_detects_recycled_buffer` | GPU cache key disambiguates recycled buffers (feature `gpu`) |
| `gpu::tests` (feature `gpu`) | Hadamard matrix vs CPU butterfly + `folded_q_matmul_matches_unfolded`, 1-D and per-head 2-D RMSNorm, DeltaNet recurrence (single-head golden + `deltanet_all_matches_per_head`), `gqa_batched_matches_per_head`, partial-RoPE tail preservation, device sampling vs host (`device_sample_matches_host_greedy_and_softmax`, `device_sample_matches_host_top_k_and_top_p`, `device_logprobs_match_host`), snapshot logits restore, forward-then-sample end-to-end, f16 GEMM and bit round-trip, synthetic `q_matmul`, quantized-`lm_head`-over-embedding resolution, and a full multi-head `forward` on candle's CPU device (catches shape/indexing bugs without a GPU) |
| `model::tests::spawned_instance_shares_weights_but_not_state` | Arc core sharing + independent per-instance state |
| `model::tests::f16_embedding_matmul_matches_f32` / `tied_f16_embedding_supports_forward` | Threaded f16 lm_head matmul and tied embedding forward |
| `server::tests::multiple_instances_serve_concurrent_requests` | Two model instances serving parallel requests |
| `exl3::tests` unpack-tile equivalence | `unpack_tile_f32` is bit-identical to unpack + fp16 convert |
| `tests::cli_parses_instance_count` | `--instances` parsing and bounds |
| `fuzz::tests` | Seeded random inputs against JSON/HTTP/regex/chat/tokenizer/API parsers (no panics) |
| `model::tests::config_validation_rejects_shapes_that_crash_the_cpu_forward` | Shared head-divisibility / RoPE-range validation (CPU no longer panics where GPU rejects) |
| `exl3::tests::f32_to_half_carries_into_smallest_normal` | fp16 subnormal→normal rounding boundary |
| `server::tests::sse_write_failure_is_latched_for_generation_callbacks` | Chat SSE disconnect propagation to `on_token` |
| `server::tests::router_applies_stop_strings_inside_reasoning` / `router_strips_longest_overlapping_stop_regardless_of_order` | Stop-string semantics in reasoning and with overlapping stops |
| `server::tests::prefill_honours_request_deadline` / `generation_deadline_stops_before_first_token` / `generation_deadline_stops_during_decode` | One request-wide deadline covers prefill (n=1 and shared) and decode |
| `http::tests::transfer_encoding_is_rejected` | Ambiguous/unsupported request framing |
| `tokenizer::tests::bpe_uses_merge_rank_lookup` | BPE merge lookup with a reusable key buffer (no per-pair allocations) |
| `gpu::tests::device_forward_matches_cpu_forward_on_same_weights` | Whole-forward CPU↔GPU logit parity (feature `gpu`) |
| `fused::tests::fused_phase2b_matches_fallback_on_cpu` | Gated norm / DeltaNet gates / softmax / sigmoid gate vs candle-op fallbacks |
| `fused::cuda_*_match*` (5 tests) | Real-GPU numeric parity for all fused PTX kernels — RMSNorm/RoPE, phase 2b, q/k norm, KV row write, conv step (`gpu-cuda`, serial `--test-threads=1`) |

Run `cargo test` — 131 pass, 11 ignored (161 with `--features gpu`, including 4 CPU `fused::tests` and 4 packed-device-math tests; 167 with `--features gpu-cuda`, adding the real-GPU `cuda_packed_dot_matches_cpu_reference` check). The ignored set
is 6 real-model / `/tmp`-fixture tests (now explicitly marked with reasons) plus
5 benchmarks (including `bench_n_choices_prefill_reuse`). On a fresh clone the dequantization/matmul guarantee is still
verified by the committed synthetic fixtures.

---

## 10. Git / CI State

- Branch `main`, latest release tag `v0.7.1`, placeholder author identity. Crate version in `Cargo.toml` matches the tag. `v0.7.1` is the Priority 0 measurement/optimization release: the packed dequant PTX uses a power-of-two ring mask and `shf.r.wrap.b32` funnel shift; `input_hadamard` (`x⊙suh` + H128 butterfly) and `output_hadamard` (H128 + `svh`) fuse the sign/H128 chain into two launches with exact CPU fallbacks; an opt-in INT8 `dp4a` path (`EXLLAMA_PACKED_DP4A=1`, on-the-fly codebook decode, no stored requantized weights) is included but measured **slower** (8.3 vs 8.4 tok/s) and changes greedy output, so it is not recommended. A/B on the RTX 5060 Ti: packed 7.2 → 8.4 tok/s (8 bpw) and 8.3 → 8.5 (4 bpw), f16 ~60; quantized-linear VRAM 3586 → 1676 / 1022 MiB. The 40 tok/s target was **not met** — the scalar trellis decode is the ceiling; a warp-parallel split was reverted because its device reductions were racy. The CPU f32 reference agrees with packed (not f16) at 4 bpw. `IMPROVEMENT_PLAN.md` has the full numbers and recommendation (f16 for speed, packed for VRAM). Tests: 131 / 162 (`gpu`) / 6 real-GPU `cuda_*` serial. `v0.7.0` adds the opt-in `--gpu-weight-mode packed` device path (`GpuWeights::build_packed`, `QuantWeight::Packed`, `exl3::packed_upload`/`codebook_lut_f32`, and the `exl3_packed_dot` PTX kernel in `fused.rs` with an exact CPU fallback): the EXL3 trellis is uploaded verbatim and decoded on device with no requantization or full-size intermediates, cutting quantized-linear VRAM from ~3.5 GiB to ~1.6 GiB on the 2B 8 bpw model at ~7 vs ~60 tok/s (speed work is Priority 0 in `IMPROVEMENT_PLAN.md`). Tests: 131 / 161 (`gpu`) / 167 (`gpu-cuda`). `v0.6.0` (commits `832c27c`, `efb847d`) is the GPU launch-count reduction release: host-side `suh`/`svh` baking into the folded weight, precomputed `1 + w` norm vectors, sampling fast paths, and fused RMSNorm/RoPE, gated norm, DeltaNet gates + q/k norm, GQA softmax + sigmoid gate, depthwise conv1d + SiLU, plus in-place KV/conv row writes in `src/fused.rs` (CPU fallback + CUDA PTX, `EXLLAMA_DISABLE_FUSED=1` to A/B). Real-GPU A/B on an RTX 5060 Ti with the Qwen3.8 2B EXL3 model (`qwen3.8_2` directory, `hidden_size=2048`, 24 layers, `model_type=qwen3_5`), all with `EXLLAMA_BENCH_COMPUTE=gpu` (decode tok/s, fused vs `EXLLAMA_DISABLE_FUSED=1`): Phase 2a **35.48 vs 30.10 (+18%)**; Phase 2b **40.65 vs 28.73 (+41%)**; DeltaNet q/k fusion **46.17 vs 26.34 (+75%)**; in-place KV writes **49.01 vs 27.22 (+80%)**; conv1d fusion **57.37 vs 25.98 (+121%)**. Run-to-run variance is ~±5 tok/s. All five `cuda_*_match*` tests pass on device. `cargo test`: 130 passed; `--features gpu`: 156 passed. Build note (Windows): invoke cargo via the real toolchain binary in `.rustup\toolchains\...\bin` (the `.cargo\bin` symlinks fail under SSH logon), put MSVC `bin\Hostx64\x64` ahead of `Hostx86\x86` in `PATH` for nvcc, and use CUDA 12.9 headers with the installed nvcc (12.9 has no `bin\`). `v0.5.8` adds shared load-time validation of quantized-linear dimensions (non-128 multiples are now rejected instead of silently leaving Hadamard tail blocks untransformed), one tile-unpack/sign-vector path with exact sign-vector lengths, tiled Hadamard rows, a reusable per-instance matmul scratch, zero-allocation single-thread matmuls, MSRV 1.88 and a clippy-clean tree (commits `24fa9d4`, `45a4916`; `cargo test`: 130 passed, `--features gpu`: 152 passed). `v0.5.7` fixes the first-decode OOM: candle 0.11's `Tensor::cumsum` materializes a `(vocab, vocab)` triangular matrix (~246 GB); the sampler now does the inverse-CDF walk on the host. `v0.5.6` contains the host-side Hadamard fold (load-time OOM), bounded `top_k`, GPU context-size warning, and device-memory diagnostics; see §3. `v0.5.5` is a hotfix for a CUDA-only `matmul` contiguity failure in the batched GQA path (see §3); `v0.5.4` fixed the `fold_hadamards()` dimension bug that made v0.5.3 fail to load any model with non-square quantized linears. `v0.5.3` is a GPU performance release: batched DeltaNet/full attention, load-time Hadamard folding (one GEMV per quantized linear), and on-device sampling (no vocabulary-sized logits copy per token) — see §3, with numeric equivalence tests against the old paths (`cargo test --features gpu`: 134 passed). `v0.5.2` is a GPU correctness hotfix: partial RoPE dropped each head's unrotated tail, producing `Shape mismatch, got buffer of size 128 which is compatible with shape [1, 512]` on real models (now fixed, with a numeric regression test and a partial-rotary end-to-end fixture). A follow-up audit fixed two more silently-wrong bugs — 2-D q/k RMSNorm pooled statistics across heads, and an untied quantized `lm_head` shadowed by the tied-embedding fallback — and added named shape/dimension diagnostics throughout the device forward (commits `6a80d45`, `be48234`; see §3/§8/§9). `v0.5.1` is a GPU hotfix: the device-resident forward failed on a real model with `shape mismatch in mul, lhs: [16,128], rhs: [1,128]`; five shape/indexing bugs were fixed and an end-to-end CPU-device forward test added. `v0.5.0` adds the device-resident GPU forward (feature `gpu`: weights/activations/state stay on device, fp16 device tensors, no per-layer transfers) and the real-client tool-call fixes (malformed `<parameter=` headers, truncated-call XML withheld from content). Phase 0–3 improvements plus two hotfixes are committed and pushed to `origin/main`: `d0c5cb4` (hardening), `30d4f55` (chat/API fidelity), `12baa8b` (memory/threading), `21a07a0` (API completeness + fuzz harness), `174d21a` (v0.4.1: case-insensitive Split behavior parsing), and the v0.4.2 hotfix (embedding prefix). `v0.4.0` was the first tagged release of the Phase 0–3 work; `v0.4.1` fixed the startup regression where newer `tokenizers` serialization of `Split.behavior` as `"Isolated"` was rejected; `v0.4.2` fixed "missing embedding tensor" on the first request for untied multimodal checkpoints. `v0.4.3` adds the deferred Phase 2 work (multi-instance decoding, f16 tied embeddings, direct-LUT packed decode), a representative real-model benchmark harness with split prefill/decode timing, a 4x-weighted CPU backend picker, a LUT-backed f16 cache build, and GPU/CUDA builds in the main Release workflow. Tagging `v*` triggers the main Release workflow (6 targets, including CUDA); GPU binaries are built automatically with every release — there is no separate manual dispatch step anymore.
- `.github/workflows/ci.yml`: build + `cargo test` on push/PR for `ubuntu-latest` (linux x86_64), `ubuntu-24.04-arm` (linux arm64), `macos-latest` (arm64, with `gpu-metal`), `windows-latest` (x86_64). fmt check; clippy `-D warnings` (Linux); uploads artifacts. The macOS build compiles the GPU backend; GPU runtime correctness is not exercised on CI runners.
- `.gitattributes`: `tests/golden/*.txt` is forced to `text eol=lf`; Windows runners check out with `core.autocrlf=true`, which used to turn those files into CRLF and fail `chat::tests::prompts_match_committed_golden_files` (generated prompts always use `\n`). The test also normalizes CRLF on both sides so it is robust to any checkout configuration.
- `.github/workflows/release.yml`: on `v*` tags, builds all **6** targets and publishes one GitHub Release with all archives: the 4 native legs (macOS with `gpu-metal`) plus Linux x86_64 and Windows x86_64 with `gpu-cuda`. CUDA legs install the toolkit via `Jimver/cuda-toolkit@v0.2.21` (12.6.0) and are pinned to `ubuntu-24.04` / `windows-2022` because `ubuntu-latest` (26.04) lacks CUDA 12.6 packages and `windows-latest` ships MSVC 14.52 (CUDA 12.6 needs 2022's MSVC). `CUDA_COMPUTE_CAP=80` is set for PTX generation. `if: matrix.cuda` gates every CUDA setup step; a GPU build failure blocks the release so GPU binaries are never silently missing.
- `.github/workflows/release-gpu.yml`: **removed** — its CUDA legs were merged into `release.yml`, so GPU binaries are no longer a separate manual `workflow_dispatch` step. If a backfill workflow for old tags is needed later, reintroduce it as a thin reusable-workflow caller rather than duplicating the CUDA setup.
- `origin` is configured (`https://github.com/fakezeta/exllama.rs.git`); to push: `git push -u origin main`.
