# Improvement Plan — exllama.rs

This is the **source of truth for open priorities**. Read it before starting
non-trivial work; keep it updated when priorities change. Architecture, format
research and the *why* behind the code live in `AGENTS.md`. `TODO.md` tracks
individual checked-off tasks; this file groups the remaining work into ordered
priorities with acceptance criteria.

> **Status (2026-11-27):** v0.7.0 — the opt-in packed-GPU mode is committed and
> tagged. Commit only on explicit user approval.

## Current state

- Branch `main`, latest tag `v0.7.0`. The packed-GPU feature described in
  Priority 0 below is included in this release.
- Tests: `cargo test` **131 passed / 11 ignored**; `cargo test --features gpu`
  **161 / 11**; on the RTX 5060 Ti host with `gpu-cuda`
  **167 / 11** (real-GPU tests run with `-- --test-threads=1`).
  `cargo fmt --all -- --check` and
  `cargo clippy --all-targets --features gpu-cuda -- -D warnings` are clean.
- GPU host: `192.168.178.46`, user `fakez`, Windows, RTX 5060 Ti 16 GB (sm_120),
  CUDA 13 toolkit with `ptxas`. Use the real toolchain binaries:
  `C:\Users\fakez\.rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\cargo.exe`
  with `RUSTC` set to the matching `rustc.exe`; put MSVC `bin\Hostx64\x64`
  ahead of `Hostx86\x86` in `PATH`. Test model:
  `D:\Apps\AI\llama-swap\exllama.rs\qwen3.8_2` (2B, 8 bpw, MCG, tied embeddings).
  Scratch build: `C:\Users\fakez\exllama-ab`.

## Guiding principles

- Std-only core; `matrixmultiply` is the only required crate, `candle-core`/`half`
  opt-in. No general-purpose crates (serde, tokio, serde_json, …) without approval.
- Preserve the hand-written dequantizer and the `AGENTS.md` research notes.
- Fresh-clone-safe tests: no `/tmp` fixtures or model downloads required.
- Keep fmt and clippy green on every change.
- Use the weights on disk **as-is**; do not requantize EXL3 into INT8/FP8/FP4
  or GGML formats without explicit user approval.

---

## Done (condensed — details in `AGENTS.md` §10 and the git history)

- **Phase 0–2 (hardening, API fidelity, performance):** in-memory dequant
  equivalence fixtures, decoder golden tests, bounded HTTP request line,
  serialized threadpool `run()`, checked `dot()`, single-iterator CLI, SSE
  logprob ordering / usage / tool-call deltas, request deadline, bounded regex
  quantifiers, shared prefill for `n>1`, `--instances N`, fp16 KV cache and
  embeddings, GPU cache fingerprints.
- **Device-resident GPU forward** (`src/gpu.rs`): all weights/activations/state
  stay on device; on-device sampling; fixed shape/indexing bugs, partial-RoPE
  tail, per-head RMSNorm, untied `lm_head` shadowing; host-side Hadamard fold
  to avoid vocab-sized OOM; `cumsum` OOM avoided by host inverse-CDF.
- **v0.6.0 launch-count reduction:** host-baked Hadamards/signs, precomputed
  `1+w` norms, fused pointwise kernels in `src/fused.rs`, in-place KV/conv
  writes (measured +18%…+121% on an RTX 5060 Ti).
- **Packed GPU mode (current working tree):** `--gpu-weight-mode f16|packed`
  (default `f16`). `packed` uploads the EXL3 trellis verbatim as U32 words with
  `suh`/`svh` and exact codebook decode LUTs; a new PTX kernel
  `exl3_packed_dot` decodes 16×16 tiles during the forward, with candle ops for
  input/output H128 and signs. No requantization, no full-size device
  intermediates. Supports 3INST/MCG/MUL1, integer and half bitrates.
  Measured on the 2B 8 bpw model: quantized-linear VRAM **1,676 MiB vs
  3,586 MiB** for f16; decode **~7.2 vs ~59.6 tok/s**; identical 32-token
  greedy output. Speed optimization was explicitly deferred by the user.

---

## Open priorities (ordered)

### Priority 0 — Make packed GPU mode fast enough to be useful
*Owner: next agent. This is the highest priority; functionality is already
proven correct.*

Context: packed mode fits ~2× more weights in VRAM but is currently ~8× slower
than f16 because the per-token path streams and decodes the whole trellis with
one thread per output and several extra kernels for the Hadamards/signs.

Work, in order:

1. **Measure first.** Build a repeatable A/B harness on the host (existing
   `[decode]` log lines + `nvidia-smi`) for `--gpu-weight-mode packed` at 8/6/4
   bpw. Record tok/s, peak VRAM, load time, and output parity against `f16`
   for the same prompt. Target: **>= 40 tok/s** on the 2B model, 16 GB card.
2. **Fuse the Hadamards and scales into the dequant kernel** so one launch per
   linear replaces the current `mul → H128 → packed_dot → H128 → mul` chain.
   Keep the exact CPU fallback and numeric parity tests.
3. **Then consider a `dp4a` variant** (exllamav3-style fused codebook decode
   into INT8 dot products) if step 2 cannot reach the target. Tensor cores
   (FP8/FP4) are *not* useful at decode batch 1; do not pursue them for decode.
4. **Validate large/low-bitrate models** (6/4/2 bpw, then a 27B checkpoint if
   available) on 16 GB: VRAM, quality (greedy output/logit comparison), and
   context-size limits. Document the recommended `f16`/`packed` choice.

*Acceptance:* documented A/B numbers; >= 40 tok/s target met or an explicit
measured ceiling with reasons; packed mode end-to-end on at least one 4 bpw
model; fmt/clippy/tests still green.

### Priority 1 — Batched / continuous inference across requests
`--instances N` overlaps non-matmul work but matmuls are not batched across
requests and `threadpool::RUN_LOCK` serializes dispatch. Needs batch-aware
attention and Gated DeltaNet, not just a bigger GEMM.

*Acceptance:* a two-request benchmark shows overlapping decode without state
cross-talk; `multiple_instances_serve_concurrent_requests` still passes.

### Priority 2 — GPU coverage in CI
Real-GPU correctness and throughput are not exercised by CI (only candle CPU
device tests). Keep `cuda_*` tests serial (per-device module cache) and add a
packed-mode opt-in benchmark so regressions are measurable on a GPU host.

### Priority 3 — Feature exploration (optional, only after 0–2)
- Vision tower inference for `Qwen3_5ForConditionalGeneration`.
- Other EXL3 architectures behind a generic config/weight abstraction.
- Unicode NFC normalization; read `chat_template` from `tokenizer.json`.
- Dockerfile, crates.io publish, Anthropic `/v1/messages` endpoint.
- Structured logging (`EXLLAMA_LOG`) and dead-code cleanup (`multi_step_tool`).

---

## Suggested execution order

Work Priority 0 first (measure → fuse → `dp4a` if needed → 27B/low-bitrate
validation), then 1, 2, 3. Update this file with results and re-prioritize after
each measurement. No commit until the user explicitly approves.
