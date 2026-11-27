# Improvement Plan — exllama.rs

This is the **source of truth for open priorities**. Read it before starting
non-trivial work; keep it updated when priorities change. Architecture, format
research and the *why* behind the code live in `AGENTS.md`. `TODO.md` tracks
individual checked-off tasks; this file groups the remaining work into ordered
priorities with acceptance criteria.

> **Status (2026-11-27):** v0.7.1 — Priority 0 measurement/optimization work
> committed and tagged. Commit only on explicit user approval.

## Current state

- Branch `main`, latest tag `v0.7.0`. The packed-GPU feature described in
  Priority 0 below is included in this release.
- Tests: `cargo test` **131 passed / 11 ignored**; `cargo test --features gpu`
  **162 / 11** (one new butterfly test); on the RTX 5060 Ti host with
  `gpu-cuda` all **6** real-GPU `cuda_*` tests pass serially
  (`-- --test-threads=1`), in both exact and opt-in dp4a modes.
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

### Priority 0 — Make packed GPU mode fast enough to be useful — MEASURED, TARGET NOT MET

*Status (2026-11-27, uncommitted working tree):* a repeatable A/B harness
(`ab.ps1` on the GPU host) now records decode tok/s, quantized-linear MiB,
`nvidia-smi` VRAM, load time and a greedy-output SHA for both models. All
measurements below are 3-run medians on the RTX 5060 Ti with no competing
process (a stray `ninfer-serve.exe` initially held ~12 GB and skewed the first
baseline).

| checkpoint | f16 | packed before | packed now | q-linear VRAM (f16 → packed) |
|---|---|---|---|---|
| qwen3.8_2 (2B, 8 bpw, MCG) | 59.6–61.3 tok/s | 7.2 | **8.4** | 3586 → 1676 MiB |
| qwen3.5_2_4bpw (2B, 4 bpw, MCG) | 52.1–60.9 tok/s | 8.3 | **8.5** | 3586 → 1022 MiB |

Load time ~1.3–1.7 s in both modes; greedy output SHA is unchanged from the
v0.7.0 baseline for both checkpoints (exact path).

Work completed:
1. Exact PTX micro-optimizations in `exl3_packed_single`: power-of-two ring
   mask instead of `rem.s32`, `shf.r.wrap.b32` funnel shift instead of a 64-bit
   shift/mask/roundtrip, non-coherent `.nc` loads for trellis/LUT.
2. Hadamards + signs fused into two device kernels: `input_hadamard`
   (`x ⊙ suh` + H128 butterfly) and `output_hadamard` (H128 + `svh`), replacing
   `mul → matmul → packed_dot → matmul → mul`. Exact CPU fallbacks and numeric
   tests added (`butterfly_h128_matches_exl3_reference`).
3. An opt-in INT8 `dp4a` variant (`EXLLAMA_PACKED_DP4A=1`): on-the-fly codebook
   decode to int8 + `dp4a` dot products, no stored requantized weights. It
   passes its numeric test with a wider tolerance but is **slower**
   (8.3/8.5 tok/s) and changes the greedy output, so it is not recommended.
   Tensor cores are still not useful at decode batch 1.

Ceiling and reasons: the remaining per-token cost is the scalar trellis decode
itself (inverse `tile_perm`, funnel shift, LUT gather, ~2·10⁹ positions/token
for 8 bpw). A k-split warp-parallel kernel was prototyped and reached
~9.9/12.1 tok/s, but its device reductions were not reliably correct on the
RTX 5060 Ti driver (racy shared-memory/shuffle reductions), so it was reverted
in favour of the verified single-thread kernel. Reaching 40 tok/s needs a
different decode algorithm (e.g. a correct warp-parallel reduction or
exllamav3-style packed trellis with a proven GPU kernel), not another launch
fusion.

Quality validation: on the 4 bpw checkpoint the CPU f32 reference and `packed`
produce the *same* 32-token greedy output, while `f16` diverges at token 3 —
i.e. packed is the more accurate mode; the divergence is f16 folded-weight
rounding, not a packed-mode bug.

**Recommendation:** use `f16` when latency matters (~60 tok/s) and `packed`
when VRAM is the binding constraint (about half the quantized-linear memory).
`packed` remains interactive only in the sense of ~8 tok/s, not 40 tok/s.

*Acceptance:* A/B numbers documented ✅; 40 tok/s not met — measured ceiling
with reasons above; packed works end-to-end on the 4 bpw model ✅; fmt, clippy
and all tests green ✅ (162 passed / 11 ignored with `--features gpu`; 6/6
real-GPU `cuda_*` tests pass serially in both exact and dp4a modes).

*Remaining work if Priority 0 is reopened:* fix the warp-parallel dequant
reduction deterministically (a two-kernel partial-write + reduction was also
attempted), then revisit dp4a with a cheaper int8 pack path. 6/2 bpw and 27B
checkpoints were not available on the host.

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
