# Improvement Plan — exllama.rs

This document is the reviewed, prioritized plan for hardening and extending the
server. It is derived from a full read of the four markdown docs (`README.md`,
`TODO.md`, `tests/README.md`, `AGENTS.md`) and a code review of every `src/`
module. It complements `TODO.md`: that file tracks individual checked-off tasks,
while this file groups the *open* work into executable phases with acceptance
criteria.

> **Status:** Phases 0–2 are **implemented and verified** (see the completion
> notes below); Phase 3 remains a plan. Commit only on explicit approval.

## Baseline (at time of review)

- Branch `main`, tag `v0.4.2`, working tree clean at review time.
- At review: `cargo test` **84 passed, 4 ignored**. After Phase 0:
  **88 passed, 10 ignored**. After Phase 1: **99 passed, 10 ignored**. After
  Phase 2: **102 passed, 11 ignored**; after the deferred Phase 2 work:
  **115 passed, 11 ignored** (6 real-model / `/tmp` fixtures explicitly ignored
  with reasons, 5 benchmarks). `cargo fmt --all -- --check` and
  `cargo clippy --all-targets -- -D warnings` remain clean; the GPU feature
  also compiles and its cache-key test passes.
- The **forward math is correct**. `src/model.rs` matches both `AGENTS.md` §6
  and the upstream HuggingFace `modeling_qwen3_5.py` / `modeling_qwen3_next.py`
  reference on every subtlety checked: q_proj `[query|gate]` per-head
  interleaving, per-head q/k norm before RoPE, partial rotate-half RoPE,
  GQA grouping, gate sigmoid, depthwise conv layout, head-local L2 norm, the
  Gated DeltaNet recurrence, and per-value-head RMSNormGated. The findings
  below are therefore about **test coverage, robustness, API fidelity, and
  performance**, not math errors.

## Guiding principles

- Keep the std-only core. `matrixmultiply` stays the only required crate;
  `candle` stays opt-in behind features. Do not add general-purpose crates.
- Preserve the hand-written dequantizer and the `AGENTS.md` research notes.
- Every new behavior gets a committed test that passes on a **fresh clone**
  (no `/tmp` fixtures, no model download).
- Keep `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`
  green on every commit.

---

## Phase 0 — Test & docs foundation (low risk, do first)

The single biggest risk today is that the most important correctness guarantees
are *not actually tested on CI*: the dequantization-equivalence and tokenizer
tests silently return early when `/tmp` files are absent.

1. **Make dequant/matmul equivalence self-contained.** Rewrite
   `quantized_matmul_matches_f32_dequant` (`src/exl3.rs`, ~line 1672) to build a
   small synthetic EXL3 linear (trellis + suh/svh) committed under `tests/`, so
   all three `--weight-mode` paths are verified on every `cargo test`. Mark the
   real-model `/tmp/unstable_*.bin` tests `#[ignore]` with an explanatory
   message so their intent is explicit.
   *Acceptance:* a clean clone with no `/tmp` fixtures proves dequant and
   quantized-matmul equivalence for 3INST / MCG / MUL1 codebooks.
   **Done:** `exl3::synthetic_tests` builds a deterministic 128x128 EXL3 linear
   entirely in memory and compares all three storage modes (packed / f32 cache
   / f16 cache) against `dequant_linear` for 3INST, MCG, MUL1 integer and
   MUL1 half-bitrate. Real-model `/tmp` tests are `#[ignore]`d with reasons.

2. **Add numeric unit tests for the decoder primitives.** `src/model.rs` has
   only one unit test today. Add hand-computed golden tests for: zero-centered
   RMSNorm, partial RoPE (rotate-half), q_proj interleaving, GQA gather/scale,
   attention gate sigmoid, a single Gated DeltaNet recurrence step (state
   update with the *updated* state), and per-value-head RMSNormGated.
   *Acceptance:* each primitive in `AGENTS.md` §6 has at least one assertion
   that does not require a model download.
   **Done:** extracted `split_q_gate`, `gqa_attention` and `deltanet_step`
   (plus `AttnShape` / `DeltaNetScratch`) without changing the math and added
   seven `model::tests` golden tests covering RMSNorm, per-head RMSNormGated,
   partial RoPE, q_proj interleaving, GQA grouping/scale/gate, the DeltaNet
   state update with decay, and fp16 embedding resolution.

3. **Fix doc drift.** At review `AGENTS.md` §9 said "82 tests / 4 ignored"
   and §3 said `exl3.rs` was 1556 lines (actual then: 84 / 4 and 1949 lines);
   `TODO.md` "Suggested Next Steps" #1 (reduce embedding memory) was already
   done. Update these to match reality and point `TODO.md` at this plan.
   *Acceptance:* no stale counts or completed items presented as open.
   **Done:** `AGENTS.md`, `TODO.md`, `README.md` and `tests/README.md` now
   reflect 88 passed / 10 ignored, the synthetic fixtures, the new golden
   tests and the explicit `#[ignore]` tagging of `/tmp`-dependent tests.

---

## Phase 1 — API, streaming fidelity & robustness (medium risk)

4. **Guard the request line in `http.rs`.** `read_request()` (`src/http.rs:35`)
   calls `BufReader::read_line` with no cap; the `MAX_HEADER_BYTES` check only
   starts *after* the request line is read (`:62`). A client streaming a
   huge line without `\n` grows `request_line` until OOM (600 s read timeout,
   up to 64 connections). Use `take(MAX_HEADER_BYTES as u64)` on the reader for
   both the request line and headers, and reject overlong lines.
   *Acceptance:* fuzz test with a >64 KiB newline-free request line returns
   an error without unbounded allocation.
   **Done:** the request line and headers are read through
   `Read::take(MAX_HEADER_BYTES + 1)`; overlong lines return "request line too
   large". `http::tests` covers both the flood and a normal request.

5. **Serialize global threadpool `run()` calls.** `threadpool::run()`
   (`src/threadpool.rs:78`) uses one global `POOL` and a per-call `Barrier`.
   Concurrent callers can interleave sends so that W0 waits on its barrier
   for a job queued behind another caller's barrier-blocked job — a deadlock.
   Mitigations (pick one): a `run_mutex` around dispatch, or assign jobs to
   workers so each `run()` gets a disjoint worker set. `catch_unwind` in the
   worker does not help (the block is `Barrier::wait()`).
   *Acceptance:* a stress test issues concurrent `matmul_t` calls from
   multiple threads and completes without hanging.
   **Done:** a global `RUN_LOCK` serializes whole `run()` calls (recovering
   from a poisoned lock); `threadpool::tests::concurrent_runs_complete_without_deadlock`
   exercises 8 callers x 50 batches.

6. **Strengthen `dot()` and CLI validation.**
   - `tensor::dot()` (`src/tensor.rs:82`) uses `a.len().min(b.len())`, silently
     hiding dimension mismatches. Make it assert equal lengths (or add a
     checked variant) while keeping the hot loop allocation-free.
   - `main.rs` CLI uses a closure that increments `i` *and* a separate
     `i += 1` (`:87-92`, `:124`) — fragile; refactor to a single increment.
     Clamp `--context-size` (and thread count) at startup.
   *Acceptance:* mismatched dot inputs are caught in tests; CLI parsing tests
   cover oversized context values.
   **Done:** `dot()` and `dot_f16()` assert equal lengths; `parse_cli()` now
   uses one iterator (no double index bump) and clamps context to
   `1..=1_000_000` and threads to `1..=1024`. Tests cover both.

7. **Align SSE streaming with OpenAI.**
   - Emit the `logprobs` delta in the **same chunk as** the content delta (or
     after it), not before it as today (`server.rs` ~479-493 vs 1599-1616).
   - `/v1/completions` streaming should emit a final empty-choices `usage`
     chunk, as chat already does (~1660-1670), rather than attaching usage to
     the last choice's finish chunk (~1789-1796).
   - Stream indexed `tool_calls` deltas (name / argument fragments) instead
     of only XML text with `finish_reason: "tool_calls"` (~1620-1626).
   *Acceptance:* a recorded SSE transcript matches the OpenAI SDK's expected
   chunk shape; add assertions in `server::tests`.
   **Done:** logprob deltas now follow the content delta; `/v1/completions`
   emits a final `choices: []` usage chunk; `ToolCallStreamer` + `tool_call_delta`
   emit indexed name/argument deltas and keep XML out of content. Assertions
   cover chunk ordering, usage placement, streamer parsing and delta shape.
   **Follow-up (real-client trace):** a client injecting an object-valued
   `__labels__` pseudo-parameter into every tool schema made Qwen3.5 emit
   `<parameter=__labels__\n{...}` without `>`; the old parser scanned to the
   next `>` and merged both parameters into one garbage argument, so tools
   received no `chat_id`. The parameter name now ends at `>` **or** a newline,
   and `trailing_prefix_len()`/`finish()` drop unfinished blocks instead of
   streaming raw XML. Regressions cover both streaming and non-streaming.

8. **Tighten generation semantics.**
   - Reject `repetition_penalty <= 0` (currently `0` silently disables the
     penalty, `server.rs` ~643-665 / validation ~1083-1088).
   - Decide and document how reasoning delimiters (`<think>`/`</think>`) are
     counted: they consume forward steps but are excluded from
     `completion_tokens` and sampling history. Either count them consistently
     or document the exclusion (already noted in `README.md`).
   - Add a server-side generation deadline / hard output cap so a slow client
     cannot hold the mutex-bound model indefinitely.
   *Acceptance:* invalid penalties return 400; a deadline test terminates a
   long generation; `completion_tokens` behavior is documented and tested.
   **Done:** `repetition_penalty` must be finite and `> 0`; a 300 s
   `MAX_GENERATION_SECONDS` deadline is checked each step (with an internal
   `Params.deadline` override for tests); reasoning delimiters were explicitly
   documented as excluded from `completion_tokens` in `README.md` / `AGENTS.md`.

9. **Bound regex quantifier allocations.** `Re::Quant` (`src/regex.rs`, ~164-196)
   builds `reachable: Vec<Vec<usize>>` with one entry per repetition, capped at
   `text.len() + 1`. The shipped Qwen pre-tokenizer pattern has only sequential
   quantifiers (no nested blow-up), but a crafted `tokenizer.json` with
   `(a+)+b`-style patterns causes extreme memory/time use. Generate end sets
   iteratively and cap work; validate untrusted patterns on load. This matters
   for the 8 MiB body-cap threat model.
   *Acceptance:* a pathological pattern returns an error or bounded-time
   result instead of unbounded allocation.
   **Done:** unbounded quantifiers cap at `MAX_QUANT_REPEATS = 4096` and a
   `WorkBudget` of 1M steps is threaded through `possible_ends` / `all_seq_ends`;
   `regex::tests::unbounded_quantifier_is_bounded` covers a long run and a
   crafted `(a+)+b` pattern.

---

## Phase 2 — Performance (higher risk, measure first)

10. **Reuse post-prefill state for `n > 1`.** Today each choice re-runs the full
    prefill (`server.rs` ~1674-1686 / 1817-1829). Snapshot the model state after
    prefill and restore it per choice; `n=8` should not cost `8x` prefill.
    *Acceptance:* benchmark shows sublinear cost growth for `n`; equivalence
    test confirms identical outputs for the same seed.
    **Done:** `Model::snapshot()/restore()` copy only the used KV prefix plus
    DeltaNet state; `Engine::prepare_*_prefill()` captures the post-prefill
    logits, and both endpoints prepare once when `n > 1`. Equivalence test
    `prefill_reuse_matches_fresh_prefill` passes; opt-in benchmark
    `bench_n_choices_prefill_reuse` reports fresh vs reused timing.

11. **Batched / continuous inference.** `Engine.model: Mutex<Model>` serializes
    all requests. Add a request queue and batch prefill via the existing
    multi-row sgemm path; potentially allow multiple `Model` instances behind
    a semaphore rather than a single mutex. This is the main throughput ceiling.
    **Done:** `n > 1` choices share one prefill (item 10); bounded admission
    (`Engine::try_admit()`, 16 slots, 429 overflow) prevents thread pileup.
    For independent requests, `Model` was split into an `Arc`-shared
    `ModelCore` (weights/config/tables) plus per-instance state; `Model::spawn()`
    creates cheap instances and `--instances N` wires them into an `Engine`
    pool with leases (`Vec<Mutex<Model>>` + `Condvar`). Extra instances cost
    KV/recurrent state, not another copy of the weights. Matmul kernels are
    still not batched across requests (that needs batch-aware attention and
    DeltaNet); documented in `AGENTS.md` §8.

12. **Memory optimizations (in order of value).**
    - f16 KV cache with per-step f32 upcast (halves context RAM).
    - f16 embedding for tied models without breaking `lm_head` (~1 GB f32
      matrix remains for the 0.8B model).
    - Only if `packed` mode is to be supported interactively: SIMD gather for
      the LUT decode (~3.5 s/token today — bottleneck is codebook decode, not
      the dot product).
    *Acceptance:* before/after numbers from `bench_generation_throughput` and
    peak-RSS measurements; no correctness regression.
    **Done:** fp16 KV cache (2 bytes/value, LUT-decoded per step) halves context
    RAM (`kv_cache_stores_fp16_bits`). All plain embeddings are fp16 and tied
    `lm_head` uses the new threaded `matmul_f16` (`f16_embedding_matmul_matches_f32`,
    `tied_f16_embedding_supports_forward`). The packed decode hot path now uses
    `unpack_tile_f32` — a direct codebook f32 LUT lookup that removes the
    per-codeword arithmetic + fp16 conversion (bit-identical result, asserted
    against `unpack_tile`). True SIMD gather is not available on stable std-only
    Rust (no portable gather API); the arithmetic bottleneck it targeted is
    gone. Real-model RSS numbers still require `EXLLAMA_BENCH_MODEL`.

13. **GPU weight-cache key.** `compute.rs` (~270-289) keys cached device
    tensors by raw pointer; an address reuse can yield a stale entry (bounded
    by a 512-entry clear). Key by tensor identity/generation instead.
    *Acceptance:* a reuse test forces the eviction path and verifies refresh.
    **Done:** the GPU cache key is now `(pointer, rows, cols, fingerprint)`,
    where the fingerprint is FNV-1a over sampled weight elements (including the
    last). A GPU-only test verifies different buffers map to different keys;
    `cargo check --features gpu` passes.

---

## Phase 3 — Exploration (large, optional)

14. Vision tower inference for `Qwen3_5ForConditionalGeneration`
    (`image_url` / `video` content) — vision tensors are currently skipped.
15. Other EXL3 architectures behind a generic config/weight abstraction
    (Llama, Qwen3, ...).
16. Unicode NFC normalization (currently ignored), reading `chat_template`
    from `tokenizer.json` instead of a hard-coded template, Dockerfile,
    crates.io publish, and an Anthropic-compatible `/v1/messages` endpoint.
17. Structured logging (`EXLLAMA_LOG`, replacing the ad-hoc `EXLLAMA_DEBUG`
    dump), remove dead logic (`multi_step_tool` computed then discarded in
    `chat.rs` ~117-130).

---

## Suggested execution order

**Phases 0–2 are fully implemented** → **Phase 3**. Phase 2 now includes shared
prefill, bounded admission, multiple Arc-shared model instances (`--instances`),
fp16 KV cache, fp16 tied embeddings, and the direct-LUT packed decode. Real-model
timing/RSS numbers still need `EXLLAMA_BENCH_MODEL`. Do not commit until the
user explicitly approves.

**Post-review benchmark fixes (not a new phase):** the opt-in benchmarks now
initialize the threadpool and compute backend exactly like the server (before,
matmuls ran single-threaded and the thread count was ignored), accept
`EXLLAMA_BENCH_THREADS` / `EXLLAMA_BENCH_WEIGHT_MODE` /
`EXLLAMA_BENCH_COMPUTE`, and report prefill and decode rates separately via new
`GenResult` timing fields — the old "decode tok/s" was completion tokens divided
by total request time (including the prompt pass). The auto backend picker's
GEMV-vs-prefill score now weights decode 4x (extracted as `pick_by_score` with a
regression test), since decode repeats per token and a 4x GEMV win must not be
masked by a 2x prefill win. The f16 decode-cache build now uses a `decode_bits`
LUT instead of per-codeword arithmetic (it was ~10x slower than the f32 cache
build), with a bit-identity test; the rerun confirmed f16 load dropped from
54.6 s to 3.9 s for 159 linears. Real-model A/B on a 12th-gen host showed f32
fastest (~11.9 / 8.6 / 1.6 tok/s for f32 / f16 / packed) — documented in
`AGENTS.md` §8 and `README.md`. Verified: 115 passed / 11 ignored, fmt and
clippy clean.

## Post-v0.4.3 follow-up: tool-call robustness

A real-client trace exposed two edge cases in tool-call handling, both fixed
with regression tests based on the exact observed output:

- A client that injects an object-valued `__labels__` pseudo-parameter into
  every tool schema makes Qwen3.5 emit `<parameter=__labels__\n{...}` without
  the closing `>`. The old parser scanned forward to the next `>`, merging
  `__labels__` and `chat_id` into one garbage argument — tool execution then
  failed with "missing required positional argument". Parameter names now
  terminate at `>` **or** a line break, so each argument stays separate.
- A generation that stops mid-call (`finish_reason=length`) left the unfinished
  `<tool_call>` XML in `content`. `ToolCallStreamer::finish()` and
  `parse_tool_calls()` now drop incomplete blocks (and partial opener
  prefixes), in both streaming and non-streaming paths.

Verification: **115 passed, 11 ignored**, `cargo fmt --all -- --check` and
`cargo clippy --all-targets -- -D warnings` clean.


## Device-resident GPU forward (post-v0.4.3)

Implemented per the requirement that, on GPU builds, all weights stay on device
and no activations cross the bus during prefill/decode.

- `src/gpu.rs` (new, feature `gpu`): `GpuWeights` decodes every EXL3 W_hat once
  and uploads it as an fp16 device tensor; plain matrices/embeddings are fp16,
  vectors/RoPE/H128 are fp32. `GpuRunner` owns shared weights plus per-instance
  device state; the whole forward (RMSNorm, RoPE, Hadamard, GQA attention,
  conv1d, Gated DeltaNet recurrence, gated RMSNorm, MLP, lm_head) runs with
  candle tensor ops. KV cache and DeltaNet state stay on device. Prefill copies
  logits to the host once for the whole prompt; decode once per token.
- `Model::load` builds the GPU runner when the backend is `Gpu`, loads EXL3 in
  `packed` mode (small host trellis as fallback) and ignores `--weight-mode`.
  `Model::spawn`, `reset`, `snapshot`/`restore` and `forward` dispatch to it.
  The CPU path is unchanged and remains the fallback.
- `half` is an opt-in dependency under the `gpu` feature (candle already
  compiles it transitively). Device GEMMs use fp16 inputs with f32 accumulate,
  so outputs can differ slightly from the f32 CPU path.
- Verified: `cargo test` (115 passed / 11 ignored), `cargo test --features gpu`
  (123 passed / 11 ignored), fmt and clippy clean with and without `gpu`.
  Device-op tests run on candle's CPU device; **real-GPU correctness and
  throughput must be validated on the target hardware** with
  `EXLLAMA_BENCH_COMPUTE=gpu` A/B runs (the benchmark accepts `gpu` now).

### Follow-up: GPU forward shape/indexing fixes
A real-model prefill failed with `shape mismatch in mul, lhs: [16,128],
rhs: [1,128]`. Candle's `.mul()` needs exactly equal shapes (only
`broadcast_mul()` broadcasts). Fixed in `src/gpu.rs`: `rms_norm` now
broadcasts the 1-D weight over 2-D q/k tensors; per-head gated norm broadcasts
`(1,vd)` and `(nvh,vd)`; q_proj reshape is `(nh,hd*2)` (query|gate contiguous
per head, not `(nh,2,hd)`); RoPE tables are indexed with one row
(`narrow(0,pos,1)`); and the full-attention KV gather selects one KV head
(`narrow(1,kh,1)` + `squeeze(1)`). Added
`gpu::tests::device_forward_runs_end_to_end_on_cpu` (multi-head tiny model,
prefill + decode) and a synthetic `q_matmul` test — together they exercise
every device forward branch and catch shape bugs without a GPU.

### REMINDER — future work: fully on-device sampling
Sampling still runs on the host after the single logits copy. Implement
temperature / top-k / top-p / greedy sampling with candle ops so only the
sampled token id crosses back, eliminating the last per-token device→host
transfer. Tracked in `TODO.md` as a P1 item.
