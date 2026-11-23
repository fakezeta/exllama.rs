# Improvement Plan — exllama.rs

This document is the reviewed, prioritized plan for hardening and extending the
server. It is derived from a full read of the four markdown docs (`README.md`,
`TODO.md`, `tests/README.md`, `AGENTS.md`) and a code review of every `src/`
module. It complements `TODO.md`: that file tracks individual checked-off tasks,
while this file groups the *open* work into executable phases with acceptance
criteria.

> **Status:** Phase 0 is **implemented and verified** (see the completion notes
> below); Phases 1–3 remain plans. Commit only on explicit approval.

## Baseline (at time of review)

- Branch `main`, tag `v0.4.2`, working tree clean at review time.
- At review: `cargo test` **84 passed, 4 ignored**. After Phase 0:
  **88 passed, 10 ignored** (6 real-model / `/tmp` fixtures now explicitly
  ignored with reasons, 4 benchmarks). `cargo fmt --all -- --check` and
  `cargo clippy --all-targets -- -D warnings` remain clean.
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

5. **Serialize global threadpool `run()` calls.** `threadpool::run()`
   (`src/threadpool.rs:78`) uses one global `POOL` and a per-call `Barrier`.
   Concurrent callers can interleave sends so that W0 waits on its barrier
   for a job queued behind another caller's barrier-blocked job — a deadlock.
   Mitigations (pick one): a `run_mutex` around dispatch, or assign jobs to
   workers so each `run()` gets a disjoint worker set. `catch_unwind` in the
   worker does not help (the block is `Barrier::wait()`).
   *Acceptance:* a stress test issues concurrent `matmul_t` calls from
   multiple threads and completes without hanging.

6. **Strengthen `dot()` and CLI validation.**
   - `tensor::dot()` (`src/tensor.rs:82`) uses `a.len().min(b.len())`, silently
     hiding dimension mismatches. Make it assert equal lengths (or add a
     checked variant) while keeping the hot loop allocation-free.
   - `main.rs` CLI uses a closure that increments `i` *and* a separate
     `i += 1` (`:87-92`, `:124`) — fragile; refactor to a single increment.
     Clamp `--context-size` (and thread count) at startup.
   *Acceptance:* mismatched dot inputs are caught in tests; CLI parsing tests
   cover oversized context values.

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

9. **Bound regex quantifier allocations.** `Re::Quant` (`src/regex.rs`, ~164-196)
   builds `reachable: Vec<Vec<usize>>` with one entry per repetition, capped at
   `text.len() + 1`. The shipped Qwen pre-tokenizer pattern has only sequential
   quantifiers (no nested blow-up), but a crafted `tokenizer.json` with
   `(a+)+b`-style patterns causes extreme memory/time use. Generate end sets
   iteratively and cap work; validate untrusted patterns on load. This matters
   for the 8 MiB body-cap threat model.
   *Acceptance:* a pathological pattern returns an error or bounded-time
   result instead of unbounded allocation.

---

## Phase 2 — Performance (higher risk, measure first)

10. **Reuse post-prefill state for `n > 1`.** Today each choice re-runs the full
    prefill (`server.rs` ~1674-1686 / 1817-1829). Snapshot the model state after
    prefill and restore it per choice; `n=8` should not cost `8x` prefill.
    *Acceptance:* benchmark shows sublinear cost growth for `n`; equivalence
    test confirms identical outputs for the same seed.

11. **Batched / continuous inference.** `Engine.model: Mutex<Model>` serializes
    all requests. Add a request queue and batch prefill via the existing
    multi-row sgemm path; potentially allow multiple `Model` instances behind
    a semaphore rather than a single mutex. This is the main throughput ceiling.

12. **Memory optimizations (in order of value).**
    - f16 KV cache with per-step f32 upcast (halves context RAM).
    - f16 embedding for tied models without breaking `lm_head` (~1 GB f32
      matrix remains for the 0.8B model).
    - Only if `packed` mode is to be supported interactively: SIMD gather for
      the LUT decode (~3.5 s/token today — bottleneck is codebook decode, not
      the dot product).
    *Acceptance:* before/after numbers from `bench_generation_throughput` and
    peak-RSS measurements; no correctness regression.

13. **GPU weight-cache key.** `compute.rs` (~270-289) keys cached device
    tensors by raw pointer; an address reuse can yield a stale entry (bounded
    by a 512-entry clear). Key by tensor identity/generation instead.
    *Acceptance:* a reuse test forces the eviction path and verifies refresh.

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

**Phase 0 is complete** → **Phase 1 → Phase 2 → Phase 3**. Phase 0 was
low-risk, closed the most dangerous gap (untested core guarantees) and makes
every later change safer. Start the remaining work with Phase 1 item 4 (the
bounded HTTP request line), then item 5 (threadpool deadlock). Do not commit
until the user explicitly approves.
