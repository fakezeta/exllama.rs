# TODO — exllama.rs roadmap

A prioritized list of known gaps and potential improvements. Completed items are
checked off with a short note; open items are unchecked.

> Legend: **P0** = blocks real-world use · **P1** = important · **P2** = nice-to-have · **P3** = exploration

> **See also:** [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) — the reviewed,
> phased plan (with acceptance criteria) that groups the open items below into
> executable work. This file remains the per-item checklist.

---

## Validation & Correctness

- [x] **(P0) End-to-end test with a real EXL3 model.** Done — validated on
  `UnstableLlama/Qwen3.5-0.8B-exl3-8.00bpw`. Chat completions produce coherent
  answers ("The capital of France is **Paris**.", arithmetic, lists, greetings)
  with correct `finish_reason: "stop"`. See `tests/README.md`.
- [x] **(P0) Dequantization ground-truth test.** Done —
  `exl3::real_model_tests::dequant_matches_python_reference` validates real
  layer-0 data (unpack, MCG decode, H_k, suh, H_n, svh) against a pure-NumPy
  reference across multiple Hadamard blocks and tiles. K=6 (lm_head) unpack is
  also covered by a consistency test.
- [x] **(P1) Verify conv1d weight layout.** Done — `cw[c*kernel + j]` matches
  PyTorch `(conv_dim, 1, 4)` depthwise layout; confirmed on the real model.
- [x] **(P1) Verify q_proj interleaving.** Done — per-head `[query, gate]`
  interleaving verified from HF source and by real-model output.
- [x] **(P2) Golden-file test for chat prompts.** Done — committed prompt
  fixtures in `tests/golden/` are asserted by `chat::tests` (no `/tmp`
  dependency); `tokenizer::tests` additionally verify added-token protection
  against the real tokenizer when available.
- [x] **(P2) Test streaming SSE end-to-end.** std-only HTTP tests in
  `server::tests` verify chunk framing, `[DONE]`, `reasoning_content` deltas,
  usage chunks, and CORS against a tiny synthetic model. Real-client
  compatibility (`curl -N` / OpenAI SDK) still worth a manual smoke test.
- [x] **(P0) Phase 0 hardening.** JSON parser depth cap + strict strings;
  checked safetensors headers and trellis-width validation; context-budget
  clamping and empty-input rejection; single-decode of plain 2-D tensors;
  half-integer (MUL1) unpack reference test. Regression tests added.
- [x] **(P1) Phase 1 fidelity.** Regex grouped backtracking + lazy quantifiers +
  isolated split; f64 sampling draw, top_p/seed validation, collision-free RNG
  seed 0; strict pre-tokenizer parsing and config added tokens; EOS/thinking
  defaults from config; chat `reasoning_content` / tool-call round-trip and
  committed golden prompts. Regression tests added.
- [x] **(P2) Phase 2 memory/perf.** fp16 embedding storage with per-row decode,
  direct fp16 codebook bits for the f16 cache (no f32 round-trip), blocked
  Hadamard rows, shard-byte compaction after the linear pass, and a persistent
  worker pool replacing per-matmul thread spawning. Equivalence tests added;
  real-model throughput benchmark still optional.
- [x] **(P0) Make the core equivalence tests fresh-clone-safe.** Done — see
  plan Phase 0 #1: `exl3::synthetic_tests` builds a deterministic 128x128 EXL3
  linear in memory and checks all three `--weight-mode` paths against full
  dequant for 3INST, MCG, MUL1 (integer + half-bitrate). The `/tmp`-dependent
  real-model tests (and the real-tokenizer tests) are now explicitly `#[ignore]`d
  with reasons; `cargo test` proves the guarantee on a fresh clone.
- [x] **(P0) Numeric unit tests for decoder primitives.** Done — see plan
  Phase 0 #2: extracted `split_q_gate`, `gqa_attention` and `deltanet_step`
  helpers without changing the math, and added hand-computed golden tests for
  RMSNorm, per-head RMSNormGated, partial RoPE, q_proj interleaving, GQA
  grouping/scale/gate and the DeltaNet recurrence (with decay). `model.rs`
  now has 7 unit tests.
- [x] **(P1) Fix doc drift.** Done — `AGENTS.md` counts/line numbers, `TODO.md`
  stale "next steps", and this plan's baseline are updated to 115 passed /
  11 ignored (88 after Phase 0, 99 after Phase 1, 102 after Phase 2, 110 after
  the benchmark fixes, 115 after the tool-call parser fixes).
- [x] **(P1) Bounded request-line reading.** Done — see plan Phase 1 #4:
  `read_request()` reads the request line and headers through
  `Read::take(MAX_HEADER_BYTES + 1)` and rejects overlong lines. `http::tests`
  covers a 128 KiB newline-free flood and a normal request.

### Resolved during real-model validation (for the record)

These were real bugs found by running the 0.8B EXL3 model, all fixed:
- Bitrate detection: `0.0f32.is_normal()` misclassified every integer bpw as
  half → "half bitrate requires mul1 codebook" error. Now integer math.
- Newer `tokenizers` format: missing `byte_encoder`, string merges,
  `pretokenizers` key, `{"Regex": ...}` pattern — all handled.
- `lm_head` name/BF16: quantized head stored as `lm_head` (no `.weight`), many
  tensors BF16 → all-zero logits emitting only token 0. Fixed aliases + BF16.
- Non-quantized 2-D linears (`in_proj_a`/`in_proj_b`) now loaded as matrices.
- Conv circular buffer holds `kernel-1` history slots, not `kernel`.
- Gated RMSNorm normalizes **per value head**, not over the whole vector.
- UTF-8 char-boundary panic + poisoned mutex on multibyte token boundaries.

---

## Memory & Performance

- [x] **(P0) Compute directly with the quantized weights.** Done — `QuantLinear`
  applies H_k/H_n to the input/output vectors (not the weight matrix), with
  `--weight-mode` choosing storage: `f32` (default, ~4× disk, fastest on a
  12th-gen x86 host), `f16` (~2× disk, ~9 tok/s there), `packed` (~1× disk but
  ~1.7 tok/s — not interactive). Ranking is host-dependent; see `AGENTS.md` §8.
  All paths verified equivalent by `quantized_matmul_matches_f32_dequant`.
  Tied-embedding models skip the lm_head decode (plain embedding used directly).
  All three modes validated on the real 0.8B EXL3 model: `f32` and `f16` gave
  byte-identical, coherent answers quickly; `packed` gave the same correct
  answer but at ~3.5 s/token (long requests hit client timeouts).
- [x] **(P1) Reduce embedding memory.** Done for **all** models — `embed_tokens`
  is stored as fp16 code bits (2 bytes/weight); one row is decoded per token and
  tied `lm_head` uses the threaded `matmul_f16` instead of an f32 matrix. Saves
  roughly 2 GB for 9B and ~1 GB for tied 0.8B.
- [x] **(P2) Speed up the `packed` mode.** Done as far as stable std-only Rust
  allows: `unpack_tile_f32()` looks up the codebook's f32 table directly,
  removing per-codeword arithmetic and the fp16 conversion (bit-identical to
  the reference path, asserted by a test). True SIMD gather has no portable
  stable API; `f16` remains the recommended low-RAM mode.
- [x] **(P2) Speed up the `f16` cache build.** Done — `build_decode_cache()` used
  the old per-codeword `decode()` arithmetic while the f32 path used a LUT,
  making f16 load ~10x slower (54.6 s vs 10.8 s for 159 linears). It now uses a
  `decode_bits` fp16-pattern LUT (bit-identical, asserted by a test); f16 load
  dropped to ~3.9 s for the linears, on par with f32. Real-model A/B on a
  12th-gen host: f32 ~11.9 tok/s, f16 ~8.6, packed ~1.6 — f32 is the
  fastest mode there, so the old "f16 is near-f32 speed" claim is corrected in
  `AGENTS.md` §8 and `README.md`.
- [x] **(P1) Free shard bytes before allocating weights.** Done — quantized
  linears copy only their trellis; after the linear pass, `Shards::compact_to`
  drops all non-plain tensor byte ranges before plain tensors are decoded.
  `Shards` itself is dropped when `Exl3Model::load` returns.
- [x] **(P1) BLAS/GPU compute backends.** Done — `src/compute.rs` with `matrixmultiply::sgemm` (pure-Rust SIMD BLAS) and opt-in `candle-core` GPU (features `gpu-metal`/`gpu-cuda`), auto-selected at startup by a GEMV micro-benchmark. CLI `--compute auto|blas|gpu|scalar`. Key finding: for memory-bound decode GEMV the hand-written threaded dot is 3-4x faster than sgemm, so `auto` usually picks `scalar`; BLAS/GPU help matrix-matrix batches.
- [x] **(P2) GPU weight-cache key.** Done — see plan Phase 2 #13: cache keys
  include a sampled-weight fingerprint so a recycled pointer cannot reuse a
  stale device tensor. GPU-only test added.
- [x] **(P2) Batched / continuous inference (see plan #11).** `n > 1` choices share one prefill; bounded admission (16 slots, 429) prevents pileup; `--instances N` creates independent `Model` instances over an Arc-shared core (weights shared, only KV/recurrent state duplicated) with a pooled lease. Matmul kernels are not batched across requests — that needs batch-aware attention/DeltaNet; see `AGENTS.md` §8.
- [x] **(P2) KV cache quantization.** Done — full-attention KV cache stores fp16 bits (2 bytes/value) and decodes the used prefix via the `half_f32` LUT per step, halving context RAM. `model::tests::kv_cache_stores_fp16_bits` added.
- [x] **(P2) Precompute RoPE cos/sin tables.** Done — tables built once at load; DeltaNet per-head scratch is also reused.
- [x] **(P1) Guard the global threadpool against concurrent `run()` calls.** Done
  — see plan Phase 1 #5: a global `RUN_LOCK` serializes whole `run()` calls,
  with a concurrency stress test covering 8 callers.
- [x] **(P1) Catch dimension mismatches in `tensor::dot()`.** Done — see plan
  Phase 1 #6: `dot()` and `exl3::dot_f16()` assert equal lengths; `parse_cli()`
  uses one iterator and clamps context size / thread count. Tests added.
- [x] **(P2) Reuse post-prefill state for `n > 1`.** Done — see plan Phase 2
  #10: `Model::snapshot()/restore()` + `Engine::prepare_*_prefill()` let all
  choices share one prompt pass; equivalence test and opt-in benchmark added.

---

## Model & Architecture Support

- [x] **(P1) Support `tie_word_embeddings = true`.** Done — `Model::w()` aliases
  `lm_head` and falls back to `embed_tokens.weight`.
- [x] **(P1) Read `eos_token_id` from `config.json`.** Done — `ModelConfig.eos_token_id` (top-level or `text_config`, scalar or first array element) overrides the tokenizer fallback at startup.
- [x] **(P2) Support other Qwen3.5 sizes (0.8B / 2B / 4B) explicitly.** Done — `hidden_size >= 2560` selects the thinking-on default (9B/4B); smaller models default off. An explicit `enable_thinking` in config wins.
- [ ] **(P2) Vision tower inference** for `Qwen3_5ForConditionalGeneration` with `image_url` / `video` content. Large effort; vision tensors currently skipped.
- [ ] **(P3) Support other EXL3 architectures** (Llama, Qwen3, etc.) behind a generic config/weight abstraction.

---

## API & Protocol Compatibility

- [x] **(P1) Full OpenAI parameter coverage.** Added `logprobs` / `top_logprobs` (chat: bool + 0..=20; completions: integer, parallel-array shape; SSE deltas), `n` (1..=128, sequential choices with per-choice seeds), `user` (validated, no-op), and `response_format` (`text` / `json_object`; best-effort prompt hint, no constrained decoding; `json_schema` rejected). Together with `presence_penalty`, `frequency_penalty`, `logit_bias`, `stop_token_ids`, and `stream_options.include_usage`, all documented request parameters are now covered.
- [x] **(P1) Proper usage accounting** for streamed responses. Final chunk includes `usage` when `stream_options.include_usage` is set.
- [x] **(P1) Separate `reasoning_content` field.** Done — thinking traces go to `reasoning_content` (message field / SSE delta), answer to `content`, matching vLLM/SGLang.
- [x] **(P2) `stop` as token-id array.** Done — `stop_token_ids` plus numeric arrays in `stop`.
- [x] **(P2) HTTP robustness.** Done — 64 KiB header / 8 MiB body caps, strict Content-Length, 600 s read/write timeouts, 64-connection cap, write-error cancellation, 503 on overload. Keep-alive is intentionally not supported (Connection: close).
- [x] **(P2) CORS headers** on responses. Present on all responses and preflight (was already present before Phase 2; TODO was stale).
- [x] **(P2) `model` field validation** on requests. `/v1/models` config-derived capabilities remain open.
- [x] **(P1) Apply `repetition_penalty`.** Done — `sample()` now divides the
  logit of already-generated tokens by the penalty (positive logits) / multiplies
  (negative), with HashSet dedup so repeated history doesn't stack. Breaks
  greedy repetition loops when set > 1. Unit test added.
- [x] **(P1) Align SSE streaming with OpenAI.** Done — see plan Phase 1 #7:
  logprob deltas follow content; `/v1/completions` emits a final `choices: []`
  usage chunk; `ToolCallStreamer` emits indexed `tool_calls` deltas (name +
  argument fragments) without leaking the XML into content. Tests assert
  chunk ordering, usage placement and delta shape.
- [x] **(P1) Generation deadline / hard output cap.** Done — see plan Phase 1
  #8: `repetition_penalty` must be finite and `> 0`; a 300 s deadline is
  checked each step; reasoning delimiters are documented as excluded from
  `completion_tokens`. Deadline path covered by a test.
- [ ] **(P3) Anthropic-compatible API** (`/v1/messages`) as a secondary endpoint.

---

## Tokenizer

- [ ] **(P1) Unicode NFC normalization.** `normalizer: {"type": "NFC"}` is currently ignored. Implement via a small table or accept the limitation explicitly.
- [ ] **(P2) `chat_template` from tokenizer.json.** Template is hard-coded in `chat.rs`; read/interpret it (or at least verify it matches).
- [x] **(P2) Added tokens from `added_tokens_decoder`.** Done — ids found only in the sibling `tokenizer_config.json` are loaded and protected.
- [ ] **(P2) Byte-fallback / BPE edge cases.** Fuzz `bpe_encode_word` against Python `tokenizers` with random UTF-8.
- [x] **(P1) Bound regex quantifier allocations / ReDoS.** Done — see plan
  Phase 1 #9: unbounded quantifiers cap at `MAX_QUANT_REPEATS = 4096` and a
  `WorkBudget` (1M steps) is threaded through the matcher. Test covers a long
  whitespace run and a crafted `(a+)+b` pattern.

---

## Chat Template

- [x] **(P1) Tool calling (streaming + non-streaming).** Done — OpenAI `tools` are rendered into the Qwen3.5 system preamble, assistant `tool_calls` into `<function=...>` XML, and generated `<tool_call>` blocks are parsed back into `tool_calls` with `finish_reason: "tool_calls"`. Streaming emits indexed `tool_calls` deltas (see plan Phase 1 #7). A real-client trace exposed two edge cases, both fixed with regressions: a `<parameter=name\n{...}` header without `>` used to swallow the following argument (breaking `chat_id`), and a call truncated at `finish_reason=length` leaked raw XML into `content`.
- [x] **(P1) `enable_thinking` model-size defaults.** Done — see Model & Architecture Support above.
- [ ] **(P2) Multi-modal content parts** (`content: [{type: "text", ...}]`) — currently only plain strings.
- [x] **(P2) `reasoning_content` on responses.** Done (see API section); `Message` also accepts `reasoning_content` on input for multi-turn history.
- [ ] **(P3) Full Jinja2 subset** or a template engine (hard with zero deps; likely not worth it).

---

## Developer Experience

- [x] **(P1) Model loading progress / diagnostics.** Done — `[load]` per 16 linears, timing totals and final RSS (`rss=… MiB` on Linux).
- [ ] **(P1) Better error messages** for missing/incompatible files (e.g. "model looks like EXL2, not EXL3").
- [ ] **(P2) Structured logging** with configurable verbosity (`RUST_LOG`-like env var). A temporary `EXLLAMA_DEBUG` logit dump exists but is ad-hoc.
- [ ] **(P2) Docker image / Dockerfile** for easy deployment.
- [x] **(P2) Example/test scripts.** Done — `tests/` contains smoke and diagnostic scripts with `tests/README.md`.
- [x] **(P3) Benchmark suite (baseline).** Ignored tests `bench_generation_throughput` (requires `EXLLAMA_BENCH_MODEL`), `bench_blas_vs_scalar`, `bench_quant_matmul` and `profile_packed_decode_vs_dot` cover end-to-end, CPU and quantized paths. `bench_generation_throughput` and `bench_n_choices_prefill_reuse` now initialize the threadpool/compute backend like the server, take `EXLLAMA_BENCH_THREADS` / `EXLLAMA_BENCH_WEIGHT_MODE` / `EXLLAMA_BENCH_COMPUTE`, and report prefill vs decode time separately (the old "decode tok/s" divided by total time). The auto backend picker now weights the GEMV probe 4x over prefill so a large decode win is not masked (`compute::pick_by_score`, with a regression test).

---

## Infrastructure

- [x] **(P1) CI: aarch64 Linux build.** Done — native `ubuntu-24.04-arm` runner for `aarch64-unknown-linux-gnu` (no cross-compilation needed). macOS Intel (`macos-13`) target removed.
- [x] **(P2) CI: run `cargo test` on Windows/macOS.** All matrix jobs run `cargo test`; verified in `ci.yml`.
- [ ] **(P2) Publish crates.io release** (currently GitHub-only).
- [x] **(P3) Fuzzing harness.** `src/fuzz.rs` runs deterministic, seeded random inputs through the JSON, HTTP, regex, chat, tokenizer and API-parameter parsers on every `cargo test` (no new dependency). `cargo fuzz` with libfuzzer remains optional if deeper coverage is needed.

---

## Suggested Next Steps (ordered)

The authoritative phased plan (with acceptance criteria) is
[`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md). **Phases 0–2 are fully complete**;
execute the remaining work in this order:

1. **Phase 3 — exploration.** Vision, other architectures, NFC / `chat_template`
   loading, structured logging, Anthropic endpoint.
2. **Manual validation still worth doing:** the end-to-end benchmark
   (`EXLLAMA_BENCH_MODEL=/path cargo test --release bench_generation_throughput -- --ignored --nocapture`),
   a real-client SSE smoke test (`curl -N`), and real-model runs on 2B/4B/9B
   and MUL1 codebooks (only the 0.8B 8-bpw MCG model has been exercised), and
   `bench_n_choices_prefill_reuse` to confirm sublinear `n > 1` cost.

Commit only on explicit approval — see the status note in `IMPROVEMENT_PLAN.md`.

---

## Notes

- Keep the core std-only (no general-purpose crates). `matrixmultiply` is the only required dependency; `candle` is opt-in for GPU. If a feature truly needs a crate, put it behind a feature flag and document the trade-off.
- Maintain `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` — CI enforces both.
- Update `AGENTS.md` whenever architecture or format details change.
- Real-model diagnostics live in `tests/`; reference data and research notes are described in `AGENTS.md` §8.
