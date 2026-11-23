# TODO — exllama.rs roadmap

A prioritized list of known gaps and potential improvements. Completed items are
checked off with a short note; open items are unchecked.

> Legend: **P0** = blocks real-world use · **P1** = important · **P2** = nice-to-have · **P3** = exploration

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
- [x] **(P2) Golden-file test for chat prompts.** Done (partial) —
  `tokenizer::tests` and `chat::prompt_debug` lock down token ids for canned
  conversations, including added-token protection (special tokens are single ids).
- [x] **(P2) Test streaming SSE end-to-end.** std-only HTTP tests in
  `server::tests` verify chunk framing, `[DONE]`, `reasoning_content` deltas,
  usage chunks, and CORS against a tiny synthetic model. Real-client
  compatibility (`curl -N` / OpenAI SDK) still worth a manual smoke test.

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
  `--weight-mode` choosing storage: `f32` (default, ~4× disk, fast), `f16`
  (~2×, near-f32 speed), `packed` (~1× but ~10+ s/token — too slow on CPU).
  All paths verified equivalent by `quantized_matmul_matches_f32_dequant`.
  Tied-embedding models skip the lm_head decode (plain embedding used directly).
  All three modes validated on the real 0.8B EXL3 model: `f32` and `f16` gave
  byte-identical, coherent answers quickly; `packed` gave the same correct
  answer but at ~3.5 s/token (long requests hit client timeouts).
- [ ] **(P1) Reduce embedding memory.** `embed_tokens` is still a plain f32
  matrix (~1 GB for 0.8B, ~4 GB for 9B). Store as f16 and convert rows on
  lookup, or keep the raw safetensors bytes and decode single rows lazily.
  Note: quantized linear trellises are now freed after decode, so this is the
  largest remaining allocation.
- [ ] **(P2) Speed up the `packed` mode.** Its bottleneck is codebook decode
  (not the dot product), so SIMD on `dot()` won't help. O(1) window extraction,
  decode LUTs and stack buffers are already in place; further gains need SIMD
  gather for the LUT or a different packed representation. Low priority — `f16`
  gives 2× memory at near-f32 speed.
- [x] **(P1) Free shard bytes before allocating weights.** Done — quantized
  linears copy only their trellis; `Shards` (holding raw file bytes) is dropped
  when `Exl3Model::load` returns. Peak during load is ~2× on-disk briefly.
- [x] **(P1) BLAS/GPU compute backends.** Done — `src/compute.rs` with `matrixmultiply::sgemm` (pure-Rust SIMD BLAS) and opt-in `candle-core` GPU (features `gpu-metal`/`gpu-cuda`), auto-selected at startup by a GEMV micro-benchmark. CLI `--compute auto|blas|gpu|scalar`. Key finding: for memory-bound decode GEMV the hand-written threaded dot is 3-4x faster than sgemm, so `auto` usually picks `scalar`; BLAS/GPU help matrix-matrix batches.
- [ ] **(P2) Batched / continuous inference.** `Engine.model` is a `Mutex<Model>` and serializes all requests. Add a request queue, or support `n > 1` by expanding the prompt batch.
- [ ] **(P2) KV cache quantization.** Cache is f32 `(ctx, heads, head_dim)`; f16 or int8 could double context within the same RAM.
- [x] **(P2) Precompute RoPE cos/sin tables.** Done — tables built once at load; DeltaNet per-head scratch is also reused.

---

## Model & Architecture Support

- [x] **(P1) Support `tie_word_embeddings = true`.** Done — `Model::w()` aliases
  `lm_head` and falls back to `embed_tokens.weight`.
- [ ] **(P1) Read `eos_token_id` from `config.json`** instead of relying on the tokenizer fallback. Use `text_config.eos_token_id` when present.
- [ ] **(P2) Support other Qwen3.5 sizes (0.8B / 2B / 4B) explicitly.** They differ in `enable_thinking` default (off for 0.8B/2B) — detect size from config and set the default accordingly.
- [ ] **(P2) Vision tower inference** for `Qwen3_5ForConditionalGeneration` with `image_url` / `video` content. Large effort; vision tensors currently skipped.
- [ ] **(P3) Support other EXL3 architectures** (Llama, Qwen3, etc.) behind a generic config/weight abstraction.

---

## API & Protocol Compatibility

- [x] **(P1) Full OpenAI parameter coverage (partial):** `presence_penalty`, `frequency_penalty`, `logit_bias`, `stop_token_ids`, and `stream_options.include_usage` are implemented. Still open: `logprobs`, `n`, `user`, `response_format`.
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
- [ ] **(P3) Anthropic-compatible API** (`/v1/messages`) as a secondary endpoint.

---

## Tokenizer

- [ ] **(P1) Unicode NFC normalization.** `normalizer: {"type": "NFC"}` is currently ignored. Implement via a small table or accept the limitation explicitly.
- [ ] **(P2) `chat_template` from tokenizer.json.** Template is hard-coded in `chat.rs`; read/interpret it (or at least verify it matches).
- [ ] **(P2) Added tokens from `added_tokens_decoder`** (ids 248070..248076 exist only in `tokenizer_config.json`). Load them when available.
- [ ] **(P2) Byte-fallback / BPE edge cases.** Fuzz `bpe_encode_word` against Python `tokenizers` with random UTF-8.

---

## Chat Template

- [ ] **(P1) Tool calling.** Implement `<tool_call>` / `<tool_response>` rendering with function arguments, and parse model-generated tool calls back into `tool_calls` response fields.
- [ ] **(P1) `enable_thinking` model-size defaults.** Detect 0.8B/2B from config and flip the default to false. Currently defaults to true (9B/4B behavior).
- [ ] **(P2) Multi-modal content parts** (`content: [{type: "text", ...}]`) — currently only plain strings.
- [x] **(P2) `reasoning_content` on responses.** Done (see API section); the `Message` input struct still lacks a `reasoning_content` field for multi-turn history.
- [ ] **(P3) Full Jinja2 subset** or a template engine (hard with zero deps; likely not worth it).

---

## Developer Experience

- [x] **(P1) Model loading progress / diagnostics.** Done — `[load]` per 16 linears, timing totals and final RSS (`rss=… MiB` on Linux).
- [ ] **(P1) Better error messages** for missing/incompatible files (e.g. "model looks like EXL2, not EXL3").
- [ ] **(P2) Structured logging** with configurable verbosity (`RUST_LOG`-like env var). A temporary `EXLLAMA_DEBUG` logit dump exists but is ad-hoc.
- [ ] **(P2) Docker image / Dockerfile** for easy deployment.
- [x] **(P2) Example/test scripts.** Done — `tests/` contains smoke and diagnostic scripts with `tests/README.md`.
- [x] **(P3) Benchmark suite (baseline).** Ignored tests `bench_generation_throughput` (requires `EXLLAMA_BENCH_MODEL`), `bench_blas_vs_scalar`, `bench_quant_matmul` and `profile_packed_decode_vs_dot` cover end-to-end, CPU and quantized paths.

---

## Infrastructure

- [x] **(P1) CI: aarch64 Linux build.** Done — native `ubuntu-24.04-arm` runner for `aarch64-unknown-linux-gnu` (no cross-compilation needed). macOS Intel (`macos-13`) target removed.
- [x] **(P2) CI: run `cargo test` on Windows/macOS.** All matrix jobs run `cargo test`; verified in `ci.yml`.
- [ ] **(P2) Publish crates.io release** (currently GitHub-only).
- [ ] **(P3) Fuzzing harness** for JSON/HTTP parsers using `cargo fuzz` (would need a dev-dependency; conflicts with zero-dep goal — keep optional).

---

## Suggested Next Steps (ordered)

1. **Reduce embedding memory** — the f32 `embed_tokens` matrix is now the largest allocation. f16 storage + lazy row conversion is the next memory win.
2. **Run the end-to-end benchmark** with a real model: `EXLLAMA_BENCH_MODEL=/path cargo test --release bench_generation_throughput -- --ignored --nocapture`.
3. **Remaining API completeness** — `logprobs`, `n`, `user`, `response_format`.
4. **`enable_thinking` model-size defaults** — flip to false for 0.8B/2B.
5. **Tool calling** — the most-requested OpenAI feature for Qwen3.5 chat models.
6. **Manual SSE validation** with a real OpenAI client / `curl -N` (automated framing tests already pass).

---

## Notes

- Keep the core std-only (no general-purpose crates). `matrixmultiply` is the only required dependency; `candle` is opt-in for GPU. If a feature truly needs a crate, put it behind a feature flag and document the trade-off.
- Maintain `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings` — CI enforces both.
- Update `AGENTS.md` whenever architecture or format details change.
- Real-model diagnostics live in `tests/`; reference data and research notes are described in `AGENTS.md` §8.
