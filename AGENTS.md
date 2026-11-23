# AGENT.md — Developer Guide for exllama.rs

This document is a complete orienteering guide for AI agents (or humans) continuing development of this codebase. Read it fully before making changes. It captures the *why* behind the code, not just the *what*, so you don't have to re-derive the research.

---

## 1. Project Overview

**Goal:** A minimal Rust inference server for ExLlamaV3 (EXL3) quantized models, focused on the **Qwen3.5 text decoder** (`Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM`). It exposes an OpenAI-compatible HTTP API.

**Dependency policy:** JSON, HTTP, tokenizer, regex, PRNG and dequantizer are hand-written in `src/` with the standard library only. The only (non-optional) crate is `matrixmultiply` for the SIMD sgemm kernel; `candle-core` is opt-in for GPU. Do not add general-purpose crates (serde, tokio, serde_json, etc.) without explicit approval.

**Model scope:** Qwen3.5 only. The vision tower of multimodal checkpoints is ignored; text-only inference works. Other architectures are out of scope.

**Runtime:** CPU only. No GPU/CUDA/Python needed at runtime.

---

## 2. Quick Reference

### Build & test
```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build                      # debug
cargo build --release            # optimized binary at target/release/exllama-server
cargo test                       # 45 unit tests (4 ignored)
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
`temperature` (default 0.7), `top_p` (1.0), `top_k` (0 = off), `max_tokens` / `max_completion_tokens`, `stop` (string or array), `stop_token_ids`, `seed`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias` (token-id → bias), `stream` (bool), `stream_options.include_usage`, and for chat: `enable_thinking` (bool, default **true** for 9B/4B). `model` is validated against the server name. Invalid parameters are rejected with a 400.

---

## 3. Module Map

### `src/main.rs` — CLI + routing
- `Cli` struct + `parse_cli()` — hand-rolled arg parser (`-m/--model`, `-c/--context-size`, `-p/--port`, `--host`, `--tokenizer`, `--max-tokens`, `--threads`, `--name`).
- `main()` — loads `Model`, `Tokenizer`, builds `Engine`, wraps in `Arc`, binds `TcpListener`, calls `http::serve`.
- `route()` — dispatches HTTP requests (see table above).
- `models_json()` — builds `/v1/models` response.
- Has `#![allow(dead_code)]` at crate root (intentional unused helpers).

### `src/http.rs` — Minimal HTTP/1.1 server
- `Request { method, path, headers, body }` — parsed by `read_request()` (64 KiB header cap, 8 MiB body cap, rejects duplicate/invalid Content-Length).
- `respond()` — writes a complete response with Content-Length and CORS headers.
- `Streamer` — chunked SSE writer with `Cache-Control: no-cache`; `write_chunk()` returns false on a broken connection so generation aborts; `end()` writes `0\r\n\r\n`.
- `serve(listener, handler)` — accept loop; one OS thread per connection (capped at 64), 600 s read/write timeouts. Handler signature: `Fn(TcpStream, Request)` (takes ownership of the stream).

### `src/json.rs` — JSON parser/serializer
- `Json` enum: `Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>)` (no separate Float variant).
- Free function `json::parse(&str) -> Result<Json, String>` (NOT a method). Recursive-descent `Parser` handles strings with `\u` escapes, numbers, arrays, objects.
- `Json` has `get`, `as_str`, `as_i64`, `as_usize`, `as_f64`, `as_bool`, `as_arr`, `as_obj`, `to_string()` (via `Display`).
- `Json::Obj` preserves insertion order (used for response field ordering).
- `obj(&[(&str, Json)])` convenience constructor.

### `src/tensor.rs` — Matrices + matmul
- `Matrix { rows, cols, data: Vec<f32> }` — row-major. **Weights are stored `(out_features, in_features)`**; a linear computes `y = x @ W^T` via `matmul_t`.
- `F16Matrix { rows, cols, data: Vec<u16> }` — fp16 embedding storage; `row_f32(row)` decodes one row on lookup (used for untied models).
- `matmul_t(x, x_rows, x_cols, w, y, nthreads)` — splits work over the persistent pool (over rows if many input rows, over output neurons otherwise).
- **Rust 1.100 quirk:** `NonNull<f32>` and `*mut f32` are NOT `Send`. Workaround: `SharedPtr(*mut f32)` newtype with `unsafe impl Send`, `#[derive(Clone, Copy)]`, and a safe `write_at(offset, value)` method. Closures capture `SharedPtr` **by value** (Copy). Do not revert to raw pointers. Immutable inputs cross threads via `threadpool::SendSlice<T>`.

### `src/threadpool.rs` — Persistent worker pool
- `init(threads)` spawns a fixed set of workers (called from `main` with `--threads`); `run(jobs)` dispatches `Box<dyn FnOnce() + Send>` jobs through std channels and waits on a `Barrier` before returning.
- Jobs execute sequentially if `init` was never called (unit tests). A `BarrierGuard` releases the barrier even if a job panics, and the worker catches panics so the pool survives.
- Used by `tensor::matmul_t_scalar` and `QuantLinear::matmul` (f16/packed paths) instead of spawning scoped threads per call.

### `src/compute.rs` — Compute backend selection
- `Backend::{Scalar, Blas, Gpu}`; `init(preference)` picks at startup. `auto` (default): if GPU compiled in (`--features gpu-metal`/`gpu-cuda`) and a device is present → Gpu; else micro-benchmarks a 4096x1024 GEMV (`pick_fastest_cpu()`) and picks the faster of scalar vs `matrixmultiply::sgemm`.
- `matmul_t(x, x_rows, x_cols, w: &[f32], w_rows, y, nthreads)` dispatches to sgemm (Blas), candle `Tensor::matmul` (Gpu, x_rows==1), or `tensor::matmul_t_scalar`.
- CLI `--compute auto|blas|gpu|scalar`. Global `OnceLock<Compute>` holds backend + optional candle device.
- **NOTE:** for CPU decode GEMV (memory-bound), the hand-written threaded scalar dot is typically 3-4x *faster* than sgemm — so `auto` often picks `scalar`. That is expected, not a regression. sgemm wins for matrix-matrix (prefill batches).
- `blas-src`/`openblas-src` were rejected: netlib needs a Fortran compiler, OpenBLAS needs a native lib — both break portable CI. `matrixmultiply` is pure Rust with SIMD kernels (the sgemm core used by ndarray).
- Cargo features: `gpu` (candle-core), `gpu-metal` (Apple), `gpu-cuda` (NVIDIA); default build has no GPU deps.

### `src/rng.rs` — SplitMix64 PRNG
- `Rng::new(seed)`, `next_f64() -> f64`, `next_u32()`, private `next_u64()`. Used for sampling.

### `src/regex.rs` — Minimal backtracking regex engine
- `Regex::compile(pattern)` → `Regex` with `match_at` and `pre_tokenize` (isolated matches; used by tokenizer).
- Supports: alternation (`|`), concat, groups `(?:...)`, lookaheads `(?=...)`, `(?!...)`, inline case-insensitive `(?i:...)`, greedy and lazy quantifiers (`* + ? {n} {n,m}`, each with optional trailing `?`), character classes with ranges, escapes, `\s \d \w \S \D \W`, Unicode property classes `\p{L} \p{M} \p{N} \p{P} \p{S} \p{Z} \p{C}` (including inside `[...]`).
- `pre_tokenize()` implements **isolated** split behavior: matches are kept, gaps are dropped. This is exactly what the Qwen3.5 `Split` pre-tokenizer needs.
- Grouped subexpressions backtrack fully (`all_seq_ends`), so patterns like `(?:ab?|c)bb` match correctly.
- 9 tests, including `test_qwen_style` with the exact Qwen3.5 pattern and expected word lists.

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
  - `tools_system_prompt()` builds the system preamble for OpenAI `tools`; `parse_tool_calls()` extracts generated `<tool_call>` blocks into response fields (streaming still emits the XML as content).
  - Generation prompt: `<|im_start|>assistant\n` + `<think>\n` (thinking on) or `<think>\n\n</think>\n\n` (off).
- `enable_thinking` default is model-size-aware: `hidden_size >= 2560` → true (9B/4B), otherwise false (0.8B/2B); an explicit config value wins. `Params` seeds its default from `Engine.default_enable_thinking`.

### `src/server.rs` — Inference engine + API handlers
- `Engine { model: Mutex<Model>, tokenizer, model_name, default_max_tokens, threads, default_enable_thinking }`.
- `GenResult { text, reasoning_content, prompt_tokens, completion_tokens, finish_reason }`.
- `generate()` — core loop:
  1. `model.reset()`, `logits = model.forward(&prompt_ids)` (prefill).
  2. Each step: `sample()` → check stop token → decode incrementally → `on_token(chunk, is_reasoning)` → `logits = model.forward(&[tok])`.
  3. Stops on EOS or matching `stop` strings; strips the matched stop string from output.
  4. `flush_utf8()` buffers incomplete multibyte chars across token boundaries (avoids char-boundary panics).
  5. When `split_reasoning=true`, tokens 248068/248069 (`<think>`/`</think>`) toggle `in_reasoning` and are never emitted; they **are still fed through the model** so state stays aligned. `OutputRouter` handles toggling and buffered text; reasoning text accumulates in `GenResult.reasoning_content`, answer text in `GenResult.text`.
- `run_chat()` / `run_completion()` — render prompt, tokenize, call `generate()`. **Chat uses `add_generation_prompt=true`** (critical: model needs `<think>\n` to start) and `split_reasoning=true`; completions use `split_reasoning=false`.
- `sample()` — repetition penalty (divides logits of already-generated tokens by `repetition_penalty` > 1, HashSet dedup), presence/frequency penalties, `logit_bias`, then temperature (≤0 = argmax), top-k, top-p nucleus sampling. The default path (no top-k, top_p=1) uses an allocation-free two-pass categorical draw; other paths use one candidate vector with partial selection. `generate()` passes completion history into `sample()`.
- Progress logging to stderr: `[load]` per 16 linears + totals in `exl3.rs::load`; `[prompt] processed/total (tok/s)` per 32-token prefill batch; `[decode] N tokens generated (tok/s)` every 16 tokens; `[done]` final summary. Constants `PROMPT_BATCH=32`, `DECODE_BATCH=16` in `server.rs`.
- `handle_chat()` / `handle_completions()` — parse JSON body, build SSE chunks or JSON responses. SSE uses `chat_chunk()` / completion chunk builders; terminates with `data: [DONE]`.
- `Params` includes `enable_thinking`, penalties, `logit_bias`, stop token ids and `include_usage`. `parse_params()` validates all fields and returns 400 on bad input.
- `OutputRouter` owns UTF-8 assembly, stream toggling and stop-string tail buffering; `on_token` returns false to abort generation when the client disconnects. RoPE tables and DeltaNet head scratch are precomputed/reused for throughput.

### `src/exl3.rs` — EXL3 safetensors reader + dequantizer (1556 lines, most research-heavy)
See §5 for the full format. Key items:
- `half_to_f32`, `f32_to_half`, `half_add`, `half_fma` — hand-written fp16 emulation (round-to-nearest-even).
- `decode_3inst`, `decode_mcg`, `decode_mul1` — the three codebooks (16-bit index → fp16).
- `window()` — extracts a 16-bit codeword from the packed MSB-first bitstream with wraparound.
- `tile_perm()` — maps ring position → row-major 16×16 tile offset.
- `unpack_tile_bits()` — unpacks one tile into 256 raw fp16 codebook bits; `unpack_tile()` converts those to f32 via the `half_f32` LUT. The f16 cache stores the bits directly (no f32→f16 round-trip).
- `hadamard128()`, `hadamard_rows()`, `hadamard_cols()` — normalized Sylvester H128 butterflies. `hadamard_rows` uses a stack buffer per 128-column slice (cache-friendly) instead of strided butterflies.
- `read_safetensors()`, `load_all_shards()` — safetensors header parsing, sharding via index JSON or glob. Header arithmetic is checked and malformed metadata (dtype/shape/data_offsets) is rejected; `validate_tile_width()` bounds trellis widths before unpacking. `Shards::compact_to()` drops trellis byte ranges after the linear pass.
- `dequant_linear()` — reference full dequant pipeline for one quantized linear: unpack tiles → H_k → ×suh → H_n → ×svh → transpose to `(n, k)` → `Matrix`. **Used only by tests** as ground truth.
- `QuantLinear` — a quantized linear (trellis + suh/svh) with three storage modes set by `--weight-mode`:
  - `packed`: `matmul(x)` decodes one 16×16 tile at a time into a small scratch (1× on-disk RAM but ~10+ s/token — too slow for interactive use on CPU). O(1) `window()` (64-bit funnel shift), `DECODE_*` LUTs (65536 f32 entries per codebook, avoids log2/powi per codeword), `TILE_PERM` table, and stack buffers keep it from being worse.
  - `f16`: `build_decode_cache()` decodes W_hat once into fp16 (2 bytes/weight, lossless) stored as `(n,k)`; `matmul` uses `half_f32` LUT for a fast dot.
  - `f32` (default): `build_decode_cache_f32()` decodes W_hat once into f32 `(n,k)`; `matmul` is a direct f32 dot. ~0.02 s for a 1024×6144 layer.
  - All modes use the derivation `y[j] = svh[j] * (H_n @ (W_hat^T @ (H_k @ (x ⊙ suh))))[j]` and split output rows across the persistent worker pool (`SharedPtr` newtype / `SendSlice`, ≥65536-element threshold). Verified equivalent by `quantized_matmul_matches_f32_dequant`.
  - `linear()` in model.rs prefers a plain `Matrix` over `QuantLinear` — for tied-embedding models the lm_head uses the plain embedding matrix, skipping the largest decode.
- `Exl3Model::load(path, weight_mode)` — loads config.json + all shards, detects prefix, keeps quantized linears in `linears` (skipping vision) with optional f32/f16 decode cache per `weight_mode`, loads plain tensors as flat vectors, builds plain 2-D matrices (embeddings etc.) in `weights`.
- `Exl3Model { config, weights: HashMap<String, Matrix>, embeddings_f16: HashMap<String, F16Matrix>, linears: HashMap<String, QuantLinear>, vectors: HashMap<String, Vec<f32>> }`. Untied embeddings are read straight from the F16 shard into `embeddings_f16`; tied ones stay in `weights` for `lm_head`.
- `read_signs()` — reads suh/svh (F16) with legacy packed su/sv fallback.

### `src/model.rs` — Qwen3.5 decoder forward (846 lines)
See §6 for the full math. Key items:
- `ModelConfig` — extracted from `config.json` (handles nested `text_config`).
- `Model { config, prefix, weights, linears, vectors, threads, context_size, position, full_states, linear_states }`.
- `w(name)` / `ql(name)` / `v(name)` — resolve unprefixed names by trying `{prefix}{name}`, `{prefix}{name}.weight`, bare `name`, `name.weight`; `ql` also aliases `lm_head` → the quantized head.
- `linear(x, name)` — the universal linear: prefers a packed `QuantLinear` (dequantize-on-matmul), falls back to a plain `Matrix` via `matmul_t`. All projections (q/k/v/o, in_proj_*, out_proj, MLP gate/up/down, lm_head) go through it; only the embedding stays a direct row lookup on a `Matrix`.
- `FullAttnState { k, v }` — KV cache `(max_ctx, num_kv_heads, head_dim)`.
- `LinearAttnState { conv, conv_pos, state }` — conv circular buffer + recurrent state.
- `rms_norm()` — zero-centered: `x * rsqrt(mean(x²)+eps) * (1+w)`.
- `rms_norm_gated()` — `norm(x) * w * silu(gate)`, applied **per value head** (groups of `linear_value_head_dim`); normalizing over the whole concatenated vector was a bug that broke linear attention.
- `forward_full_attn()`, `forward_linear_attn()`, `forward_token()`, `forward(&[u32])`.

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

---

## 8. Known Limitations & Gotchas

- **Memory vs speed:** `--weight-mode` trades RAM for tokens/s. `f32` (default): quantized linears decoded to f32 at load (~4× on-disk, fast). `f16`: ~2× on-disk, near-f32 speed. `packed`: ~1× on-disk but ~10+ s/token on CPU — unusable interactively. On CPU, SIMD on `dot()` would not help the packed path; its bottleneck is codebook decode, not the MACs. Tied-embedding models skip the lm_head decode (plain embedding matrix used directly). Peak RAM during load includes raw shard bytes plus decoded caches (in `f32` mode, several times on-disk); the old "≈ 2× on-disk" claim was inaccurate. Plain 2-D tensors are decoded once (previously stored in both `vectors` and `weights`), trellis byte ranges are compacted away after the linear pass, and untied embedding tables are stored as fp16 with per-row decode. Tied models (0.8B) still keep the ~1 GB f32 embedding for `lm_head`.
- **No GPU:** CPU only; large models are slow.
- **Single-request generation:** `Engine.model` is a mutex; concurrent requests serialize.
- **No full Jinja template engine:** chat template covers text-only common cases; tool-call arguments and advanced Jinja features (e.g. `tojson` of tool definitions) are not implemented.
- **No Unicode NFC normalization:** tokenizer assumes input is already NFC (chat content typically is).
- **No vision inference:** vision tensors are dequantized-skipped and ignored.
- **Rust Send quirk:** see `SharedPtr` note in §3 (tensor.rs). Do not use raw pointers in scoped threads.
- **Bitrate detection gotcha (fixed):** `k_half` must be derived with integer math (`words_per_tile % 16 != 0`). A previous float check used `!k_float.fract().is_normal()` which is ALWAYS true for integer bitrates because `0.0f32.is_normal()` is false — this broke every integer-bpw model (e.g. 8.00 bpw) with a bogus "half bitrate requires mul1 codebook" error. See `bitrate_from_tile_width()` in exl3.rs.
- **tokenizer.json format variants (fixed):** newer `tokenizers` versions omit `model.byte_encoder` (derive with `build_byte_level_map()`), store merges as space-separated strings (`"Ġ Ġ"`) instead of 2-element arrays, rename the sequence key `pre_tokenizers` → `pretokenizers`, and wrap the regex as `{"Regex": "..."}` instead of `{"String": "..."}`. The UnstableLlama EXL3 repos use all four newer forms; `Tokenizer::load()` now handles both old and new. Do not revert these compatibility paths.
- **EXL3 tensor naming / dtypes (fixed):** quantized linears in the UnstableLlama repos use base names **without** `.weight` (e.g. `lm_head.trellis` → matrix key `lm_head`, `...mlp.gate_proj.trellis` → key `...gate_proj`). `Model::w()` now resolves both with and without `.weight`, with an `lm_head` alias and tied-embedding fallback to `embed_tokens.weight`. Many tensors (norms, conv1d, dt_bias) are **BF16**, which `read_f32()` now converts (top 16 bits of f32). Non-quantized 2-D linears (`in_proj_a`/`in_proj_b`, shape `[16, hidden]`) are now loaded as `Matrix` entries — previously only embed/lm_head were, leaving `a`/`b` empty (g=beta=0) and breaking the Gated DeltaNet recurrence. The conv circular buffer holds `kernel-1` history slots (not `kernel`). Without these fixes generation degenerated into token loops.
- **README claims CI covers fmt/clippy** — keep those checks passing (`cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`).
- **Reference files:** `/tmp/qwen35/tokenizer.json` is used by tokenizer tests; `/tmp/exl3src/` has exllamav3 source; `/tmp/trellis_ref.py` is the trellis reference. These are test-time-only and may not exist in a fresh clone (tests skip gracefully if tokenizer file absent).

---

## 9. Tests Quick Map

| Test file/module | What it validates |
|------------------|-------------------|
| `json::tests` | JSON parse/serialize (if present) |
| `regex::tests::test_qwen_style` | Pre-tokenizer regex against ground-truth word lists |
| `tokenizer::tests` | Encode + added-token protection vs real Qwen3.5 tokenizer |
| `chat::tests` | Chat template rendering (4 cases) |
| `exl3::tests::codebook_decode_matches_reference` | All 3 codebooks vs numpy reference |
| `exl3::tests::half_roundtrip` | fp16 conversion correctness |
| `exl3::tests::tile_perm_matches_reference` | Tile permutation vs reference |

Run `cargo test` — 72 pass, 4 ignored on the current state.

---

## 10. Git / CI State

- Branch `main`, latest release tag `v0.3.0`, placeholder author identity.
- `.github/workflows/ci.yml`: build + `cargo test` on push/PR for `ubuntu-latest` (linux x86_64), `ubuntu-24.04-arm` (linux arm64), `macos-latest` (arm64, with `gpu-metal`), `windows-latest` (x86_64). fmt check; clippy `-D warnings` (Linux); uploads artifacts. The macOS build compiles the GPU backend; GPU runtime correctness is not exercised on CI runners.
- `.github/workflows/release.yml`: on `v*` tags, builds all 4 targets (macOS with `gpu-metal`), packages, publishes GitHub Release.
- `.github/workflows/release-gpu.yml`: manually triggered (`workflow_dispatch`, optional tag input). Installs CUDA toolkit via `Jimver/cuda-toolkit@v0.2.21` (12.6.0) on Linux/Windows, builds with `--features gpu-cuda`, attaches to a release by tag. Separate from main CI because stock runners have no nvcc (candle-kernels/cudarc need it; `fallback-latest` only fixes cudarc version detection, not candle-kernels' nvcc requirement).
- `origin` is configured (`https://github.com/fakezeta/exllama.rs.git`); to push: `git push -u origin main`.
