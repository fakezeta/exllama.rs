# exllama.rs

A minimal Rust inference server for [ExLlamaV3 (EXL3)](https://github.com/turboderp-org/exllamav3) quantized models, focused on the `Qwen3_5ForConditionalGeneration` text decoder (text-only). It serves an OpenAI-compatible HTTP API with streaming SSE support.

The JSON parser, HTTP server, tokenizer, regex engine, PRNG and EXL3 dequantizer are hand-written with the standard library only. Linear algebra uses the pure-Rust `matrixmultiply` crate (optionally `candle` for GPU); there is no CUDA toolchain or Python dependency at runtime.

## Features

- **Hand-written core** — JSON parser/serializer, HTTP server, BPE tokenizer, regex engine, PRNG, and EXL3 dequantizer need no crates. Linear algebra uses `matrixmultiply` (SIMD BLAS kernel) and optionally `candle` for GPU; see [Compute backends](#compute-backends).
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, with streaming SSE (`stream: true`).
- **Qwen3.5 text decoder** — full attention (GQA + partial RoPE) and linear attention (Gated DeltaNet), SwiGLU MLP, zero-centered RMSNorm.
- **Configurable context size** and thread count from the CLI.
- **Thinking mode** — Qwen3.5 `<think>` blocks, configurable via `enable_thinking` in chat requests (default follows model size: off for 0.8B/2B, on for 4B/9B).
- **Tool calling** — OpenAI `tools` / assistant `tool_calls` are rendered into Qwen3.5's XML format; generated tool calls are parsed back into `tool_calls` response fields. Streaming emits indexed `tool_calls` deltas (name + argument fragments) and withholds the XML from `content`.
- **Cross-platform** — Linux, macOS, and Windows (see [CI](#continuous-integration)).

## Requirements

- Rust 1.70+ (edition 2021). No external toolchain or Python runtime needed. Default build uses only pure-Rust crates (no system BLAS/GPU SDK). CI builds and tests on Linux (x86_64 + ARM64), macOS ARM64 and Windows x86_64 on every push/PR.
- An EXL3-quantized Qwen3.5 model directory containing:
  - `config.json`
  - `model.safetensors` or sharded `model-*.safetensors` (+ `model.safetensors.index.json`)
  - `tokenizer.json`

> [!IMPORTANT]
> This server is **focused on Qwen3.5** (`qwen3_5` / `Qwen3_5ForConditionalGeneration` and `Qwen3_5ForCausalLM`). Other architectures are not supported. The vision tower of the multimodal checkpoint is ignored; text-only inference works.

## Build

```bash
cargo build --release
```

The binary is produced at `target/release/exllama-server`.

## Compute backends

Linear algebra (`y = W @ x`) runs through one of three backends, selected at
startup. `--compute auto` (default) picks the best one available:

| Backend | Crate | Build | Notes |
|---------|-------|-------|-------|
| `cpu-blas` | `matrixmultiply` `sgemm` | default | Pure Rust SIMD BLAS kernel, no native libs. |
| `scalar` | (built-in) | default | Hand-written threaded dot products. Often fastest for memory-bound decode (GEMV); the auto probe benchmarks both CPU paths and picks the winner. |
| `gpu` | `candle-core` | opt-in | CUDA or Metal. Much faster for large matmuls; falls back to CPU if no device is present. |

Build with GPU support (off by default so CI stays portable):
```bash
# macOS / Apple Silicon (Metal)
cargo build --release --features gpu-metal
# NVIDIA Linux/Windows (requires the CUDA toolkit)
cargo build --release --features gpu-cuda
```
Then select at runtime: `--compute auto` (default) or force one with
`--compute blas|scalar|gpu`. The chosen backend is printed at startup:
```
[compute] backend=cpu-blas (matrixmultiply sgemm)  (available: cpu-blas )
```

For this CPU-only decode workload, the auto probe usually chooses `scalar`
because GEMV is memory-bound and the hand-written threaded dot beats sgemm;
prefill batches and GPU-backed runs benefit most from the BLAS/GPU paths.

## Usage

```bash
./target/release/exllama-server \
    --model /path/to/model-dir \
    --tokenizer /path/to/model-dir/tokenizer.json \
    --context-size 4096 \
    --port 8080
```

By default, `--tokenizer` is looked up next to the model file. Run `--help` for all options:

```
-m, --model <PATH>        Path to the model directory (required)
-c, --context-size <N>    Maximum context size in tokens (default 4096)
-p, --port <PORT>         Listen port (default 8080)
    --host <HOST>         Bind address (default 0.0.0.0)
    --tokenizer <PATH>    Path to tokenizer.json
    --max-tokens <N>      Default max generation tokens (default 512)
    --threads <N>         Worker threads for matmul (default: all CPUs)
    --name <NAME>         Model name exposed by the API
    --weight-mode <MODE>  f32 (fast, default) | f16 | packed (lowest RAM, slow)
    --compute <BACKEND>   auto (best, default) | blas | gpu | scalar
    --instances <N>       Concurrent decoding instances sharing weights (default 1)
-h, --help                Show help
```

> [!NOTE]
> **`--weight-mode`** trades memory for speed:
> - `f32` (default) — decode quantized linears into f32 at load. Fast inference (~0.02 s per large layer matmul), but ~4× on-disk RAM. All EXL3 math is still applied correctly.
> - `f16` — decode into fp16 (codebook outputs are natively fp16, so lossless). ~2× on-disk, slightly slower than f32.
> - `packed` — keep weights in their on-disk form and decode tiles on every matmul. ~1× on-disk RAM but **very slow** (~10+ s/token on CPU) — use only when memory is the hard constraint.
>
> All plain embeddings are stored as fp16 code bits (2 bytes/weight) and decoded per row. For tied models (`tie_word_embeddings=true`, e.g. Qwen3.5-0.8B) `lm_head` uses that same fp16 table through a threaded matmul, so no f32 vocab×hidden matrix is kept. Peak RAM during load includes raw shard bytes plus decoded caches (in `f32` mode, several times on-disk); trellis bytes are compacted away before plain tensors are decoded. The full-attention KV cache is also fp16, halving context RAM.

## API

### Chat completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "enable_thinking": true
  }'
```

`enable_thinking` defaults to `true` (matching Qwen3.5-9B/4B). Set it to `false` to emit an empty `<think></think>` block (matching 0.8B/2B behavior).

When thinking is enabled, the reasoning trace is **not** mixed into `content`. It is returned in a separate `reasoning_content` field on the message (and as a `reasoning_content` delta when streaming), matching vLLM/SGLang behavior:

```json
"message": {
  "role": "assistant",
  "reasoning_content": "Okay, the user said hello... I should respond warmly.",
  "content": "Hello! How can I help you today?"
}
```

The `reasoning_content` field is omitted when there is no thinking trace.

> [!TIP]
> Small models (e.g. 0.8B) can loop inside `<think>` under greedy decoding
> (`temperature: 0`) — repeating the same paragraph until `max_tokens` is hit,
> leaving `content` empty. This is model behavior, not a server bug. Use
> sampling (`temperature: 0.8`), a larger token budget, and/or
> `repetition_penalty: 1.1` to break the loop. The `tests/test_api.sh` thinking
> test does exactly this.

`repetition_penalty` must be a finite number `> 0` (values `> 1` divide the
logit of already-generated tokens, discouraging repetition in both greedy and
sampled decoding).

Streaming: add `"stream": true` to receive SSE chunks terminated by `data: [DONE]`.
Each token's `logprobs` delta follows its content delta; with
`stream_options.include_usage`, both endpoints finish with a final chunk whose
`choices` array is empty and which carries aggregate `usage`. Tool calls are
streamed as indexed `tool_calls` deltas. A single generation is capped at 300 s
of wall-clock time so a stalled request cannot block other clients.

#### Logprobs, multiple choices, and response format

```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "logprobs": true,
  "top_logprobs": 2,
  "n": 2,
  "seed": 42,
  "response_format": {"type": "json_object"}
}
```

- `logprobs: true` plus `top_logprobs` (0–20) adds `choices[].logprobs.content[]`
  with one entry per generated token (`token`, `token_id`, `logprob`, and the top
  alternatives). In streaming mode each token is emitted as a separate
  `logprobs` delta. For `/v1/completions` the parameter is an integer
  (`"logprobs": 2`) and the response uses the parallel-array shape
  (`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`).
  Probabilities come from the logits after repetition/presence/frequency/bias
  penalties but before temperature/top-k/top-p sampling transforms. Reasoning
  delimiters (`<think>` / `</think>`) and stop tokens are excluded, matching
  `completion_tokens`; delimiters still consume a forward step so the model
  state stays aligned.
- `n` (1–128) returns that many choices. All choices **share one prefill**: the
  post-prompt state is snapshotted and restored for each choice, so only decoding
  is repeated (generation is still sequential under the model mutex). Each choice
  gets its own RNG stream derived from `seed`. With greedy decoding
  (`temperature: 0`) all choices are identical, as with OpenAI. SSE chunks carry
  the choice `index`, and usage is the sum across choices. At most 16 requests may
  queue at once; further callers receive HTTP 429.
- `user` is accepted and validated but otherwise ignored (OpenAI uses it for
  abuse monitoring).
- `response_format` accepts `{"type": "text"}` (default) and
  `{"type": "json_object"}`. JSON mode adds a "respond with a single JSON object"
  hint to the prompt; it is **best-effort** — there is no constrained decoding or
  output validation, and `json_schema` is rejected with a 400.

### Completions

```bash
curl http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Once upon a time", "max_tokens": 64, "stream": false}'
```

### Models / health

```
GET /v1/models
GET /health   (or /)
```

Supported request parameters: `temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `seed`, `stop`, `stop_token_ids`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `stream`, `stream_options.include_usage`, `enable_thinking`, `tools`, `logprobs`, `top_logprobs` (chat only), `n`, `user`, `response_format`. Assistant messages may carry `reasoning_content` and `tool_calls`; `content` may be a string or an array of text parts. The `model` field is validated against the server's `--name`. Unknown fields are ignored.

## How it works

1. **Model loading** — EXL3 files are standard safetensors. Each quantized linear stores a packed tail-biting trellis plus input/output sign vectors (`suh`/`svh`). Depending on `--weight-mode`, W_hat is decoded once at load into an f32/f16 cache (default f32) or kept packed.
2. **Quantized matmul** — the normalized Sylvester-Hadamard transforms are applied to the *input* and *output* vectors (not the weight matrix), so a linear needs only its W_hat in the rotated basis. `--weight-mode` decides how W_hat is stored: decoded to f32/f16 at load (fast) or kept packed and decoded per matmul (slow, minimum RAM). All three paths are verified equivalent to full reconstruction by `quantized_matmul_matches_f32_dequant`. A few O(1) optimizations (codebook LUTs, precomputed tile permutation, stack buffers) keep the packed path from being unusable.
3. **Tokenizer** — byte-level BPE with the Qwen3.5 pre-tokenizer regex, added-token protection (so `<|im_start|>` and `<think>` remain single tokens), and byte fallback.
4. **Inference** — one OS thread per HTTP connection; bounded admission returns 429 past 16 queued requests. `--instances N` runs N decoding instances that share the loaded weights (an Arc core) but keep independent KV/recurrent state, so concurrent requests need not serialize. Prefill runs once per request and its state is reused across `n` choices. Prefill processes the prompt one token at a time (logged in 32-token groups); decoding emits one token at a time with an fp16 KV cache (full attention, halved context RAM) and recurrent state (linear attention). Matmul work is split across a persistent pool of worker threads (sized by `--threads`) instead of spawning threads per call. The server prints progress to stderr: `[load]` during model load, `[prompt] X/N tokens (tok/s)` per prefill batch, and `[decode] N tokens generated (tok/s)` every 16 tokens, with a final `[done]` summary.

## Testing

```bash
cargo test
```

Tests cover JSON parsing, regex/pre-tokenizer behavior, chat-template rendering (including committed golden prompts in `tests/golden/`), tool-call parsing, EXL3 codebook/tile-layout correctness against the reference implementation, sampling, logprobs, `n`-choice handling, parameter validation, end-to-end HTTP/SSE behavior against a tiny synthetic model, and deterministic dependency-free fuzz smoke tests for the JSON/HTTP/regex/tokenizer/API parsers (`src/fuzz.rs`). Two suites need no model download and always run: `exl3::synthetic_tests` verifies all three `--weight-mode` paths against full dequantization for every EXL3 codebook using an in-memory fixture, and `model::tests` holds hand-computed golden tests for the decoder math (RMSNorm, RoPE, q_proj interleaving, GQA, DeltaNet). Tests that need `/tmp` reference files (real tokenizer / real EXL3 layer) are explicitly `#[ignore]`d with reasons — run them with `cargo test -- --ignored` when those files exist. Current totals: 107 passed, 11 ignored. For throughput benchmarking with a real model, see `bench_generation_throughput` and `bench_n_choices_prefill_reuse` in `src/server.rs`.

For manual end-to-end checks against a **running server**, see [`tests/README.md`](tests/README.md). It contains smoke-test and diagnostic shell scripts (`test_api.sh`, `test_diag.sh`, `test_sample.sh`, `test_prompts.sh`) that exercise the OpenAI-compatible endpoints with curl.

## Continuous integration

GitHub Actions builds and tests on every push and pull request for:

| OS / Arch | Runner | Target | Features |
|-----------|--------|--------|----------|
| Linux x86_64 | `ubuntu-latest` | `x86_64-unknown-linux-gnu` | — |
| Linux ARM64 | `ubuntu-24.04-arm` | `aarch64-unknown-linux-gnu` | — |
| macOS ARM64 | `macos-latest` | `aarch64-apple-darwin` | `gpu-metal` |
| Windows x86_64 | `windows-latest` | `x86_64-pc-windows-msvc` | — |

The macOS build enables `gpu-metal` (Metal ships with macOS, so no extra setup). On tagged releases (`v*`), the **Release** workflow builds optimized binaries for all four targets and attaches them to a GitHub Release.

**GPU (CUDA) binaries:** the **Release (GPU builds)** workflow (`.github/workflows/release-gpu.yml`) is manually triggered via `workflow_dispatch`. It installs the CUDA toolkit (nvcc) and builds Linux/Windows binaries with `--features gpu-cuda`, optionally attaching them to an existing release by tag. It's separate because stock GitHub runners lack nvcc and CUDA adds ~2-3 GB / several minutes to CI.

## License

MIT — see [LICENSE](LICENSE).
