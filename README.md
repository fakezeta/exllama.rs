# exllama.rs

A minimal, **dependency-free** Rust inference server for [ExLlamaV3 (EXL3)](https://github.com/turboderp-org/exllamav3) quantized models, focused on the `Qwen3_5ForConditionalGeneration` text decoder (text-only). It serves an OpenAI-compatible HTTP API with streaming SSE support.

Everything is implemented with the Rust standard library only — no crates, no CUDA, no Python at runtime.

## Features

- **Minimal dependencies** — hand-written JSON parser/serializer, HTTP server, BPE tokenizer, regex engine, PRNG, and EXL3 dequantizer. Linear algebra uses `matrixmultiply` (SIMD BLAS kernel) and optionally `candle` for GPU; see [Compute backends](#compute-backends).
- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/health`, with streaming SSE (`stream: true`).
- **Qwen3.5 text decoder** — full attention (GQA + partial RoPE) and linear attention (Gated DeltaNet), SwiGLU MLP, zero-centered RMSNorm.
- **Configurable context size** and thread count from the CLI.
- **Thinking mode** — Qwen3.5 `<think>` blocks, configurable via `enable_thinking` in chat requests.
- **Cross-platform** — Linux, macOS, and Windows (see [CI](#continuous-integration)).

## Requirements

- Rust 1.70+ (edition 2021). No external toolchain or Python runtime needed. Default build uses only pure-Rust crates (no system BLAS/GPU SDK).
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
-h, --help                Show help
```

> [!NOTE]
> **`--weight-mode`** trades memory for speed:
> - `f32` (default) — decode quantized linears into f32 at load. Fast inference (~0.02 s per large layer matmul), but ~4× on-disk RAM. All EXL3 math is still applied correctly.
> - `f16` — decode into fp16 (codebook outputs are natively fp16, so lossless). ~2× on-disk, slightly slower than f32.
> - `packed` — keep weights in their on-disk form and decode tiles on every matmul. ~1× on-disk RAM but **very slow** (~10+ s/token on CPU) — use only when memory is the hard constraint.
>
> For tied-embedding models (`tie_word_embeddings=true`, e.g. Qwen3.5-0.8B), `lm_head` uses the plain embedding matrix directly, avoiding a decode of the largest weight. Peak RAM during load is briefly ~2× on-disk while shards are copied.

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

`repetition_penalty` (> 1) divides the logit of already-generated tokens, which
discourages repetition in both greedy and sampled decoding.

Streaming: add `"stream": true` to receive SSE chunks terminated by `data: [DONE]`.

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

Supported sampling parameters: `temperature`, `top_p`, `top_k`, `max_tokens` / `max_completion_tokens`, `seed`, `stop`, `repetition_penalty`, `enable_thinking`.

## How it works

1. **Model loading** — EXL3 files are standard safetensors. Each quantized linear stores a packed tail-biting trellis plus input/output sign vectors (`suh`/`svh`). Depending on `--weight-mode`, W_hat is decoded once at load into an f32/f16 cache (default f32) or kept packed.
2. **Quantized matmul** — the normalized Sylvester-Hadamard transforms are applied to the *input* and *output* vectors (not the weight matrix), so a linear needs only its W_hat in the rotated basis. `--weight-mode` decides how W_hat is stored: decoded to f32/f16 at load (fast) or kept packed and decoded per matmul (slow, minimum RAM). All three paths are verified equivalent to full reconstruction by `quantized_matmul_matches_f32_dequant`. A few O(1) optimizations (codebook LUTs, precomputed tile permutation, stack buffers) keep the packed path from being unusable.
3. **Tokenizer** — byte-level BPE with the Qwen3.5 pre-tokenizer regex, added-token protection (so `<|im_start|>` and `<think>` remain single tokens), and byte fallback.
4. **Inference** — one OS thread per HTTP connection; the model is guarded by a mutex. Prefill processes the prompt in 32-token batches; decoding emits one token at a time with KV cache (full attention) and recurrent state (linear attention). Matmul optionally splits work across threads. The server prints progress to stderr: `[load]` during model load, `[prompt] X/N tokens (tok/s)` per prefill batch, and `[decode] N tokens generated (tok/s)` every 16 tokens, with a final `[done]` summary.

## Testing

```bash
cargo test
```

Tests cover JSON parsing, regex/pre-tokenizer behavior, chat-template rendering, tokenizer encoding against the real Qwen3.5 tokenizer, and EXL3 codebook/tile-layout correctness against the reference implementation.

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
