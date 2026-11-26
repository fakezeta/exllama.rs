# API Test & Diagnostic Scripts

Shell scripts for smoke-testing and diagnosing the running `exllama-server`.
These are **manual** tests — start the server first, then run a script from this
directory. They are not part of `cargo test` (which runs the Rust unit tests).
The `golden/` directory holds committed chat-prompt fixtures asserted by
`chat::tests::prompts_match_committed_golden_files` (no downloaded files needed).

`cargo test` itself is fully self-contained: the dequantization/matmul
equivalence guarantee is covered by `exl3::synthetic_tests` (in-memory EXL3
fixture, all codebooks and weight modes) and the decoder math by hand-computed
tests in `model::tests`. Tests that require real-model `/tmp` files (e.g.
`/tmp/qwen35/tokenizer.json`, `/tmp/unstable_layer0.bin`) are marked `#[ignore]`
with a reason; run them explicitly with `cargo test -- --ignored` when the
fixtures are present.

## Prerequisites

- `exllama-server` is built and running (see the root [README.md](../README.md)).
- `curl` and (for `test_prompts.sh`) `python3` are available.
- Scripts default to port `8080`. Most accept optional arguments:

  ```bash
  ./<script>.sh [PORT] [MODEL_NAME]
  ```

  `MODEL_NAME` must match the `--name` passed to the server: a non-empty
  `model` field that differs from `--name` is rejected with HTTP 400. Keeping
  them consistent avoids confusion with other clients.
- The server's `--weight-mode` affects speed, not output: `f32` (default) and
  `f16` are fast and produce identical answers; `packed` uses the least RAM but
  runs at ~3.5 s/token on CPU, so long requests exceed these scripts' timeouts.
  Use `f32`/`f16` for these tests.

All scripts print the exact prompt and raw response for each request, so output
can be pasted back to a developer for diagnosis.

## Scripts

### `test_api.sh` — full API smoke test

The main end-to-end check. Exercises health/models endpoints plus three chat
completions:

| Test | Prompt | `enable_thinking` | Sampling |
|------|--------|-------------------|----------|
| greedy short factual | `"The capital of France is"` | `false` | greedy, 128 tokens |
| thinking with answer | `"Hello"` | `true` | `temperature=0.8`, `repetition_penalty=1.1`, 512 tokens |
| counting | `"Count: 1, 2, 3, 4,"` | `false` | greedy, 128 tokens |

The script echoes the target URL and the `PORT`/`MODEL` args it is using — make
sure they match the server's `--port` and `--name`. If you see the wrong model
id in `/v1/models` or no `[prompt]`/`[decode]` progress lines, the requests are
going to a different (older) server instance than the one whose logs you are
watching.

The thinking test deliberately uses sampling and a larger token budget: a small
0.8B model under greedy decoding tends to spend its entire budget inside the
`<think>` block (`finish_reason: "length"`, empty `content`), which is expected
model behavior. Sampling + more tokens lets it finish reasoning and emit an
answer in `content` with `reasoning_content` holding the trace.

First hits `/health` and `/v1/models` to confirm the server is up. Use this as
the first check after a build or a model change — coherent answers here (e.g.
"Paris." with `finish_reason: "stop"`) mean the whole stack works.

```bash
./test_api.sh 8080 my-model
```

### `test_diag.sh` — raw completions (no chat template)

Tests the `/v1/completions` endpoint with raw text prompts, bypassing the chat
template entirely. Greedy (`temperature: 0`), 40 tokens each:

- `"The capital of France is"` — continuation
- `"1 + 1 ="` — arithmetic
- `"Monday, Tuesday, Wednesday,"` — list continuation
- `"hello world"` — gibberish check

Useful to isolate whether a problem is in the model forward pass (both
endpoints affected) or the chat template (only chat affected). Historically,
list continuation staying coherent while chat failed pointed to the template.

```bash
./test_diag.sh 8080 my-model
```

### `test_sample.sh` — sampled generation

Same idea as `test_api.sh` but with **realistic sampling** (`temperature: 0.8`,
`top_p: 0.9`) and `enable_thinking: false`. Small models (e.g. 0.8B) tend to
loop degenerately under greedy decoding, so this distinguishes "model quality
/ greedy degeneration" from a real forward-pass bug.

Prompts: a factual question, an arithmetic question ("just the number"), and a
greeting.

```bash
./test_sample.sh 8080 my-model
```

### `test_prompts.sh` — assistant-prefix format probe

Sends the same factual question through `/v1/completions` with four different
assistant prefixes (`temperature: 0.6`, `top_p: 0.9`) to discover which prompt
format the model responds to:

1. **RAW question** — no chat structure
2. **ASSISTANT plain prefix** — `<|im_start|>assistant\n` with no think block
3. **ASSISTANT + think open** — ends with `<think>\n`
4. **ASSISTANT + empty think** — ends with `<think>\n\n</think>\n\n` (what our
   chat template produces for `enable_thinking: false`)

Requires `python3` to pretty-print the `text` field. This was used to confirm
that the Qwen3.5 `<think>` handling in `chat.rs` matches the model's template.

```bash
./test_prompts.sh 8080 my-model
```

## Debugging tips

- If responses are garbled/repetitive, run `test_diag.sh` (raw completions)
  first to isolate the model from the chat template.
- To inspect logits directly, start the server with `EXLLAMA_DEBUG=1`; the first
  five decode steps print logit min/max/nan/inf and the top-6 tokens with text.
- If the server returns "Empty reply" / connection errors, check the server
  stderr for panics (e.g. UTF-8 char-boundary issues have been fixed, but the
  log will confirm any new one).
- Keep script arguments consistent with the server's `--name` and `--port`.
