#!/usr/bin/env bash
# Test different assistant-prefix formats via raw completions to find what this model expects.
PORT="${1:-8080}"; MODEL="${2:-qwen3.5_8bpw}"; URL="http://localhost:${PORT}/v1"
SEP="============================================================"
comp () {
  echo "$SEP"; echo "$1"
  curl -sS --max-time 120 "$URL/completions" -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{"model":"$MODEL","prompt":$2,"temperature":0.6,"top_p":0.9,"max_tokens":32}
EOF
)" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "(parse fail)"
  echo
}
# raw question, no template
comp "RAW question" '"The capital of France is"'
# chat-style with im_start, plain assistant prefix (no think block)
comp "ASSISTANT plain prefix" '"<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"'
# chat-style with <think> open
comp "ASSISTANT + think open" '"<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n<think>\n"'
# chat-style with empty closed think block
comp "ASSISTANT + empty think" '"<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"'
echo "$SEP"; echo DONE
