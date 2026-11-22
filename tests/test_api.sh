#!/usr/bin/env bash
# Quick API smoke test. Usage: ./test_api.sh [PORT] [MODEL_NAME]
# Run this while exllama-server is running. Prints requests and responses so
# they can be pasted back for diagnosis.
#
# NOTE: the thinking-mode test uses sampling + a larger token budget. A small
# 0.8B model under greedy decoding tends to spend its whole budget inside the
# <think> block (finish_reason "length", empty content) — that is expected
# model behavior, not a server bug. Sampling + more tokens lets it finish the
# reasoning and emit an answer.

PORT="${1:-8080}"
MODEL="${2:-qwen3.5-0.8b}"
URL="http://localhost:${PORT}/v1"
SEP="============================================================"

post_chat () {
  # $1 label, $2 prompt (pre-quoted JSON string), $3 extra JSON fields,
  # $4 temperature (default 0), $5 max_tokens (default 128)
  local label="$1" prompt="$2" extra="$3"
  local temp="${4:-0}" maxtok="${5:-128}"
  echo "$SEP"
  echo "TEST: $label"
  echo "PROMPT: $prompt"
  echo "EXTRA: $extra (temperature=$temp, max_tokens=$maxtok)"
  echo "--- RESPONSE ---"
  curl -sS --max-time 180 "$URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": $prompt}],
  "temperature": $temp,
  "max_tokens": $maxtok,
  $extra
}
EOF
)"
  echo
}

echo "$SEP"
echo "HEALTH:"
curl -sS --max-time 5 "$URL/../health" || curl -sS --max-time 5 "http://localhost:${PORT}/health"
echo
echo "MODELS:"
curl -sS --max-time 5 "$URL/models"
echo

post_chat "greedy short factual" '"The capital of France is"' '"enable_thinking": false' 0 128
# Thinking mode: sample with a generous budget so the model can finish its
# <think> trace and produce an actual answer in "content".
post_chat "thinking with answer" '"Hello"' '"enable_thinking": true' 0.8 512
post_chat "counting" '"Count: 1, 2, 3, 4,"' '"enable_thinking": false' 0 128

echo "$SEP"
echo "DONE"
