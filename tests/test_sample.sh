#!/usr/bin/env bash
# Diagnostic with realistic sampling (not greedy). Small models loop under greedy.
# Usage: ./test_sample.sh [PORT] [MODEL_NAME]
PORT="${1:-8080}"
MODEL="${2:-qwen3.5_8bpw}"
URL="http://localhost:${PORT}/v1"
SEP="============================================================"

chat () {
  local label="$1" prompt="$2" thinking="$3"
  echo "$SEP"
  echo "TEST: $label (temp=0.8, top_p=0.9, thinking=$thinking)"
  echo "PROMPT: $prompt"
  echo "--- RESPONSE ---"
  curl -sS --max-time 180 "$URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": $prompt}],
  "temperature": 0.8,
  "top_p": 0.9,
  "max_tokens": 128,
  "enable_thinking": $thinking
}
EOF
)"
  echo
}

echo "$SEP"
echo "DIAGNOSTIC: sampled generation (temp=0.8, top_p=0.9)"
echo

chat "factual" '"The capital of France is"' false
chat "arithmetic" '"What is 2 + 2? Answer with just the number."' false
chat "hello" '"Hello! How are you?"' false

echo "$SEP"
echo "DONE"
