#!/usr/bin/env bash
# Diagnostic: test the raw completions endpoint (no chat template).
# Usage: ./test_diag.sh [PORT] [MODEL_NAME]
PORT="${1:-8080}"
MODEL="${2:-qwen3.5_8bpw}"
URL="http://localhost:${PORT}/v1"
SEP="============================================================"

completion () {
  local label="$1" prompt="$2"
  echo "$SEP"
  echo "TEST: $label"
  echo "PROMPT: $prompt"
  echo "--- RAW RESPONSE ---"
  curl -sS --max-time 180 "$URL/completions" \
    -H 'Content-Type: application/json' \
    -d "$(cat <<EOF
{
  "model": "$MODEL",
  "prompt": $prompt,
  "temperature": 0,
  "max_tokens": 40
}
EOF
)"
  echo
}

echo "$SEP"
echo "DIAGNOSTIC: raw completions (no chat template)"
echo "Server: $URL"
echo

completion "single word continuation" '"The capital of France is"'
completion "simple arithmetic" '"1 + 1 ="'
completion "list continuation" '"Monday, Tuesday, Wednesday,"'
completion "gibberish check" '"hello world"'

echo "$SEP"
echo "DONE"
