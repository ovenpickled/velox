#!/usr/bin/env bash
# Which part of KV quantization hurts perplexity, and does a finer scale fix it?
# usage: bash scripts/kv_ablation.sh <model_dir> <text_file> [tokens=1024] [windows=4]
# Uses INT8 weights and an FP32 cache with fake quantization (no INT8 layout involved).
set -euo pipefail
M=$1; T=$2; TOK=${3:-1024}; WIN=${4:-4}
run() {  # label, env assignments...
  local label=$1; shift
  local out
  out=$(env "$@" ./build/eval_perplexity --model "$M" --file "$T" --max-tokens "$TOK" --num-windows "$WIN" 2>/dev/null | grep RESULT_JSON)
  printf "%-34s PPL %s\n" "$label" "$(echo "${out#RESULT_JSON }" | python3 -c 'import sys,json;print("%.4f"%json.load(sys.stdin)["ppl"])')"
}
run "no KV quantization (baseline)"          VELOX_KV_FAKEQ=
run "K+V, group 64 (= current INT8 KV)"      VELOX_KV_FAKEQ=kv VELOX_KV_GROUP=64
run "K only, group 64"                       VELOX_KV_FAKEQ=k  VELOX_KV_GROUP=64
run "V only, group 64"                       VELOX_KV_FAKEQ=v  VELOX_KV_GROUP=64
run "K+V, group 32 (like llama.cpp Q8_0)"    VELOX_KV_FAKEQ=kv VELOX_KV_GROUP=32
run "K+V, group 16"                          VELOX_KV_FAKEQ=kv VELOX_KV_GROUP=16