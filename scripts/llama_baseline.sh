#!/usr/bin/env bash
# llama.cpp baseline on the same machine, same text, same context length as the Velox runs.
# usage: LLAMA=~/llama.cpp scripts/llama_baseline.sh <hf_model_dir> <text_file> [ctx=1024] [chunks=4]
set -euo pipefail
MODEL_DIR=$1; TEXT=$2; CTX=${3:-1024}; CHUNKS=${4:-4}
LLAMA=${LLAMA:?set LLAMA to your llama.cpp checkout (built with cmake -B build && cmake --build build -j)}
BIN=$LLAMA/build/bin
mkdir -p gguf
[ -f gguf/model-f16.gguf ] || python3 "$LLAMA/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype f16 --outfile gguf/model-f16.gguf
[ -f gguf/model-q8_0.gguf ] || "$BIN/llama-quantize" gguf/model-f16.gguf gguf/model-q8_0.gguf Q8_0
for M in f16 q8_0; do
  echo "== perplexity $M ctx=$CTX chunks=$CHUNKS"
  "$BIN/llama-perplexity" -m gguf/model-$M.gguf -f "$TEXT" -c "$CTX" --chunks "$CHUNKS" -t "${OMP_NUM_THREADS:-6}" 2>&1 | grep -E "Final estimate|PPL"
  echo "== decode speed $M (64 tokens, empty context and ctx-depth 1024/2048/4096)"
  "$BIN/llama-bench" -m gguf/model-$M.gguf -p 0 -n 64 -d 0,1024,2048,4096 -t "${OMP_NUM_THREADS:-6}" || \
  "$BIN/llama-bench" -m gguf/model-$M.gguf -p 0 -n 64 -t "${OMP_NUM_THREADS:-6}"   # older builds lack -d
done