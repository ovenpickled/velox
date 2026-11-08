#!/usr/bin/env python3
"""Perplexity matrix: {English, Japanese} x {FP32 weights, INT8 weights} x {FP32 KV, INT8 KV}.

usage: python scripts/run_eval.py --model <dir> --en data/...wiki.test.raw.nfc --ja data/ja_test.raw.nfc \
           [--tokens 1024] [--windows 4] [--binary ./build/eval_perplexity]
Parses the RESULT_JSON line printed by eval_perplexity and prints a markdown table.
"""
import argparse, json, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--en", required=True)
ap.add_argument("--ja", required=True)
ap.add_argument("--tokens", type=int, default=1024)
ap.add_argument("--windows", type=int, default=4)
ap.add_argument("--binary", default="./build/eval_perplexity")
a = ap.parse_args()

rows = []
for lang, path in (("English", a.en), ("Japanese", a.ja)):
    for weights in ("fp32", "int8"):
        for kv in ("fp32", "int8"):
            cmd = [a.binary, "--model", a.model, "--file", path, "--max-tokens", str(a.tokens),
                   "--num-windows", str(a.windows), "--kv-cache-type", kv]
            if weights == "fp32":
                cmd.append("--fp32")
            print("running:", " ".join(cmd), file=sys.stderr)
            out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
            if out.returncode != 0 or not line:
                sys.exit(f"failed: {' '.join(cmd)}\n{out.stderr[-500:]}")
            rows.append((lang, json.loads(line[len("RESULT_JSON "):])))

print("\n| Language | Weights | KV cache | Tokens | PPL | Bits/byte | tok/s |")
print("|---|---|---|---:|---:|---:|---:|")
for lang, r in rows:
    print(f"| {lang} | {r['weights']} | {r['kv']} | {r['tokens']} | {r['ppl']:.3f} | {r['bpb']:.4f} | {r['tok_s']:.1f} |")