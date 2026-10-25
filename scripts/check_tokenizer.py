#!/usr/bin/env python3
"""Compare the C++ tokenizer with Hugging Face's on strings that are easy to get wrong.

usage: python scripts/check_tokenizer.py <model_dir> [path/to/inference]

Needs `transformers`. The engine side uses `inference --tokenize-only`, which does not load the model.
"""
import json
import subprocess
import sys

STRINGS = [
    "Hello world", "Hello, world!", "The capital of France is", "I have 2024 apples", "price: $12.50",
    "caf\u00e9 au lait", "na\u00efve fa\u00e7ade", "(hello) world", "e-mail me", "don't stop, we're here",
    "Hi!\nNext line", "  two  spaces", "trailing space ", "tabs\tand\nnewlines\r\n", "x=3+4; y = x*2",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30b9\u30c8", "mixed \u65e5\u672c\u8a9e and English", "emoji \U0001F600 test",
    "<|im_start|>user\nHi<|im_end|>\n", "<|endoftext|>", "https://example.com/a_b?c=d&e=1", "def f(x):\n    return x ** 2\n",
    "\u0627\u0644\u0633\u0644\u0627\u0645 \u0639\u0644\u064a\u0643\u0645", "\u00bd cup and 3\u00b2", "A\u00a0B\u3000C",
]


def engine_ids(binary, model_dir, text):
    out = subprocess.run([binary, "--model", model_dir, "--prompt", text, "--tokenize-only"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "engine failed")
    return json.loads(out.stdout.strip().splitlines()[-1])


def compare(hf_encode, binary, model_dir, strings=STRINGS):
    bad = 0
    for s in strings:
        want, got = hf_encode(s), engine_ids(binary, model_dir, s)
        if want != got:
            bad += 1
            print(f"MISMATCH {s!r}\n  hf    : {want}\n  engine: {got}")
    print(f"{len(strings) - bad}/{len(strings)} strings tokenize identically")
    return bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_dir = sys.argv[1]
    binary = sys.argv[2] if len(sys.argv) > 2 else "./build/inference"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    sys.exit(1 if compare(lambda s: tok.encode(s, add_special_tokens=False), binary, model_dir) else 0)
