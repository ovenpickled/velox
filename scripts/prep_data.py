#!/usr/bin/env python3
"""NFC-normalize the evaluation corpora so the C++ tokenizer (no NFC step) sees what Hugging Face sees.

usage: python scripts/prep_data.py data/wikitext-2-raw/wiki.test.raw data/ja_test.raw
Writes <name>.nfc next to each input and reports how many characters NFC changed.
"""
import sys, unicodedata
for path in sys.argv[1:]:
    t = open(path, encoding="utf-8").read()
    n = unicodedata.normalize("NFC", t)
    open(path + ".nfc", "w", encoding="utf-8").write(n)
    print(f"{path}: {len(t)} chars -> {len(n)} chars after NFC ({len(t) - len(n)} removed); wrote {path}.nfc")