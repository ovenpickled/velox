import sys, struct
import numpy as np

def load(p):
    with open(p, "rb") as f:
        n = struct.unpack("i", f.read(4))[0]
        return np.frombuffer(f.read(4 * n), dtype=np.float32)

ref, eng = load(sys.argv[1]), load(sys.argv[2])
assert ref.shape == eng.shape, f"size mismatch {ref.shape} vs {eng.shape}"
d = np.abs(ref - eng)
tr, te = np.argsort(ref)[::-1][:10], np.argsort(eng)[::-1][:10]
print(f"max |diff|        : {d.max():.4f}")
print(f"mean |diff|       : {d.mean():.4f}")
print(f"top-1 match       : {tr[0] == te[0]}  (ref {tr[0]}, engine {te[0]})")
print(f"top-10 overlap    : {len(set(tr) & set(te))}/10")
print(f"top-1 vs top-2 gap: ref {ref[tr[0]] - ref[tr[1]]:.4f}, engine {eng[te[0]] - eng[te[1]]:.4f}")
print(f"ref top-10 ids    : {tr.tolist()}")
print(f"engine top-10 ids : {te.tolist()}")