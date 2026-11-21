"""Compile per-query TPC coverage into maps/tpcq.m: {test_number: (weights, {file: {line: mask}})}."""
import json, marshal, pickle, sys
pk, tests_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = pickle.load(open(pk, "rb"))
T = json.load(open(tests_json))["tests"]
num = {t["id"].rsplit("::", 1)[-1]: t["number"] for t in T if t["file"] == "tests/test_executor.py"}
res = {}
for name, qs in d.items():
    tot = sum(x[1] for x in qs) or 1.0
    weights = [x[1] / tot for x in qs]
    idx = {}
    for i, (lines, dt, ok) in enumerate(qs):
        for f, ln in lines:
            idx.setdefault(f, {})
            idx[f][ln] = idx[f].get(ln, 0) | (1 << i)
    res[num[name]] = (weights, idx)
marshal.dump(res, open(out, "wb"))
print({k: len(v[0]) for k, v in res.items()}, file=sys.stderr)
