"""maps/optq.m: {test_number: (runs [(fname, i)], weights, {file: {line: mask}})}"""
import json, marshal, pickle, sys
pk, tests_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = pickle.load(open(pk, "rb"))
T = json.load(open(tests_json))["tests"]
num = {t["id"].rsplit("::", 1)[-1]: t["number"] for t in T if t["file"] == "tests/test_optimizer.py"}
res = {}
for name, v in d.items():
    runs = v["runs"]
    tot = sum(r[3] for r in runs) or 1.0
    idx = {}
    for j, (fname, i, lines, dt, ok) in enumerate(runs):
        for f, ln in lines:
            m = idx.setdefault(f, {})
            m[ln] = m.get(ln, 0) | (1 << j)
    res[num[name]] = ([(r[0], r[1]) for r in runs], [r[3] / tot for r in runs], idx)
marshal.dump(res, open(out, "wb"))
print(len(res), "tests", file=sys.stderr)
