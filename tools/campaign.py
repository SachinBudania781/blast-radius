"""Calibration: synthetic mutants on lines covered by expensive tests.

For each mutant run every candidate test (cheap and expensive) with the selector's
runner and record the outcomes. Output: one JSON line per mutant.
"""

import difflib
import io
import json
import os
import random
import re
import sys
import time
import tokenize

sys.path.insert(0, "/home/claude/blast-radius/app")
os.environ.setdefault("BR_SRC", "/home/claude/sqlglot")
os.environ.setdefault("BR_DATA", "/home/claude/kit/sample")
os.environ.setdefault("BR_MAPS", "/home/claude/blast-radius/maps")
os.environ.setdefault("BR_TMP", "/tmp/brcamp")
import selector  # noqa: E402
selector.DEADLINE = time.time() + 1e7

SRC = os.environ["BR_SRC"]
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
OUT = sys.argv[1]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
TARGET = sys.argv[4] if len(sys.argv) > 4 else "expensive"
MODE = sys.argv[5] if len(sys.argv) > 5 else "all"

tests = json.load(open(os.path.join(os.environ["BR_DATA"], "tests.json")))["tests"]
ms = [round(t["seconds"] * 1000) for t in tests]
meta = []
for t in tests:
    f, cls, name = t["id"].split("::")
    meta.append((t["number"], t["file"], cls, name))
maps = selector.Maps(len(tests))

# lines covered per expensive test (from narrow tables we can't invert cheaply) -> use raw pickles
import glob  # noqa: E402
import pickle  # noqa: E402

ID = {t["id"]: t["number"] for t in tests}
cov_by_test = {}
for f in glob.glob("/home/claude/brmap2/map-gw*.pkl"):
    d = pickle.load(open(f, "rb"))
    for tid, r in d["results"].items():
        if tid in ID:
            cov_by_test[ID[tid]] = [fl for fl in r["lines"] if fl[0].startswith("sqlglot/")]

if TARGET == "expensive":
    targets = [i for i in range(len(tests)) if ms[i] >= 1000]
elif TARGET == "big":
    targets = [i for i in range(len(tests)) if ms[i] >= 6000]
elif TARGET == "executor":
    targets = [i for i in range(len(tests)) if tests[i]["id"].endswith("::test_execute_tpcds")]
    for k in list(cov_by_test):
        cov_by_test[k] = [fl for fl in cov_by_test[k] if fl[0].startswith("sqlglot/executor/") or fl[0] == "sqlglot/planner.py"]
else:
    targets = list(range(len(tests)))

REL = [("==", "!="), ("!=", "=="), ("<", "<="), ("<=", "<"), (">", ">="), (">=", ">"), ("is", "=="),
       ("and", "or"), ("or", "and"), ("True", "False"), ("False", "True"), ("+", "-"), ("-", "+"),
       ("in", "not in")]


def mutate_line(line):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(line.strip() + "\n").readline))
    except Exception:
        return None
    opts = []
    for tk in toks:
        if tk.type in (tokenize.OP, tokenize.NAME):
            for a, b in REL:
                if tk.string == a:
                    opts.append((tk, b))
        if tk.type == tokenize.NUMBER and re.fullmatch(r"\d+", tk.string):
            opts.append((tk, str(int(tk.string) + 1)))
        if tk.type == tokenize.NAME and tk.string == "not":
            opts.append((tk, ""))
    if not opts:
        return None
    tk, rep = random.choice(opts)
    indent = len(line) - len(line.lstrip())
    s = line.strip()
    c0, c1 = tk.start[1], tk.end[1]
    if tk.start[0] != 1:
        return None
    new = s[:c0] + rep + s[c1:]
    if rep == "":
        new = (s[:c0] + s[c1:]).replace("  ", " ")
    return line[:indent] + new


def make_patch(rel, ln):
    src = open(os.path.join(SRC, rel), encoding="utf-8").read().split("\n")
    old = src[ln - 1]
    new = mutate_line(old)
    if new is None or new == old:
        return None
    mod = list(src)
    mod[ln - 1] = new
    try:
        compile("\n".join(mod), rel, "exec")
    except Exception:
        return None
    a = [x + "\n" for x in src[:-1]] + ([src[-1]] if src[-1] else [])
    b = [x + "\n" for x in mod[:-1]] + ([mod[-1]] if mod[-1] else [])
    diff = "".join(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", n=3))
    return f"diff --git a/{rel} b/{rel}\n" + diff, old.strip(), new.strip()


done = 0
seen = set()
with open(OUT, "a") as out:
    while done < N:
        e = random.choice(targets)
        lines = cov_by_test.get(e)
        if not lines:
            continue
        rel, ln = random.choice(lines)
        if (rel, ln) in seen:
            continue
        seen.add((rel, ln))
        r = make_patch(rel, ln)
        if r is None:
            continue
        patch, old, new = r
        c = selector.analyze(f"m{done}", patch.encode(), maps, len(tests), None)
        if c.unknown:
            continue
        cand = c.cands - c.always
        t0 = time.time()
        if MODE == "dead":
            cheap = {t for t in cand if ms[t] < 1000}
            exp_ = cand - cheap
            if not exp_ or not cheap:
                continue
            c.cheap, c.exp = set(cheap), set()
            c.early_blind = {next(iter(exp_))}  # makes the runner stop at the first cheap failure
            c.sure = set()
            selector.run_jobs([c], meta, ms, maps)
            if getattr(c, "early_stopped", False) or (c.result or set()) & cheap:
                rec = {"rel": rel, "ln": ln, "old": old, "new": new, "live": True, "cands": sorted(cand), "failed": None}
                out.write(json.dumps(rec) + "\n"); out.flush(); done += 1
                print(done, "live", rel, ln, file=sys.stderr, flush=True)
                continue
            c2 = selector.analyze(f"m{done}b", patch.encode(), maps, len(tests), None)
            c2.cheap, c2.exp, c2.sure = set(exp_), set(), set()
            c2.early_blind = None
            selector.run_jobs([c2], meta, ms, maps)
            res = (c2.result or set()) | (c.result or set())
        else:
            c.cheap = set(cand)
            c.exp = set()
            c.always = set(c.always)
            # run everything in one phase
            selector.run_jobs([c], meta, ms, maps)
            res = c.result if c.result is not None else set()
        rec = {
            "rel": rel, "ln": ln, "old": old, "new": new, "target": e,
            "cands": sorted(cand), "failed": sorted(res), "import_failed": sorted(getattr(c, "import_failed", [])),
            "cpu": getattr(c, "cpu", None), "wall": round(time.time() - t0, 2),
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        done += 1
        print(done, rel, ln, len(cand), len(res), rec["cpu"], file=sys.stderr, flush=True)
