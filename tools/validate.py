"""Recall check: random sample-like mutants anywhere in sqlglot; run ALL tests; any failing
test outside our candidate set is a potential miss."""

import json
import os
import random
import sys
import time

sys.path.insert(0, "/home/claude/blast-radius/app")
sys.path.insert(0, "/home/claude/work")
os.environ.setdefault("BR_SRC", "/home/claude/sqlglot")
os.environ.setdefault("BR_DATA", "/home/claude/kit/sample")
os.environ.setdefault("BR_MAPS", "/home/claude/blast-radius/maps")
os.environ.setdefault("BR_TMP", "/tmp/brval")
os.environ.setdefault("BR_TIMEOUT_MULT", "12")
os.makedirs(os.environ["BR_TMP"], exist_ok=True)
import selector  # noqa: E402

selector.DEADLINE = time.time() + 1e9
import campaign_mut as cm  # noqa: E402

OUT = sys.argv[1]
random.seed(int(sys.argv[2]))
N = int(sys.argv[3])

tests = json.load(open(os.path.join(os.environ["BR_DATA"], "tests.json")))["tests"]
ms = [round(t["seconds"] * 1000) for t in tests]
meta = [(t["number"], t["file"], *t["id"].split("::")[1:]) for t in tests]
maps = selector.Maps(len(tests))
SRC = os.environ["BR_SRC"]
MODE = sys.argv[4] if len(sys.argv) > 4 else "any"
sys.path.insert(0, "/home/claude/mapper")
from astinfo import analyze  # noqa: E402

if MODE == "tests":
    files = [p for p in maps.index["files"] if p.startswith("tests/") and "/fixtures/" not in p]
else:
    files = [p for p in maps.index["files"] if p.startswith("sqlglot/")]
_ai = {}


def pick_line(rel, lines):
    if MODE != "deftime":
        return random.randint(1, len(lines))
    a = _ai.get(rel)
    if a is None:
        a = _ai[rel] = analyze("\n".join(lines))
    cands = [ln for ln in range(1, len(lines)) if a["line_stmt"][ln] is not None and ln not in a["runtime"] and ln not in a["docstring_lines"]]
    return random.choice(cands) if cands else 1

done = 0
with open(OUT, "a") as out:
    while done < N:
        rel = random.choice(files)
        lines = open(os.path.join(SRC, rel), encoding="utf-8").read().split("\n")
        ln = pick_line(rel, lines)
        r = cm.make_patch(rel, ln)
        if r is None:
            continue
        patch, old, new = r
        c = selector.analyze("v", patch.encode(), maps, len(tests), None)
        if c.unknown:
            continue
        predicted = set(c.cands) | set(c.sure) | set(c.always)
        # ground truth: run everything
        g = selector.Commit("g")
        g.patched, g.deleted = c.patched, c.deleted
        g.cheap, g.exp, g.sure, g.always = set(range(len(tests))), set(), set(), set()
        g.import_check = set(c.import_check)
        g.import_fail_sets = dict(c.import_fail_sets)
        t0 = time.time()
        selector.run_jobs([g], meta, ms, maps)
        truth = set(g.result or ())
        if g.import_failed:
            truth_note = "import-fail"
        else:
            truth_note = ""
        missed = sorted(truth - predicted) if not g.import_failed else []
        rec = {"rel": rel, "ln": ln, "old": old, "new": new, "pred": len(predicted), "truth": sorted(truth),
               "missed": missed, "note": truth_note, "wall": round(time.time() - t0, 1)}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        done += 1
        print(done, rel, ln, "pred", len(predicted), "truth", len(truth), "MISSED", len(missed), truth_note, file=sys.stderr, flush=True)
