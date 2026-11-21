"""Per-fixture-query line coverage for optimizer tests that loop over fixture files
(TestOptimizer.check_file). For each test and each (fixture file, query index), run the
real test with the fixture loader reduced to that one query and record sqlglot lines.

Output pickle: {test_name: {"files": {fname: n_pairs}, "runs": [(fname, i, lines, cpu)]}}
"""

import os
import pickle
import sys
import time

ROOT = os.path.realpath(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/sqlglot")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/brmap2/opt_queries.pkl"
ONLY = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["SKIP_INTEGRATION"] = "0"
os.chdir(ROOT)

import syncpool  # noqa: E402

syncpool.install()
import unittest  # noqa: E402

import tests.test_optimizer as to  # noqa: E402

mon = sys.monitoring
TOOL = 4
cur = set()
_keys = {}


def key(code):
    k = _keys.get(code, 0)
    if k == 0:
        fn = code.co_filename
        k = fn[len(ROOT) + 1 :] if fn.startswith(ROOT + "/sqlglot/") else None
        _keys[code] = k
    return k


def on_line(code, line):
    k = key(code)
    if k is not None:
        cur.add((k, line))
    return mon.DISABLE


mon.use_tool_id(TOOL, "optq")
mon.register_callback(TOOL, mon.events.LINE, on_line)

orig_loader = to.load_sql_fixture_pairs
MODE = {"allowed": None, "seen": None}


def loader(filename):
    pairs = list(orig_loader(filename))
    if MODE["seen"] is not None:
        MODE["seen"][filename] = len(pairs)
    allowed = MODE["allowed"]
    for i, p in enumerate(pairs):
        if allowed is None or (filename, i) in allowed:
            yield p


to.load_sql_fixture_pairs = loader
K = to.TestOptimizer
K.setUpClass()


class R(unittest.TestResult):
    pass


src = open(os.path.join(ROOT, "tests", "test_optimizer.py")).read()
names = sorted(n for n in dir(K) if n.startswith("test_"))
# which tests call check_file (by source inspection)
import inspect  # noqa: E402

cf_tests = [n for n in names if "check_file(" in inspect.getsource(getattr(K, n))]
if ONLY:
    cf_tests = [n for n in cf_tests if n in ONLY]

result = {}
# warm-up: everything once, unrecorded
for n in cf_tests:
    MODE["allowed"], MODE["seen"] = None, {}
    K(n).run(R())
print("warm-up done", len(cf_tests), file=sys.stderr)

for n in cf_tests:
    MODE["allowed"], MODE["seen"] = None, {}
    K(n).run(R())
    files = dict(MODE["seen"])
    runs = []
    for fname, cnt in files.items():
        for i in range(cnt):
            MODE["allowed"], MODE["seen"] = {(fname, i)}, None
            g = sys.modules.get("sqlglot.generator")
            if g is not None and hasattr(g, "_DISPATCH_CACHE"):
                g._DISPATCH_CACHE.clear()
            cur.clear()
            mon.set_events(TOOL, mon.events.LINE)
            mon.restart_events()
            t = time.process_time()
            r = R()
            K(n).run(r)
            dt = time.process_time() - t
            mon.set_events(TOOL, 0)
            runs.append((fname, i, frozenset(cur), dt, not r.failures and not r.errors))
    result[n] = {"files": files, "runs": runs}
    bad = sum(1 for x in runs if not x[4])
    print(n, files, "runs", len(runs), "failing", bad, file=sys.stderr, flush=True)

with open(OUT, "wb") as f:
    pickle.dump(result, f)
