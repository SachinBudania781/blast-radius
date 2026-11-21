"""Per-query line coverage for the TPC executor tests.

For each TPC-H / TPC-DS query i, run the real test methods (test_execute_tpch/_tpcds,
test_optimized_tpch) with the class's query list reduced to [query i] and record the
sqlglot lines executed. A change touching none of query i's lines cannot change query i's
outcome, so the selector can re-run only the affected queries.

Output: pickle {test_name: [(lines frozenset, cpu_seconds) per query index]}
"""

import os
import pickle
import sys
import time

ROOT = os.path.realpath(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/sqlglot")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/brmap2/tpc_queries.pkl"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["SKIP_INTEGRATION"] = "0"
os.chdir(ROOT)

import syncpool  # noqa: E402

syncpool.install()

import unittest  # noqa: E402

import tests.test_executor as te  # noqa: E402

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


mon.use_tool_id(TOOL, "tpcq")
mon.register_callback(TOOL, mon.events.LINE, on_line)

K = te.TestExecutor
K.setUpClass()
all_h = list(K.tpch_sqls)
all_ds = list(K.tpcds_sqls)

result = {}


class R(unittest.TestResult):
    pass


def run(test_name, attr, full):
    out = []
    for i, pair in enumerate(full):
        setattr(K, attr, [pair])
        K.cache = {}
        g = sys.modules.get("sqlglot.generator")
        if g is not None and hasattr(g, "_DISPATCH_CACHE"):
            g._DISPATCH_CACHE.clear()
        cur.clear()
        mon.set_events(TOOL, mon.events.LINE)
        mon.restart_events()
        t = time.process_time()
        r = R()
        K(test_name).run(r)
        dt = time.process_time() - t
        mon.set_events(TOOL, 0)
        ok = not r.failures and not r.errors
        out.append((frozenset(cur), dt, ok))
    setattr(K, attr, full)
    result[test_name] = out
    print(test_name, len(out), "queries", sum(1 for x in out if not x[2]), "failing", file=sys.stderr)


# warm-up: trigger every lazy import / one-time initialisation outside the recording, so
# per-query line sets only contain work that each query really does
for _name in ("test_execute_tpch", "test_optimized_tpch", "test_execute_tpcds"):
    K.cache = {}
    K(_name).run(R())
print("warm-up done", file=sys.stderr)

run("test_execute_tpch", "tpch_sqls", all_h)
run("test_optimized_tpch", "tpch_sqls", all_h)
run("test_execute_tpcds", "tpcds_sqls", all_ds)
with open(OUT, "wb") as f:
    pickle.dump(result, f)
