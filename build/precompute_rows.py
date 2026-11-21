"""Build step: parse the pristine TPC-H/TPC-DS fixture rows once and store them (marshal)."""
import marshal
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import runner  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.environ.pop("BR_ROWS_CACHE", None)
runner.precompute_executor_rows(src)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    marshal.dump(runner.PRE_ROWS, f)
print("rows cache", out, sum(len(v) for v in runner.PRE_ROWS.values()), "rows", os.path.getsize(out), "bytes")
