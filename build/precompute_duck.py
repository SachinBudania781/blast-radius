"""Build step: DuckDB's answers to the TPC-H / TPC-DS fixture queries, keyed by the exact SQL
string sent to DuckDB (what TestExecutor.cached_execute and test_optimized_tpch send at the
pinned commit). At run time a query whose transpiled text is identical gets this answer
instead of re-running DuckDB; any other text runs DuckDB as usual."""
import os
import pickle
import sys

src, out = sys.argv[1], sys.argv[2]
os.chdir(src)
sys.path.insert(0, src)

import duckdb  # noqa: E402

from sqlglot import transpile  # noqa: E402
from tests.helpers import TPCDS_SCHEMA, TPCH_SCHEMA, load_sql_fixture_pairs  # noqa: E402

DIRS = {"tpch": ("tests/fixtures/optimizer/tpc-h/", TPCH_SCHEMA), "tpcds": ("tests/fixtures/optimizer/tpc-ds/", TPCDS_SCHEMA)}
cache = {}
for db, (directory, schema) in DIRS.items():
    conn = duckdb.connect()
    for table, columns in schema.items():
        conn.execute(
            f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{directory}{table}.csv.gz', delim='|', header=True, columns={columns})
                """
        )
    fixture = "optimizer/tpc-h/tpc-h.sql" if db == "tpch" else "optimizer/tpc-ds/tpc-ds.sql"
    for _, sql, optimized in load_sql_fixture_pairs(fixture):
        texts = [transpile(sql, write="duckdb")[0]]
        if db == "tpch":
            texts.append(transpile(optimized, write="duckdb")[0])
        for text in texts:
            if (db, text) not in cache:
                try:
                    cache[(db, text)] = conn.execute(text).fetchdf()
                except Exception:  # noqa: BLE001  (not cached: run time asks DuckDB itself)
                    pass
    conn.close()
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
print("duck cache", out, len(cache), "queries", os.path.getsize(out), "bytes")
