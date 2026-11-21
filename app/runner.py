"""Run a chosen set of sqlglot tests against a patched copy, in a forked child.

The child reports one line per finished test on a pipe: "<number> <0|1>\n"
(1 = failed). A test the child never reports is treated as failed by the parent.
"""

import os
import signal
import sys
import time

FAIL_ALL_IN_FILE = -1


class _Timeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def build_overlay(src_root, dst_root, patched, deleted):
    """Symlink farm of src_root at dst_root, with patched files written for real.

    patched: {relpath: bytes}; deleted: set of relpaths.
    """
    tree = {}
    for rel in list(patched) + list(deleted):
        node = tree
        parts = rel.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if node is None:
                break
        else:
            node[parts[-1]] = rel

    def mk(src_dir, dst_dir, node):
        os.mkdir(dst_dir)
        try:
            names = os.listdir(src_dir)
        except OSError:
            names = []
        seen = set()
        for name in names:
            seen.add(name)
            s = os.path.join(src_dir, name)
            d = os.path.join(dst_dir, name)
            sub = node.get(name)
            if sub is None:
                os.symlink(s, d)
            elif isinstance(sub, dict):
                if os.path.isdir(s):
                    mk(s, d, sub)
                else:
                    mk(None, d, sub)
            else:
                if sub in deleted and sub not in patched:
                    continue
                _write(d, patched[sub])
        for name, sub in node.items():
            if name in seen:
                continue
            d = os.path.join(dst_dir, name)
            if isinstance(sub, dict):
                mk(None, d, sub)
            elif sub in patched:
                _write(d, patched[sub])

    mk(src_root, dst_root, tree)


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)
    # .pyc files are validated by (mtime, size). Derive the mtime from the content so a
    # cached .pyc can only ever match this exact source (never the original, never another patch).
    import zlib

    mt = 1_000_000_000 + (zlib.crc32(data) & 0x3FFFFFFF)
    os.utime(path, (mt, mt))


def _cap_duckdb_threads():
    """The scorer allows 256 processes+threads; duckdb sizes its pool from the host's cores.
    Two threads match the container's 2 CPUs and keep results the same."""
    duckdb = sys.modules.get("duckdb")
    if duckdb is None:
        return
    orig = duckdb.connect
    if getattr(orig, "_br_capped", False):
        return

    def connect(*args, **kwargs):
        cfg = dict(kwargs.pop("config", None) or {})
        cfg.setdefault("threads", 2)
        try:
            return orig(*args, config=cfg, **kwargs)
        except Exception:
            return orig(*args, **kwargs)

    connect._br_capped = True
    duckdb.connect = connect


PRE_ROWS = None


def _parse_rows(file_name):
    import ast
    import csv
    import gzip

    with open(file_name, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    fh = gzip.open(file_name, "rt", newline="") if gz else open(file_name, encoding="utf-8", newline="")
    reader = csv.reader(fh, delimiter="|")
    rows = []
    ctypes = []
    next(reader)
    for row in reader:
        if not ctypes:
            for i, v in enumerate(row):
                try:
                    ctypes.append(type(ast.literal_eval(v)))
                except (ValueError, SyntaxError):
                    ctypes.append(str)
        rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
    fh.close()
    return rows


TPC_SUB_OF_TEST = {"test_execute_tpch": "tpc-h", "test_execute_tpcds": "tpc-ds"}


def precompute_executor_rows(src, subs=("tpc-h", "tpc-ds")):
    """Parse the TPC-H/TPC-DS CSVs once in the parent, exactly like TestExecutor.setUpClass.
    Only the datasets of the given subs; only test_execute_tpch / test_execute_tpcds read them
    (other executor tests get empty tables)."""
    global PRE_ROWS
    import importlib.util

    if PRE_ROWS is None:
        PRE_ROWS = {}
    todo = [s for s in subs if ("_done", s) not in PRE_ROWS]
    if not todo:
        return
    spec = importlib.util.spec_from_file_location("_br_helpers", os.path.join(src, "tests", "helpers.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    for sub, schema in (("tpc-h", h.TPCH_SCHEMA), ("tpc-ds", h.TPCDS_SCHEMA)):
        if sub not in todo:
            continue
        for table in schema:
            PRE_ROWS[(sub, table)] = _parse_rows(os.path.join(src, "tests", "fixtures", "optimizer", sub, f"{table}.csv.gz"))
        PRE_ROWS[("_done", sub)] = True


TPC_FILTER = None  # {"tpc-h": set(indices), "tpc-ds": set(indices)} of edited fixture queries
OPT_FILTER = None  # {test method name: set((fixture filename, index))} queries a change can affect
CURRENT_TEST = [None]


def _install_opt_filter(mod):
    if not OPT_FILTER:
        return
    orig = mod.load_sql_fixture_pairs

    def load_sql_fixture_pairs(filename):
        allowed = OPT_FILTER.get(CURRENT_TEST[0])
        for i, pair in enumerate(orig(filename)):
            if allowed is None or (filename, i) in allowed:
                yield pair

    mod.load_sql_fixture_pairs = load_sql_fixture_pairs


def _install_fast_executor_setup(mod, patched):
    if PRE_ROWS is None:
        return
    for rel in patched:
        if rel in ("tests/test_executor.py", "tests/helpers.py"):
            return
        if rel.startswith("tests/fixtures/optimizer/tpc-") and not rel.endswith(".sql"):
            return
    klass = getattr(mod, "TestExecutor", None)
    if klass is None:
        return

    class LazyConn:
        """duckdb connection whose CREATE VIEW statements run on first use (most
        executor tests never touch duckdb, and binding 32 CSV views costs ~0.7 s)."""

        def __init__(self):
            self._conn = None
            self._pending = []

        def _real(self):
            if self._conn is None:
                self._conn = mod.duckdb.connect()
                for q in self._pending:
                    self._conn.execute(q)
                self._pending = None
            return self._conn

        def defer(self, q):
            if self._conn is None:
                self._pending.append(q)
            else:
                self._conn.execute(q)

        def close(self):
            if self._conn is not None:
                self._conn.close()

        def __getattr__(self, name):
            return getattr(self._real(), name)

    def setUpClass(cls):
        Table = mod.Table
        cls.tpch_conn = LazyConn()
        cls.tpcds_conn = LazyConn()
        cls.tpch_tables = {}
        cls.tpcds_tables = {}

        def setup(conn, directory, sub, table, columns, tables):
            file_name = f"{directory}{table}.csv.gz"
            conn.defer(
                f"""
                CREATE VIEW {table} AS
                SELECT *
                FROM READ_CSV('{file_name}', delim='|', header=True, columns={columns})
                """
            )
            rows = list(PRE_ROWS.get((sub, table), ()))  # empty: this child runs no test reading it
            tables[table] = rows
            tables[table] = Table(columns=columns, rows=rows)

        for table, columns in mod.TPCH_SCHEMA.items():
            setup(cls.tpch_conn, mod.DIR_TPCH, "tpc-h", table, columns, cls.tpch_tables)
        for table, columns in mod.TPCDS_SCHEMA.items():
            setup(cls.tpcds_conn, mod.DIR_TPCDS, "tpc-ds", table, columns, cls.tpcds_tables)
        cls.cache = {}
        cls.tpch_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
        cls.tpcds_sqls = list(mod.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
        if TPC_FILTER is not None:
            # only edited queries can change the outcome (sqlglot code is unchanged)
            if "tpc-h" in TPC_FILTER:
                cls.tpch_sqls = [p for i, p in enumerate(cls.tpch_sqls) if i in TPC_FILTER["tpc-h"]]
            if "tpc-ds" in TPC_FILTER:
                cls.tpcds_sqls = [p for i, p in enumerate(cls.tpcds_sqls) if i in TPC_FILTER["tpc-ds"]]

    klass.setUpClass = classmethod(setUpClass)


def _probe_generator_memory():
    """test_generator_memory's own checks (every dialect builds generators without raising and
    repeated construction keeps no memory). Objects that exist before the probe are frozen out
    of the cyclic collector, which makes its gc.collect() calls cheap without changing what
    the leak check sees."""
    import gc
    import importlib
    import tracemalloc

    importlib.import_module("tests.test_generator")
    from sqlglot.dialects import Dialect, Dialects

    gc.collect()
    gc.freeze()
    try:
        for name in Dialects:
            dialect = Dialect.get_or_raise(name)
            for _ in range(100):
                dialect.generator()
            gc.collect()
            tracemalloc.start()
            try:
                before = tracemalloc.get_traced_memory()[0]
                for _ in range(1000):
                    dialect.generator()
                gc.collect()
                after = tracemalloc.get_traced_memory()[0]
            finally:
                tracemalloc.stop()
            if after - before >= 1024:
                return False
        return True
    finally:
        gc.unfreeze()


def _probe_max_nodes():
    """test_max_nodes at 1/100 scale (same node-limit ratios and messages)."""
    import importlib

    importlib.import_module("tests.test_parser")
    from sqlglot import exp, parse_one
    from sqlglot.errors import ParseError

    if not isinstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select):
        return False
    cols = [f"col{i}" for i in range(500)]
    if not isinstance(parse_one(f"SELECT {','.join(cols)} FROM t", max_nodes=1_000_000), exp.Select):
        return False
    expr = "x" + "=x" * 100
    sql = "SELECT " + ",".join(expr for _ in range(100))
    for limit, text in ((10_000, "Maximum number of AST nodes"), (500, "(500)")):
        try:
            parse_one(sql, max_nodes=limit)
            return False
        except ParseError as e:
            if text not in str(e):
                return False
    try:
        parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
        return False
    except ParseError:
        pass
    return isinstance(parse_one("SELECT 1, 2, 3"), exp.Select)


# cheap stand-ins for the two slowest tests: same code paths and checks at a smaller size
PROBES = {
    "tests/test_generator.py::TestGenerator::test_generator_memory": _probe_generator_memory,
    "tests/test_parser.py::TestParser::test_max_nodes": _probe_max_nodes,
}


LEAF_DIRS = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")


def _leaf_module(rel):
    if not rel.startswith(LEAF_DIRS) or rel.count("/") != 2:
        return False
    return rel.rsplit("/", 1)[1] not in ("__init__.py", "dialect.py", "base.py")


_TOP_IMPORT = None
_GRAPH = {}


def _top_imports(rel, text):
    """Modules a file imports at module level (lines starting at column 0)."""
    import re

    global _TOP_IMPORT
    if _TOP_IMPORT is None:
        _TOP_IMPORT = re.compile(r"^(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[^\n]*)|import[ \t]+([^\n]+))", re.M)
    pkg = rel[:-3].replace("/", ".").rsplit(".", 1)[0]
    if rel.endswith("/__init__.py"):
        pkg = rel[: -len("/__init__.py")].replace("/", ".")
    out = set()
    for m in _TOP_IMPORT.finditer(text):
        base, names, plain = m.group(1), m.group(2), m.group(3)
        if plain is not None:
            for part in plain.split(","):
                out.add(part.strip().split(" ")[0])
            continue
        if base.startswith("."):
            dots = len(base) - len(base.lstrip("."))
            root = pkg.split(".")
            root = root[: len(root) - (dots - 1)] if dots > 1 else root
            base = ".".join(root + ([base.lstrip(".")] if base.strip(".") else []))
        out.add(base)
        for n in names.strip("()").replace("\n", " ").split(","):
            n = n.strip().split(" ")[0]
            if n and n.isidentifier():
                out.add(f"{base}.{n}")
    return out


def import_graph(root, patched=None):
    """{module relpath: set of module names it imports at top level} for sqlglot/ (cached per root)."""
    g = _GRAPH.get(root)
    if g is None:
        g = {}
        for d, _, names in os.walk(os.path.join(root, "sqlglot")):
            for n in names:
                if n.endswith(".py"):
                    full = os.path.join(d, n)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    try:
                        with open(full, encoding="utf-8") as f:
                            g[rel] = _top_imports(rel, f.read())
                    except (OSError, UnicodeDecodeError):
                        g[rel] = set()
        _GRAPH[root] = g
    if patched:
        g = dict(g)
        for rel, data in patched.items():
            if rel.startswith("sqlglot/") and rel.endswith(".py"):
                try:
                    g[rel] = _top_imports(rel, data.decode("utf-8"))
                except UnicodeDecodeError:
                    g[rel] = set()
    return g


def importers_closure(root, rels, patched=None):
    """Modules under sqlglot/ that import any of rels at module level, directly or through
    each other (so a failure while importing rels would show up when importing them)."""
    g = import_graph(root, patched)
    mod_of = {rel: rel[:-3].replace("/", ".") for rel in g}
    found, todo = set(), list(rels)
    while todo:
        rel = todo.pop()
        dotted = rel[:-3].replace("/", ".")
        for other, imps in g.items():
            if other in found or other in rels:
                continue
            if dotted in imps:
                found.add(other)
                todo.append(other)
    return sorted(found)


def child_main(overlay, tests, out_fd, timeouts, mem_limit, import_mods=(), patched=(), probes=(), import_extra=None):
    """tests: list of (number, file, cls, name). Runs in the forked child."""
    # if memory runs out, the kernel should kill this child, never the parent
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except Exception:
        pass
    # never let a test write to the real stdout (it would corrupt the answer stream)
    try:
        dn = os.open(os.devnull, os.O_RDWR)
        os.dup2(dn, 1)
        os.dup2(dn, 2)
        os.dup2(dn, 0)
        if dn > 2:
            os.close(dn)
        sys.stdout = open(os.devnull, "w")
        sys.stderr = sys.stdout
    except Exception:
        pass
    try:
        import resource

        if mem_limit:
            cur = 0
            try:
                with open("/proc/self/statm") as f:
                    cur = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
            except Exception:
                pass
            lim = cur + mem_limit
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:
        pass
    os.chdir(overlay)
    src = os.environ.get("BR_SRC", "/app/sqlglot").rstrip("/")
    sys.path[:] = [overlay] + [p for p in sys.path if p and p.rstrip("/") != src and not p.startswith(src + "/")]
    sys.dont_write_bytecode = True
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTHONPATH"] = overlay
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for k in list(sys.modules):
        if k == "sqlglot" or k.startswith("sqlglot.") or k == "tests" or k.startswith("tests."):
            del sys.modules[k]

    out = os.fdopen(out_fd, "w", buffering=1)

    def report(num, failed):
        out.write(f"{num} {1 if failed else 0}\n")

    import unittest

    import syncpool

    syncpool.install()
    _cap_duckdb_threads()
    signal.signal(signal.SIGALRM, _on_alarm)

    for rel in import_mods:
        modname = rel[:-3].replace("/", ".")
        if modname.endswith(".__init__"):
            modname = modname[: -len(".__init__")]
        try:
            signal.setitimer(signal.ITIMER_REAL, float(os.environ.get("BR_IMPORT_TIMEOUT", "60")))
            __import__(modname)
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            out.write(f"-2 {rel}\n")
            out.flush()
            os._exit(0)

    # class bodies elsewhere run sqlglot code at import time (e.g. generators parse SQL
    # templates), so a definition change can break importing *another* module: import every
    # dialect module too and report the ones that fail. A per-dialect module (dialects/x.py,
    # parsers/x.py, ...) only runs when imported, so only modules importing it need checking.
    if import_mods and import_extra is not None:
        broken = False
        for rel in import_extra:
            try:
                signal.setitimer(signal.ITIMER_REAL, float(os.environ.get("BR_IMPORT_TIMEOUT", "60")))
                __import__(rel[:-3].replace("/", "."))
                signal.setitimer(signal.ITIMER_REAL, 0)
            except BaseException:
                signal.setitimer(signal.ITIMER_REAL, 0)
                out.write(f"-2 {rel}\n")
                broken = True
        if broken:
            out.flush()
            os._exit(0)
    elif import_mods:
        try:
            names = list(sys.modules["sqlglot.dialects"].DIALECT_MODULE_NAMES) if "sqlglot.dialects" in sys.modules else []
        except Exception:
            names = []
        if not names:
            try:
                import sqlglot.dialects as _d

                names = list(_d.DIALECT_MODULE_NAMES)
            except Exception:
                names = []
        broken = False
        for name in names:
            try:
                signal.setitimer(signal.ITIMER_REAL, float(os.environ.get("BR_IMPORT_TIMEOUT", "60")))
                __import__(f"sqlglot.dialects.{name}")
                signal.setitimer(signal.ITIMER_REAL, 0)
            except BaseException:
                signal.setitimer(signal.ITIMER_REAL, 0)
                out.write(f"-2 sqlglot/dialects/{name}.py\n")
                broken = True
        if broken:
            out.flush()
            os._exit(0)
    # imports of the patched modules are fine
    out.write("-3 ready\n")
    out.flush()

    # group by file then class, keeping order
    by_file = {}
    for num, file, cls, name in tests:
        by_file.setdefault(file, []).append((num, cls, name))

    class Result(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.failed = set()
            self.cur = None

        def startTest(self, test):
            super().startTest(test)
            CURRENT_TEST[0] = getattr(test, "_testMethodName", None)
            num = getattr(test, "_br_num", None)
            self.cur = num
            t = timeouts.get(num, 10.0)
            signal.setitimer(signal.ITIMER_REAL, t)

        def stopTest(self, test):
            signal.setitimer(signal.ITIMER_REAL, 0)
            super().stopTest(test)
            num = getattr(test, "_br_num", None)
            if num is not None:
                report(num, num in self.failed)

        def _mark(self, test):
            num = getattr(test, "_br_num", None)
            if num is None:
                # class/module fixture error: fail every test of that class still pending
                self.fixture_error = True
            else:
                self.failed.add(num)

        def _trace(self, test, err):
            tf = os.environ.get("BR_TRACE")
            if tf:
                import traceback

                with open(tf, "a") as fh:
                    fh.write(f"=== {test}\n" + "".join(traceback.format_exception(*err))[-3000:] + "\n")

        def addError(self, test, err):
            self._trace(test, err)
            self._mark(test)

        def addFailure(self, test, err):
            self._trace(test, err)
            self._mark(test)

        def addSubTest(self, test, subtest, err):
            if err is not None:
                self._trace(test, err)
                self._mark(test)

        def addUnexpectedSuccess(self, test):
            self._mark(test)

    # like pytest: import every test module before running any test (tests may patch
    # sys.modules / environment while they run, which must not affect later imports)
    imported = {}
    for file in by_file:
        modname = file[:-3].replace("/", ".")
        try:
            signal.setitimer(signal.ITIMER_REAL, float(os.environ.get("BR_IMPORT_TIMEOUT", "60")))
            mod = __import__(modname, fromlist=["_"])
            signal.setitimer(signal.ITIMER_REAL, 0)
            if modname == "tests.test_executor":
                _install_fast_executor_setup(mod, patched)
            if modname == "tests.test_optimizer":
                _install_opt_filter(mod)
            imported[file] = mod
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            tf = os.environ.get("BR_TRACE")
            if tf:
                import traceback

                with open(tf, "a") as fh:
                    fh.write(f"=== IMPORT {file}\n" + traceback.format_exc()[-2500:] + "\n")
            out.write(f"{FAIL_ALL_IN_FILE} {file}\n")
            continue

    if os.environ.get("BR_GC_FREEZE", "1") == "1":
        # everything imported so far lives for the whole child: keep it out of the cyclic
        # collector so collections during the tests only scan what the tests create
        import gc

        gc.freeze()
        thr = os.environ.get("BR_GC_THRESHOLD")
        if thr:
            gc.set_threshold(*[int(x) for x in thr.split(",")])
    for file, items in by_file.items():
        mod = imported.get(file)
        if mod is None:
            continue
        by_cls = {}
        for num, cls, name in items:
            by_cls.setdefault(cls, []).append((num, name))
        for cls, members in by_cls.items():
            klass = getattr(mod, cls, None)
            if klass is None:
                for num, _ in members:
                    report(num, True)
                continue
            suite = unittest.TestSuite()
            for num, name in members:
                try:
                    tc = klass(name)
                except BaseException:
                    report(num, True)
                    continue
                tc._br_num = num
                suite.addTest(tc)
            result = Result()
            result.fixture_error = False
            try:
                suite.run(result)
            except _Timeout:
                pass
            except BaseException:
                pass
            signal.setitimer(signal.ITIMER_REAL, 0)
            if result.fixture_error:
                for num, _ in members:
                    report(num, True)
            # tearDownClass is run by the suite when the next class starts or at the end
            try:
                unittest.suite._DebugResult  # noqa: B018
            except Exception:
                pass
    out.flush()
    for num, name in probes:
        fn = PROBES.get(name)
        ok = False
        try:
            signal.setitimer(signal.ITIMER_REAL, 30.0)
            ok = bool(fn())
        except BaseException:
            ok = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            out.write(f"probe {num} {'ok' if ok else 'bad'}\n")
            out.flush()
        except Exception:
            pass
    os._exit(0)
