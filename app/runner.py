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


def precompute_executor_rows(src):
    """Parse the TPC-H/TPC-DS CSVs once in the parent, exactly like TestExecutor.setUpClass."""
    global PRE_ROWS
    import importlib.util

    spec = importlib.util.spec_from_file_location("_br_helpers", os.path.join(src, "tests", "helpers.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    pre = {}
    for sub, schema in (("tpc-h", h.TPCH_SCHEMA), ("tpc-ds", h.TPCDS_SCHEMA)):
        for table in schema:
            pre[(sub, table)] = _parse_rows(os.path.join(src, "tests", "fixtures", "optimizer", sub, f"{table}.csv.gz"))
    PRE_ROWS = pre


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
            rows = list(PRE_ROWS[(sub, table)])
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


def child_main(overlay, tests, out_fd, timeouts, mem_limit, import_mods=(), patched=()):
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
    # dialect module too and report the ones that fail
    if import_mods:
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

    for file, items in by_file.items():
        modname = file[:-3].replace("/", ".")
        try:
            signal.setitimer(signal.ITIMER_REAL, float(os.environ.get("BR_IMPORT_TIMEOUT", "60")))
            mod = __import__(modname, fromlist=["_"])
            signal.setitimer(signal.ITIMER_REAL, 0)
            if modname == "tests.test_executor":
                _install_fast_executor_setup(mod, patched)
            if modname == "tests.test_optimizer":
                _install_opt_filter(mod)
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            out.write(f"{FAIL_ALL_IN_FILE} {file}\n")
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
    os._exit(0)
