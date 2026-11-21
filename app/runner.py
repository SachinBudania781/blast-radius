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
    os.utime(path, (1, 1))


def child_main(overlay, tests, out_fd, timeouts, mem_limit, import_mods=()):
    """tests: list of (number, file, cls, name). Runs in the forked child."""
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
    for k in list(sys.modules):
        if k == "sqlglot" or k.startswith("sqlglot.") or k == "tests" or k.startswith("tests."):
            del sys.modules[k]

    out = os.fdopen(out_fd, "w", buffering=1)

    def report(num, failed):
        out.write(f"{num} {1 if failed else 0}\n")

    import unittest

    import syncpool

    syncpool.install()
    signal.signal(signal.SIGALRM, _on_alarm)

    for rel in import_mods:
        modname = rel[:-3].replace("/", ".")
        if modname.endswith(".__init__"):
            modname = modname[: -len(".__init__")]
        try:
            signal.setitimer(signal.ITIMER_REAL, 30.0)
            __import__(modname)
            signal.setitimer(signal.ITIMER_REAL, 0)
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            out.write(f"-2 {rel}\n")
            out.flush()
            os._exit(0)

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
            signal.setitimer(signal.ITIMER_REAL, 30.0)
            mod = __import__(modname, fromlist=["_"])
            signal.setitimer(signal.ITIMER_REAL, 0)
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
