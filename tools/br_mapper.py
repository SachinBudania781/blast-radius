"""pytest plugin: per-test line coverage, classes used (type of self/cls), files opened.

Run from the sqlglot root:  pytest -p br_mapper tests
Writes $BR_MAP_OUT (one pickle per xdist worker).
"""

import builtins
import io
import os
import pickle
import sys

import pytest

import syncpool

syncpool.install()

ROOT = os.path.realpath(os.getcwd())
OUT = os.environ.get("BR_MAP_OUT", "/tmp/brmap")
mon = sys.monitoring
TOOL = 4

_tracked = {}  # code -> file key or None
_method = {}  # code -> argname or None
cur = {"lines": None, "classes": None, "files": None}
results = {}
collection = {"lines": set(), "classes": set(), "files": set()}
_setup_lines = {}  # test id -> lines seen during setup phase


def _key(code):
    k = _tracked.get(code, 0)
    if k == 0:
        fn = code.co_filename
        if fn.startswith(ROOT + "/sqlglot/") or fn.startswith(ROOT + "/tests/"):
            k = fn[len(ROOT) + 1 :]
        else:
            k = None
        _tracked[code] = k
    return k


import_lines = {}  # (file, line) -> set of modules whose import executed it


def _importing_module():
    f = sys._getframe(2)
    while f is not None:
        c = f.f_code
        if c.co_name == "<module>":
            k = _key(c)
            if k is not None:
                return k
        f = f.f_back
    return None


def on_line(code, line):
    k = _key(code)
    if k is None:
        return mon.DISABLE
    if cur["lines"] is None:
        collection["lines"].add((k, line))
        m = _importing_module()
        if m is not None:
            import_lines.setdefault((k, line), set()).add(m)
            return None
        return mon.DISABLE
    m = _importing_module()
    if m is not None and not m.startswith("tests/"):
        import_lines.setdefault((k, line), set()).add(m)
        return None
    cur["lines"].add((k, line))
    return mon.DISABLE


def on_start(code, offset):
    a = _method.get(code, 0)
    if a == 0:
        a = None
        if _key(code) is not None and code.co_argcount > 0:
            v0 = code.co_varnames[0]
            if v0 in ("self", "cls", "klass", "mcs", "metacls"):
                a = v0
        _method[code] = a
    if a is None:
        return mon.DISABLE
    f = sys._getframe(1)
    try:
        v = f.f_locals[a]
    except Exception:
        return None
    t = v if isinstance(v, type) else type(v)
    bucket = cur["classes"] if cur["classes"] is not None else collection["classes"]
    bucket.add(t)
    return None


_real_open = io.open


def _rec(path):
    try:
        p = os.path.realpath(os.fspath(path))
    except Exception:
        return
    if p.startswith(ROOT + "/"):
        bucket = cur["files"] if cur["files"] is not None else collection["files"]
        bucket.add(p[len(ROOT) + 1 :])


def traced_open(file, *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)):
        _rec(file)
    return _real_open(file, *a, **k)


_real_listdir = os.listdir
_real_scandir = os.scandir


def traced_listdir(path="."):
    _rec(path)
    return _real_listdir(path)


def traced_scandir(path="."):
    _rec(path)
    return _real_scandir(path)


def pytest_configure(config):
    mon.use_tool_id(TOOL, "brmap")
    mon.register_callback(TOOL, mon.events.LINE, on_line)
    mon.register_callback(TOOL, mon.events.PY_START, on_start)
    mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
    builtins.open = traced_open
    io.open = traced_open
    os.listdir = traced_listdir
    os.scandir = traced_scandir


def _cls_name(t):
    return f"{t.__module__}:{t.__qualname__}"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    cur["lines"] = set()
    cur["classes"] = set()
    cur["files"] = set()
    g = sys.modules.get("sqlglot.generator")
    if g is not None and hasattr(g, "_DISPATCH_CACHE"):
        g._DISPATCH_CACHE.clear()
    mon.restart_events()
    yield
    results[item.nodeid] = {
        "lines": cur["lines"],
        "classes": {_cls_name(t) for t in cur["classes"]},
        "files": cur["files"],
        "setup_lines": _setup_lines.pop(item.nodeid, set()),
    }
    cur["lines"] = cur["classes"] = cur["files"] = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    before = set(cur["lines"]) if cur["lines"] is not None else set()
    yield
    if cur["lines"] is not None:
        _setup_lines[item.nodeid] = cur["lines"] - before


def pytest_unconfigure(config):
    mon.set_events(TOOL, 0)
    wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
    # class info for every sqlglot class seen: mro + own attribute names
    seen = set()
    for r in results.values():
        seen |= r["classes"]
    classes = {}
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", "")
        if not (name == "sqlglot" or name.startswith("sqlglot.") or name.startswith("tests")):
            continue
        for obj in list(vars(mod).values()):
            stack = [obj]
            while stack:
                o = stack.pop()
                if not isinstance(o, type):
                    continue
                try:
                    cn = _cls_name(o)
                except Exception:
                    continue
                if cn in classes:
                    continue
                classes[cn] = {
                    "mro": [_cls_name(b) for b in o.__mro__],
                    "own": sorted(k for k in vars(o).keys()),
                    "file": getattr(sys.modules.get(o.__module__), "__file__", None),
                }
                stack.extend(v for v in vars(o).values() if isinstance(v, type))
                stack.extend(type.__subclasses__(o))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"map-{wid}.pkl"), "wb") as f:
        pickle.dump(
            {
                "results": results,
                "collection": {
                    "lines": collection["lines"],
                    "classes": {_cls_name(t) for t in collection["classes"]},
                    "files": collection["files"],
                },
                "classes": classes,
                "import_lines": import_lines,
            },
            f,
        )
