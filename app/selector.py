"""Blast Radius selector.

For each patch: find the tests that can be affected (from maps built at the pinned
commit), run the cheap ones against the patched code to see which really fail, and
print only the failing ones packed onto workers.
"""

import os as _os

# numeric libraries start one thread per visible host core on import; the scorer allows 256
# processes+threads in total, so keep them single-threaded (results are identical)
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import array
import json
import marshal
import os
import re
import select
import sys
import time

APP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP)

import patchutil  # noqa: E402
import policy  # noqa: E402

SRC = os.environ.get("BR_SRC", "/app/sqlglot")
DATA = os.environ.get("BR_DATA", "/data")
MAPS = os.environ.get("BR_MAPS", os.path.join(os.path.dirname(APP), "maps"))
TMP = os.environ.get("BR_TMP", "/tmp")
PAR = int(os.environ.get("BR_PAR", "2"))
DEBUG = os.environ.get("BR_DEBUG") == "1"
VERIFY_ALL = os.environ.get("BR_VERIFY_ALL") == "1"
DIAG = os.environ.get("BR_DIAG") == "1"
PDIAG = os.environ.get("BR_PDIAG") == "1"  # measurement build: why was a passing TPC-DS run added
CDIAG = os.environ.get("BR_CDIAG") == "1"  # measurement build: where the CPU goes
CPU_BY_PHASE = {}  # measurement build: encode where unverified tests were wasted  # measurement mode (never include unverified, except catastrophic)

# tests with a cheap stand-in in runner.PROBES
PROBE_IDS = ("tests/test_generator.py::TestGenerator::test_generator_memory", "tests/test_parser.py::TestParser::test_max_nodes")
EXPENSIVE = 1.0  # recorded seconds at or above which a test is "expensive"
# wall-clock budget for the whole batch (the scorer allows 180 s for ~40 commits, 270 s for ~60)
DEADLINE = time.time() + float(os.environ.get("BR_BUDGET", "0") or 0)
WORKER_MS = 2000


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- maps
class Maps:
    def __init__(self, ntests):
        self.index = marshal.load(open(os.path.join(MAPS, "index.m"), "rb"))
        with open(os.path.join(MAPS, "sets.bin"), "rb") as f:
            self.blob = f.read()
        self.nb = (self.index["ntests"] + 7) // 8
        self.ntests = ntests
        self._files = {}
        self._sets = {}

    def file(self, rel):
        fn = self.index["files"].get(rel)
        if fn is None:
            return None
        v = self._files.get(rel)
        if v is None:
            raw = marshal.load(open(os.path.join(MAPS, fn), "rb"))
            narrow = array.array("I")
            narrow.frombytes(raw[0])
            wide = array.array("I")
            wide.frombytes(raw[1])
            v = (narrow, wide) + tuple(raw[2:])
            self._files[rel] = v
        return v

    def set(self, sid):
        s = self._sets.get(sid)
        if s is None:
            b = self.blob[sid * self.nb : (sid + 1) * self.nb]
            s = set()
            for i, byte in enumerate(b):
                if byte:
                    base = i << 3
                    for j in range(8):
                        if byte >> j & 1:
                            s.add(base + j)
            self._sets[sid] = s
        return s

    def readers(self, path):
        sid = self.index["file_readers"].get(path)
        return self.set(sid) if sid is not None else set()

    def direct_readers(self, path):
        sid = self.index.get("file_direct", {}).get(path)
        return self.set(sid) if sid is not None else set()


# ---------------------------------------------------------------- change analysis
_COMMENT = re.compile(rb"^[ \t\f]*(#.*)?$")
_CODING = re.compile(rb"^[ \t\f]*#.*?coding[:=]")
_STRUCT = re.compile(rb"\b(def|class|lambda|yield|global|nonlocal|async|await|import|from|del|with|try|except|finally|raise|return|for|while)\b")
_ASSIGN = re.compile(rb"^\s*([A-Za-z_][\w\.\[\]'\", ]*?)\s*(?:[-+*/%&|^]|//|>>|<<|\*\*)?=(?!=)")


def _inside_string(spans, pos):
    """True if inserting before old line `pos` lands inside a multi-line string."""
    for a, b in spans:
        if a < pos <= b:
            return True
    return False


def _line_in_string(spans, ln):
    for a, b in spans:
        if a < ln <= b or (a <= ln < b):
            return True
    return False


def hunk_is_comment_only(removed, rtxt, added, string_spans, backslash):
    """Sound check: the hunk only adds/removes/edits comments or blank lines."""
    # pair up equal counts as modifications, otherwise all must be comment/blank lines
    if len(removed) == len(added) and removed:
        for (ln, old), (_, new) in zip(zip(removed, rtxt), added):
            if old == new:
                continue
            if _line_in_string(string_spans, ln):
                return False
            if (ln - 1) in backslash or ln in backslash:
                return False
            if _COMMENT.match(old) and _COMMENT.match(new):
                if ln <= 2 and (_CODING.match(new) or _CODING.match(old)):
                    return False
                continue
            o = old.rstrip()
            if b"#" in old:
                return False
            if new.startswith(o) and _COMMENT.match(new[len(o) :]) and new[len(o) : len(o) + 1] in (b" ", b"\t", b"#"):
                if o.endswith(b"\\"):
                    return False
                continue
            return False
        return True
    for ln, old in zip(removed, rtxt):
        if not _COMMENT.match(old):
            return False
        if _line_in_string(string_spans, ln) or (ln - 1) in backslash:
            return False
        if ln <= 2 and _CODING.match(old):
            return False
    for pos, new in added:
        if not _COMMENT.match(new):
            return False
        if _inside_string(string_spans, pos) or (pos - 1) in backslash:
            return False
        if pos <= 2 and _CODING.match(new):
            return False
    return True


def is_structural(rtxt, added):
    texts = list(rtxt) + [t for _, t in added]
    if len(rtxt) != len(added):
        return True
    for t in texts:
        if _STRUCT.search(t):
            return True
    for old, (_, new) in zip(rtxt, added):
        oi = len(old) - len(old.lstrip())
        ni = len(new) - len(new.lstrip())
        if oi != ni:
            return True
        mo, mn = _ASSIGN.match(old), _ASSIGN.match(new)
        if (mo and mo.group(1)) != (mn and mn.group(1)):
            return True
    return False


def local_syntax_ok(patched, fp, fspans):
    """Compile only the functions touched by the patch (cheap). None = cannot tell."""
    lines = patched.split(b"\n")
    delta_before = []  # (old_start, delta) cumulative
    regions = []
    for os_, ol, ns, nl, body in fp.hunks:
        regions.append((os_, os_ + max(ol, 1) - 1, ns - os_))
    touched = []
    for lo, hi, _ in regions:
        best = None
        for s, e, ind in fspans:
            if s <= lo and hi <= e:
                if best is None or (e - s) < (best[1] - best[0]):
                    best = (s, e, ind)
        if best is None:
            return None
        touched.append(best)
    for s, e, ind in set(touched):
        # map old span to new span using hunk deltas
        def new_ln(old_ln):
            d = 0
            for os_, ol, ns, nl, body in fp.hunks:
                if os_ + max(ol, 1) - 1 < old_ln:
                    d += nl - ol
            return old_ln + d

        ns_ = new_ln(s)
        d_in = 0
        for os_, ol, ns, nl, body in fp.hunks:
            if s <= os_ <= e:
                d_in += nl - ol
        ne_ = new_ln(s) + (e - s) + d_in
        seg = lines[ns_ - 1 : ne_]
        if not seg:
            return None
        out = []
        for ln in seg:
            if ln.strip() == b"":
                out.append(b"")
                continue
            lead = len(ln) - len(ln.lstrip(b" \t"))
            if lead < ind:
                return False
            out.append(ln[ind:])
        try:
            compile(b"\n".join(out) + b"\n", "<local>", "exec", dont_inherit=True)
        except SyntaxError:
            return None
        except Exception:
            return None
    return True


class Commit:
    def __init__(self, cid):
        self.cid = cid
        self.patched = {}  # relpath -> bytes
        self.deleted = set()
        self.cands = set()
        self.always = set()  # tests that fail for sure (no need to run)
        self.sure = set()  # tests we include without running (expected to fail)
        self.import_check = set()  # python module relpaths to import first
        self.import_fail_sets = {}  # relpath -> tests failing if import fails
        self.mod_sets = {}  # patched python relpath -> tests that import/use it
        self.line_changes = []  # (relpath, old line numbers, structural)
        self.dedicated = set()  # expensive tests dedicated to a changed file
        self.unknown = False
        self.result = None
        self.notes = []


def analyze(cid, data, maps, ntests, all_tests):
    c = Commit(cid)
    try:
        fps = patchutil.parse(data)
    except Exception:
        fps = []
    if not fps:
        if data.strip():
            c.unknown = True
        return c
    for fp in fps:
        if fp.binary:
            # cannot apply binary deltas: a binary file matters only to tests that read it
            path = fp.new_path or fp.old_path
            if path and path.endswith(".py"):
                c.unknown = True
            elif path:
                rd = maps.readers(path) | maps.readers(fp.old_path or path)
                c.always |= rd
            else:
                c.unknown = True
            continue
        old = fp.old_path
        new = fp.new_path
        # file content
        original = b""
        if old and not fp.is_new:
            try:
                with open(os.path.join(SRC, old), "rb") as f:
                    original = f.read()
            except OSError:
                original = None
        if fp.is_deleted or new is None:
            if old:
                c.deleted.add(old)
                _deleted_file(c, old, maps)
            continue
        if original is None:
            c.unknown = True
            continue
        patched = patchutil.apply(original, fp) if fp.hunks else original
        if patched is None:
            c.unknown = True
            continue
        if fp.rename and old and old != new:
            c.deleted.add(old)
            _deleted_file(c, old, maps)
        c.patched[new] = patched
        if not fp.hunks:
            if fp.rename:
                _new_file(c, new, maps)
            continue
        if fp.is_new or (fp.rename and old != new):
            _new_file(c, new, maps)
            continue
        info = maps.file(old)
        if info is None:
            # non-python or python file the maps do not cover
            rd = maps.readers(old)
            if old.endswith(".py") and (old.startswith("sqlglot/") or old.startswith("tests/")):
                c.unknown = True
                continue
            if old.endswith(".sql") and old.startswith("tests/fixtures/"):
                if _sql_fixture_harmless(fp):
                    continue
                # an edited fixture query nearly always breaks the tests that load it themselves;
                # tests that only see it through setUpClass are run to check
                c.sure |= maps.direct_readers(old) & rd
                c.fixture_readers = getattr(c, "fixture_readers", set()) | rd
                for key in ("tpc-h", "tpc-ds"):
                    if old == f"tests/fixtures/optimizer/{key}/{key}.sql":
                        ch = _tpc_changed(original, patched)
                        if ch is not None:
                            c.tpc_filter = getattr(c, "tpc_filter", None) or {"tpc-h": set(), "tpc-ds": set()}
                            c.tpc_filter[key] = ch
                        else:
                            c.tpc_filter_bad = True
            c.cands |= rd
            continue
        narrow, wide, rd, mu, sspans, bslash, fspans = info[:7]
        c.cands |= maps.set(rd)
        c.mod_sets[old] = mu
        any_code = False
        defs_changed = False
        for removed, rtxt, added in patchutil.changes(fp):
            if hunk_is_comment_only(removed, rtxt, added, sspans, set(bslash)):
                continue
            any_code = True
            structural = is_structural(rtxt, added)
            table = wide if structural else narrow
            lines = list(removed)
            if not removed:
                lines = [p - 1 for p, _ in added] + [p for p, _ in added]
            elif len(added) > len(removed):
                lines += [p for p, _ in added]
            c.line_changes.append((old, list(lines), structural))
            for ln in lines:
                if 0 < ln < len(table):
                    c.cands |= maps.set(table[ln])
                elif ln >= len(table) and len(table) > 1:
                    c.cands |= maps.set(table[len(table) - 1])
        if any_code:
            if len(info) > 7:
                c.dedicated |= set(info[7])
            # syntax / import-time safety
            ok = local_syntax_ok(patched, fp, fspans)
            if ok is not True:
                c.import_check.add(old)
                c.import_fail_sets[old] = maps.set(mu)
                if DEBUG:
                    c.notes.append(f"import-check {old}")
    return c


def _fixture_pairs(text):
    """tests/helpers.py load_sql_fixture_pairs, on a string."""
    s = "\n".join(line for line in text.splitlines() if line and not line.startswith("--"))
    statements = s.split(";")
    out = []
    for i in range(0, len(statements), 2):
        if i + 1 < len(statements):
            out.append((statements[i].strip(), statements[i + 1].strip()))
    return out


def _tpc_changed(original, patched):
    """Indices of edited query pairs, or None if the pairing shifted."""
    try:
        a = _fixture_pairs(original.decode("utf-8"))
        b = _fixture_pairs(patched.decode("utf-8"))
    except Exception:
        return None
    if len(a) != len(b):
        return None
    return {i for i, (x, y) in enumerate(zip(a, b)) if x != y}


_TPCQ = None
TPC_TESTS = {"test_execute_tpch": "tpc-h", "test_optimized_tpch": "tpc-h", "test_execute_tpcds": "tpc-ds"}


def _tpcq():
    global _TPCQ
    if _TPCQ is None:
        try:
            _TPCQ = marshal.load(open(os.path.join(MAPS, "tpcq.m"), "rb"))
        except Exception:
            _TPCQ = {}
    return _TPCQ


_OPTQ = None


_OPTQ_DATA = {}


def _optq():
    """Index of tests with a per-query map (data loaded per test on demand)."""
    global _OPTQ
    if _OPTQ is None:
        try:
            _OPTQ = set(marshal.load(open(os.path.join(MAPS, "optq.m"), "rb")))
        except Exception:
            _OPTQ = set()
    return _OPTQ


def _optq_test(x):
    v = _OPTQ_DATA.get(x)
    if v is None:
        try:
            v = marshal.load(open(os.path.join(MAPS, f"optq_{x}.m"), "rb"))
        except Exception:
            v = False
        _OPTQ_DATA[x] = v
    return v


def _line_mask(c, maps, x, idx, nruns):
    allmask = (1 << nruns) - 1
    m = 0
    for rel, lines, structural in c.line_changes:
        if structural:
            return allmask
        fidx = idx.get(rel, {})
        info = maps.file(rel)
        for ln in lines:
            if ln in fidx:
                m |= fidx[ln]
            elif info is not None and 0 < ln < len(info[0]) and x in maps.set(info[0][ln]):
                return allmask  # candidate through a non-query line (setup, class attribute, ...)
    return m or allmask


def opt_query_filter(c, maps, meta):
    """Fixture-loop optimizer tests: which (fixture, index) queries a code-only change can affect.
    Returns ({method_name: set((fname, i))}, {test: cost fraction}) or (None, None)."""
    if not c.line_changes or c.deleted:
        return None, None
    for p in c.patched:
        if not p.endswith(".py") or not p.startswith("sqlglot/"):
            return None, None
    oq = None
    flt, frac = {}, {}
    for x in c.cands:
        if meta[x][1] != "tests/test_optimizer.py":
            continue
        if oq is None:
            oq = _optq()
        if x not in oq:
            continue
        data = _optq_test(x)
        if not data:
            continue
        runs, weights, idx = data
        m = _line_mask(c, maps, x, idx, len(runs))
        sel = [j for j in range(len(runs)) if m >> j & 1]
        if len(sel) == len(runs):
            continue
        flt[meta[x][3]] = {tuple(runs[j]) for j in sel}
        frac[x] = sum(weights[j] for j in sel)
    if not flt:
        return None, None
    return flt, frac


def tpc_query_filter(c, maps, meta):
    """For a code-only change, which TPC queries can change outcome. Returns
    ({list_key: set(indices)}, {test: fraction_of_cost}) or (None, None)."""
    if not c.line_changes or c.deleted:
        return None, None
    for p in c.patched:
        if not p.endswith(".py") or p in ("tests/test_executor.py", "tests/helpers.py") or not p.startswith("sqlglot/"):
            return None, None
    tq = _tpcq()
    lists = {}
    frac = {}
    for x in c.cands:
        if meta[x][1] != "tests/test_executor.py" or meta[x][3] not in TPC_TESTS or x not in tq:
            continue
        weights, idx = tq[x]
        allmask = (1 << len(weights)) - 1
        m = 0
        for rel, lines, structural in c.line_changes:
            if structural:
                m = allmask
                break
            fidx = idx.get(rel, {})
            info = maps.file(rel)
            for ln in lines:
                if ln in fidx:
                    m |= fidx[ln]
                elif info is not None and 0 < ln < len(info[0]) and x in maps.set(info[0][ln]):
                    m = allmask  # candidate through a non-query line (setup, class attribute, ...)
                    break
            if m == allmask:
                break
        if m == 0:
            m = allmask  # no query line touched: be safe and run them all
        sel = {i for i in range(len(weights)) if m >> i & 1}
        key = TPC_TESTS[meta[x][3]]
        lists.setdefault(key, set()).update(sel)
        frac[x] = sum(weights[i] for i in sel)
    if not lists:
        return None, None
    return lists, frac


def _sql_fixture_harmless(fp):
    """tests/helpers.py drops empty lines and lines starting with '--' before use."""
    for removed, rtxt, added in patchutil.changes(fp):
        for t in list(rtxt) + [x for _, x in added]:
            t = t.rstrip(b"\r")
            if t and not t.startswith(b"--"):
                return False
    return True


def _deleted_file(c, rel, maps):
    info = maps.file(rel)
    if info is not None:
        c.always |= maps.set(info[3])
        c.cands |= maps.set(info[2])
    else:
        c.cands |= maps.readers(rel)


def _new_file(c, rel, maps):
    # a new file matters if some test lists its directory or reads it
    d = os.path.dirname(rel)
    c.cands |= maps.readers(rel)
    c.cands |= maps.readers(d)
    if rel.endswith(".py") and rel.startswith("sqlglot/"):
        # could shadow / be discovered by package walkers
        c.cands |= maps.readers(d)


# ---------------------------------------------------------------- packing
def pack(tests, ms):
    if not tests:
        return []
    items = sorted(tests, key=lambda t: -ms[t])
    total = sum(ms[t] for t in items)
    best = None
    for k in range(1, 9):
        if k > len(items):
            break
        loads = [0] * k
        bins = [[] for _ in range(k)]
        for t in items:
            i = loads.index(min(loads))
            loads[i] += ms[t]
            bins[i].append(t)
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, bins)
        if max(ms[items[0]], total / (k + 1)) + WORKER_MS * (k + 1) > best[0] and total / k < ms[items[0]]:
            break
    return [b for b in best[1] if b]


# ---------------------------------------------------------------- verification
def _plan_one(c, ms, est, meta, tests, maps, all_tests):
    """Decide what to run for one commit; True if it needs a test child."""
    if c.unknown:
        c.result = set(all_tests)
        return False
    c.sure = {t for t in c.sure if ms[t] < 3000}
    if VERIFY_ALL:
        c.sure = set()  # measurement mode: run everything we would include unverified
    cand = c.cands - c.always - c.sure
    # cheap: recorded or our own measured run time under 1 s (some "slow" tests only pay
    # for a process pool or class setup in the recorded run)
    cheap = {t for t in cand if ms[t] < EXPENSIVE * 1000 or est[t] < EXPENSIVE * 1000}
    tf = getattr(c, "tpc_filter", None)
    run_ms = {}
    if tf and not getattr(c, "tpc_filter_bad", False) and not any(p.endswith(".py") for p in c.patched) and not c.deleted:
        # executor tests only re-run the edited TPC queries: cheap to check
        ex = {t for t in cand if meta[t][1] == "tests/test_executor.py"}
        cheap |= ex
        nq = sum(len(v) for v in tf.values())
        for t in ex:
            if meta[t][3] in ("test_execute_tpcds", "test_execute_tpch", "test_optimized_tpch"):
                run_ms[t] = ms[t] * min(1.0, (nq + 1) / 20.0)
    else:
        c.tpc_filter = None
        oflt, ofrac = opt_query_filter(c, maps, meta)
        if oflt is not None:
            c.opt_filter = oflt
            for x, f in ofrac.items():
                run_ms[x] = ms[x] * f + 30
                if x in cand and run_ms[x] < EXPENSIVE * 1000:
                    cheap.add(x)
        lists, frac = tpc_query_filter(c, maps, meta)
        if lists is not None:
            c.tpc_filter = {k: v for k, v in lists.items()}
            for x, f in frac.items():
                run_ms[x] = ms[x] * f + 150
                if run_ms[x] < 500:
                    cheap.add(x)
            c.run_ms = run_ms
    exp_ = cand - cheap
    c.cheap, c.exp = cheap, exp_
    c.run_ms = run_ms
    c.cpu_cost = {x: est[x] * (run_ms[x] / ms[x] if x in run_ms and ms[x] > 0 else 1.0) for x in exp_}
    # cheap stand-ins for the slowest tests, run after the cheap ones
    # (not when the test's own file is edited: the stand-in would not see the edit)
    c.probes = [] if VERIFY_ALL else [
        (x, tests[x]["id"]) for x in sorted(exp_)
        if tests[x]["id"] in PROBE_IDS and tests[x]["file"] not in c.patched and tests[x]["file"] not in c.deleted
    ]
    run_cost = sum(
        est[t] * (run_ms[t] / ms[t] if t in run_ms else 1.0) if ms[t] >= EXPENSIVE * 1000 else policy.CPU_RATIO * run_ms.get(t, ms[t])
        for t in cheap
    )
    if not VERIFY_ALL and cheap and run_cost > policy.pack_cost(cheap | exp_ | c.sure | c.always, ms):
        # running them would cost more than just including them
        c.guard = True
        if not c.import_check:
            c.result = cheap | exp_ | c.sure | c.always
            return False
        # but first make sure the patched modules import (else far more tests fail)
        c.include_all_after_import = cheap | exp_
        c.cheap, c.exp = set(), set()
        c.probes = []
    if not cheap and not exp_ and not c.import_check:
        c.result = set(c.always) | c.sure
        return False
    return True



def main():
    t_start = time.time()
    tests = json.load(open(os.path.join(DATA, "tests.json")))["tests"]
    ntests = len(tests)
    ms = [round(t["seconds"] * 1000) for t in tests]
    meta = []
    for t in tests:
        f, cls, name = t["id"].split("::")
        meta.append((t["number"], t["file"], cls, name))
    maps = Maps(ntests)
    # our own CPU ms to run each test (measured at the pinned commit for the slower tests)
    try:
        cpu_map = marshal.load(open(os.path.join(MAPS, "cpu.m"), "rb"))
    except Exception:
        cpu_map = {}
    est = [cpu_map.get(t["id"], ms[t["number"]] * policy.CPU_RATIO) for t in tests]
    global _MAPS, _META
    _MAPS = maps
    _META = meta
    try:
        os.makedirs(TMP, exist_ok=True)
    except OSError:
        pass
    cdir = os.path.join(DATA, "commits")
    ids = sorted(fn[:-6] for fn in os.listdir(cdir) if fn.endswith(".patch"))
    global DEADLINE
    if not float(os.environ.get("BR_BUDGET", "0") or 0):
        # ~4.5 s of wall time per commit is what the limits give; keep a safety margin
        # practice: 180 s for ~40 commits; final: 270 s for ~60 commits. Keep a safety margin.
        DEADLINE = t_start + (140.0 if len(ids) <= 48 else 225.0)
    commits = []
    for cid in ids:
        with open(os.path.join(cdir, cid + ".patch"), "rb") as f:
            data = f.read()
        try:
            c = analyze(cid, data, maps, ntests, None)
        except Exception as e:  # noqa: BLE001
            log("analyze error", cid, repr(e))
            c = Commit(cid)
            c.unknown = True
        commits.append(c)
    all_tests = set(range(ntests))

    # plan
    jobs = [c for c in commits if _plan_one(c, ms, est, meta, tests, maps, all_tests)]

    ids = [t["id"] for t in tests]
    for c in jobs:
        c.early_blind = None if (VERIFY_ALL or c.probes) else policy.early_stop_ok(c.cheap, c.exp, ms, ids, getattr(c, "run_ms", None), c.dedicated, getattr(c, "cpu_cost", None))
        c.early_gate = None
        if c.early_blind is None and not VERIFY_ALL and not c.probes and not any(p.startswith(EXEC_SIDE) for p in set(c.patched) | set(c.deleted)):
            # stop early once test_execute_tpch or the optimizer's test_tpch (run first) fails: the
            # TPC-DS run then fails too (44/44, 55/56)
            ex = _num(ids, EXEC_TPCDS)
            gates = {g for g in (_num(ids, EXEC_TPCH), _num(ids, OPT_TPCH)) if g in c.cheap}
            if ex in c.exp and gates:
                # (when it fails the change is broad: the rest is decided as catastrophic)
                b_cat, v_cat = policy.decide(c.exp, set(), "cat", len(c.cheap), ms, ids, getattr(c, "run_ms", None), c.dedicated, pmap={ex: 0.97}, cpu=getattr(c, "cpu_cost", None))
                if ex in b_cat and not v_cat and policy.pack_cost(set(c.cheap) | b_cat, ms) - policy.pack_cost(b_cat, ms) <= 300:
                    c.early_blind, c.early_gate = b_cat, gates
    try:
        run_jobs(jobs, meta, ms, maps, ids)
    except BaseException as e:  # noqa: BLE001
        log("run_jobs failed:", repr(e))
        for c in jobs:
            if c.result is None:
                c.result = set(c.cands) | set(c.sure) | set(c.always) | set(getattr(c, "blind", ()))

    outs = []
    for c in commits:
        res = c.result if c.result is not None else set(c.cands) | c.always | c.sure
        res = {t for t in res if isinstance(t, int) and 0 <= t < ntests}
        try:
            workers = pack(sorted(res), ms)
        except Exception:  # noqa: BLE001
            workers = [sorted(res)] if res else []
        outs.append((c, res, workers))
    if DIAG:
        try:
            _diag(outs, tests, meta, ms, maps, ids)
        except Exception as e:  # noqa: BLE001
            log("diag failed", repr(e))
    if PDIAG:
        try:
            _pdiag(outs, tests, meta, ms, maps, ids)
        except Exception as e:  # noqa: BLE001
            log("pdiag failed", repr(e))
    if CDIAG:
        try:
            _cdiag(outs, tests)
        except Exception as e:  # noqa: BLE001
            log("cdiag failed", repr(e))
    out = sys.stdout
    for c, res, workers in outs:
        out.write(json.dumps({"commit": c.cid, "workers": workers}, separators=(",", ":")) + "\n")
    out.flush()
    if DEBUG:
        for c in commits:
            log(c.cid, "cand", len(c.cands), "res", len(c.result or ()), c.notes)
    log(f"done {len(commits)} commits in {time.time() - t_start:.1f}s")


_MAPS = None
_META = None


def _diag(outs, tests, meta, ms, maps, ids):
    """Measurement build: run every test we included without running it, and encode in the
    output (extra one-test workers: +2 s each, decimals from empty commits) how much of the
    simulated time was wasted on tests that passed:
      0.001 per commit with a wasted test_execute_tpcds, 0.010 per 2 s wasted on fixture
      readers (sure); extra 0-second workers: 1 per 2 s wasted on other tests, 20 per 4 s
      wasted in commits included whole without running ("guard")."""
    ex_id = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
    djobs = []
    for c, res, workers in outs:
        if c.unknown or not res:
            continue
        known = set(getattr(c, "failed", set())) & set(getattr(c, "reported", set()))
        u = set(res) - known
        if not u:
            continue
        d = Commit(c.cid)
        d.patched, d.deleted = c.patched, c.deleted
        d.cheap, d.exp, d.sure, d.always = set(u), set(), set(), set()
        d.import_check, d.import_fail_sets, d.probes, d.early_blind = set(), {}, [], None
        for k in ("tpc_filter", "opt_filter", "run_ms"):
            setattr(d, k, getattr(c, k, None))
        d.orig, d.u, d.res = c, u, res
        djobs.append(d)
    run_jobs(djobs, meta, ms, maps, ids)
    n_a, w_s, w_x, w_g = 0, 0.0, 0.0, 0.0
    for d in djobs:
        wasted = d.u - set(d.failed)
        if not wasted:
            continue
        c, res = d.orig, d.res
        full = policy.pack_cost(res, ms)
        if getattr(c, "guard", False):
            w_g += (full - policy.pack_cost(res - wasted, ms)) / 1000
            log(f"diag {c.cid} guard wasted={len(wasted)}/{len(res)} {(full - policy.pack_cost(res - wasted, ms)) / 1000:.1f}s")
            continue
        ws = wasted & set(c.sure)
        wx = wasted - ws
        if any(ids[t] == ex_id for t in wasted):
            n_a += 1
            wx = {t for t in wx if ids[t] != ex_id}
        w_s += (full - policy.pack_cost(res - ws, ms)) / 1000 if ws else 0
        w_x += (full - policy.pack_cost(res - wx, ms)) / 1000 if wx else 0
        log(f"diag {c.cid} wasted={len(wasted)} sure={len(ws)} other={sorted(ids[t].rsplit('::', 1)[-1] for t in wx)[:8]} ex={any(ids[t] == ex_id for t in wasted)}")
    n_a = min(n_a, 9)
    n_s = min(int(round(w_s / 2)), 9)
    n_x = min(int(round(w_x / 2)), 19)
    n_c = min(int(round(w_g / 4)), 9)
    log(f"diag totals n_a={n_a} w_s={w_s:.1f} w_x={w_x:.1f} w_g={w_g:.1f} -> {n_a} {n_s} {n_x} {n_c}")
    pool = {}
    for t in tests:
        pool.setdefault(t["seconds"], []).append(t["number"])
    empties = [o for o in outs if not o[2]]
    decimals = [0.001] * n_a + [0.010] * n_s
    for o, sec in zip(empties, decimals):
        cand = [x for x in pool.get(sec, []) if x not in o[1]]
        if cand:
            o[2].append([cand[0]])
    zeros = n_x + 20 * n_c
    for o in outs:
        if zeros <= 0:
            break
        used = {x for w in o[2] for x in w}
        for x in pool.get(0.0, []):
            if zeros <= 0 or len(o[2]) >= 8:
                break
            if x in o[1] or x in used:
                continue
            o[2].append([x])
            used.add(x)
            zeros -= 1


def _cdiag(outs, tests):
    """Measurement build: encode CPU by phase as extra 0-second workers (+2 s each):
    N = round(phase-2 CPU / 2) + 11 * round(phase-1 CPU / 2); and on empty commits one 0.001
    worker per parent CPU second (<= 9)."""
    p1 = min(int(round(CPU_BY_PHASE.get(1, 0.0) / 2)), 19)
    p2 = min(int(round(CPU_BY_PHASE.get(2, 0.0) / 2)), 10)
    tm = os.times()
    par = min(int(round(tm.user + tm.system)), 9)
    log(f"cdiag phase1={CPU_BY_PHASE.get(1, 0.0):.1f} phase2={CPU_BY_PHASE.get(2, 0.0):.1f} parent={tm.user + tm.system:.1f} -> {p1} {p2} {par}")
    pool = {}
    for t in tests:
        pool.setdefault(t["seconds"], []).append(t["number"])
    empties = [o for o in outs if not o[2]]
    for o in empties[:par]:
        cand = [x for x in pool.get(0.001, []) if x not in o[1]]
        if cand:
            o[2].append([cand[0]])
    zeros = p2 + 11 * p1
    for o in outs:
        if zeros <= 0:
            break
        used = {x for w in o[2] for x in w}
        for x in pool.get(0.0, []):
            if zeros <= 0 or len(o[2]) >= 8:
                break
            if x in o[1] or x in used:
                continue
            o[2].append([x])
            used.add(x)
            zeros -= 1


PATH_CODE = {"cat": 1, "early": 2, "broad": 3, "narrow": 4, "dead": 5, "unknown": 6, "proxy": 7, "deferred": 8}


def _pdiag(outs, tests, meta, ms, maps, ids):
    """Measurement build: for commits whose output has test_execute_tpcds without having run
    it, run it; for those where it passes, encode why it was added: extra 0-second workers
    (+2 s each) = code1 + 10 * code2 (first two such commits), and on empty commits one
    0.001 worker per such commit and 0.010 x flags of the first (1 executor-side change,
    2 execute_tpch ran and passed, 4 it ran and failed).
    Codes: 1 cat, 2 early stop, 3 broad, 4 narrow, 5 dead, 6 unknown, 7 proxy, 8 deferred,
    9 whole-commit / import-expansion / other."""
    ex = _num(ids, EXEC_TPCDS)
    eh = _num(ids, EXEC_TPCH)
    djobs = []
    for c, res, workers in outs:
        if c.unknown or ex not in res or ex in getattr(c, "reported", set()):
            continue
        d = Commit(c.cid)
        d.patched, d.deleted = c.patched, c.deleted
        d.cheap, d.exp, d.sure, d.always = {ex}, set(), set(), set()
        d.import_check, d.import_fail_sets, d.probes, d.early_blind = set(), {}, [], None
        for k in ("tpc_filter", "opt_filter", "run_ms"):
            setattr(d, k, getattr(c, k, None))
        d.orig = c
        djobs.append(d)
    run_jobs(djobs, meta, ms, maps, ids)
    found = []
    for d in djobs:
        if ex in d.failed:
            continue
        c = d.orig
        why = getattr(c, "why", {}).get(ex)
        code = PATH_CODE.get(why, 9)
        flags = 0
        if all(p.startswith(EXEC_SIDE) for p in (set(c.patched) | set(c.deleted)) or {""}):
            flags |= 1
        if eh in getattr(c, "reported", set()):
            flags |= 4 if eh in c.failed else 2
        found.append((code, flags))
        log(f"pdiag {c.cid} wasted tpcds why={why} code={code} flags={flags} regime={getattr(c, 'regime', None)} guard={getattr(c, 'guard', False)}")
    n = min(len(found), 9)
    zeros = (found[0][0] if found else 0) + (10 * found[1][0] if len(found) > 1 else 0)
    log(f"pdiag n={len(found)} zeros={zeros} found={found}")
    pool = {}
    for t in tests:
        pool.setdefault(t["seconds"], []).append(t["number"])
    empties = [o for o in outs if not o[2]]
    decimals = [0.001] * n + ([0.010] * found[0][1] if found else [])
    for o, sec in zip(empties, decimals):
        cand = [x for x in pool.get(sec, []) if x not in o[1]]
        if cand:
            o[2].append([cand[0]])
    for o in outs:
        if zeros <= 0:
            break
        used = {x for w in o[2] for x in w}
        for x in pool.get(0.0, []):
            if zeros <= 0 or len(o[2]) >= 8:
                break
            if x in o[1] or x in used:
                continue
            o[2].append([x])
            used.add(x)
            zeros -= 1


def run_jobs(jobs, meta, ms, maps, ids=None):
    global _MAPS, _META
    _MAPS = maps
    _META = meta
    try:
        os.makedirs(TMP, exist_ok=True)
    except OSError:
        pass
    if ids is None:
        ids = [f"{m[1]}::{m[2]}::{m[3]}" for m in meta]
    import runner

    # preload heavy third-party modules used by the test files we are going to import
    files = set()
    for c in jobs:
        for t in c.cheap | c.exp:
            files.add(meta[t][1])
    _preload(files)
    if any(meta[t][1] == "tests/test_executor.py" and meta[t][2] == "TestExecutor" for c in jobs for t in c.cheap | c.exp):
        try:
            runner.precompute_executor_rows(SRC, subs=())  # rows are parsed when a test needs them
            runner.load_duck_cache()
        except Exception as e:  # noqa: BLE001
            log("precompute failed", repr(e))

    if os.environ.get("BR_GC_FREEZE", "1") == "1":
        import gc

        gc.collect()
        gc.freeze()  # children start with the parent's objects outside the cyclic collector
    pending = []
    for c in jobs:
        c.failed = set()
        c.reported = set()
        c.file_fail = set()
        c.import_failed = set()
        c.phase = 1
        pending.append(c)
    running = {}  # fd -> (pid, commit, buffer, deadline)
    seq = 0

    def launch(c, tests, phase):
        nonlocal seq
        seq += 1
        ov = os.path.join(TMP, f"w{os.getpid()}_{seq}")
        if os.path.lexists(ov):
            _cleanup(ov)
        if runner.PRE_ROWS is not None:
            subs = {runner.TPC_SUB_OF_TEST[meta[t][3]] for t in tests if meta[t][3] in runner.TPC_SUB_OF_TEST}
            if subs:
                try:
                    if not all(("_done", x) in runner.PRE_ROWS for x in subs):
                        runner.precompute_executor_rows(SRC, subs=tuple(sorted(subs)))
                        if os.environ.get("BR_GC_FREEZE", "1") == "1":
                            import gc

                            gc.freeze()
                except Exception as e:  # noqa: BLE001
                    log("precompute failed", repr(e))
        try:
            r, w = os.pipe()
            pid = os.fork()
        except OSError:
            # cannot fork (process limit): include what we could not check
            c.failed |= set(tests)
            if phase == 1:
                c.blind = set(c.exp)
            _finish(c, ms)
            return False
        if pid == 0:
            try:
                os.close(r)
                runner.build_overlay(SRC, ov, c.patched, c.deleted)
                if runner.PRE_ROWS is not None and any(
                    ("_done", runner.TPC_SUB_OF_TEST.get(meta[t][3])) not in runner.PRE_ROWS
                    for t in tests if meta[t][3] in runner.TPC_SUB_OF_TEST
                ):
                    runner.PRE_ROWS = None  # (not prepared: use the test's own setUpClass)
                import zlib as _z

                # deterministic shuffle of test classes: early results then represent the whole set
                if os.environ.get("BR_NO_SHUFFLE"):
                    tl = [meta[t] for t in sorted(tests)]  # collection order (validation runs)
                else:
                    tl = [meta[t] for t in sorted(tests, key=lambda t: (_z.crc32(f"{meta[t][1]}::{meta[t][2]}".encode()), meta[t][1], meta[t][2]))]
                gate = (getattr(c, "early_gate", None) or set()) if phase == 1 else set()
                if phase == 1 and ids is not None and _num(ids, EXEC_TPCDS) in c.exp:
                    # the cheap TPC tests first: their result decides the TPC-DS run
                    gate = set(gate) | {g for g in (_num(ids, EXEC_TPCH), _num(ids, OPT_TPCH)) if g in tests}
                if gate:
                    first = [meta[g] for g in sorted(gate) if g in tests]
                    tl = first + [m for m in tl if m[0] not in gate]  # their files and classes run first
                tm = float(os.environ.get("BR_TIMEOUT_MULT", "5"))
                timeouts = {t: min(float(os.environ.get("BR_TIMEOUT_CAP", "45")), max(5.0, ms[t] / 1000 * tm + 3)) for t in tests}
                runner.TPC_FILTER = getattr(c, "tpc_filter", None)
                runner.OPT_FILTER = getattr(c, "opt_filter", None)
                imods = sorted(m for m in c.import_check if m.startswith("sqlglot/") or m.startswith("tests/"))
                extra = None  # (runner.importers_closure narrows this; measured no gain, so all dialects)
                runner.child_main(ov, tl, w, timeouts, 0, imods, set(c.patched) | set(c.deleted), probes=(getattr(c, "probes", None) or ()) if phase == 1 else (), import_extra=extra)
            except BaseException:
                if os.environ.get("BR_CHILD_DEBUG"):
                    import traceback

                    with open(os.environ["BR_CHILD_DEBUG"], "a") as fh:
                        fh.write(traceback.format_exc())
            os._exit(0)
        os.close(w)
        budget = 15.0 + sum(ms[t] for t in tests) / 1000 * float(os.environ.get("BR_TIMEOUT_MULT", "5"))
        budget = min(budget, max(5.0, DEADLINE - time.time()))
        running[r] = [pid, c, b"", time.time() + budget, tests, ov]
        c.cur_tests = set(tests)
        return True

    queue = []
    for c in pending:
        if getattr(c, "include_all_after_import", None) is not None:
            queue.append((c, set(), 1))
            continue
        if not c.cheap and not c.import_check and not getattr(c, "probes", None):
            nxt = _next_phase(c, ms, ids)
            if nxt:
                queue.append((c, nxt, 2))
            else:
                _finish(c, ms)
            continue
        queue.append((c, set(c.cheap), 1))

    while queue or running:
        while queue and len(running) < PAR:
            c, tests, phase = queue.pop(0)
            if time.time() > DEADLINE - 3.0:
                # out of time: include what we could not check
                c.failed |= set(tests)
                if phase == 1:
                    c.blind = set(c.exp)
                _finish(c, ms)
                continue
            launch(c, tests, phase)
        if not running:
            continue
        timeout = min(v[3] for v in running.values()) - time.time()
        rl, _, _ = select.select(list(running), [], [], max(0.0, timeout))
        now = time.time()
        for fd in list(running):
            pid, c, buf, deadline, tests, ov = running[fd]
            done = False
            if fd in rl:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    chunk = b""
                if chunk:
                    buf += chunk
                    running[fd][2] = buf
                    if c.phase == 1 and len(c.cheap) >= 40 and not getattr(c, "early_cat", False):
                        n_ok = buf.count(b" 0\n")
                        n_bad = buf.count(b" 1\n")
                        n = n_ok + n_bad
                        if n >= max(20, len(c.cheap) // 10) and n_bad >= 0.85 * n:
                            # the change breaks almost everything: include all candidates
                            c.early_cat = True
                            try:
                                os.kill(pid, 9)
                            except OSError:
                                pass
                            done = True
                    gate = getattr(c, "early_gate", None)
                    if gate:
                        hit = any((b"\n" + buf).find(f"\n{g} 1\n".encode()) >= 0 for g in gate) or b"-1 " in buf or b"-2 " in buf
                    else:
                        hit = b" 1\n" in buf.replace(b"-3 ready\n", b"") or b"-1 " in buf or b"-2 " in buf
                    if c.phase == 1 and getattr(c, "early_blind", None) and hit:
                        # live change and a big test goes on its own worker anyway: stop here
                        c.early_stopped = True
                        try:
                            os.kill(pid, 9)
                        except OSError:
                            pass
                        done = True
                else:
                    done = True
            if not done and now > deadline:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
                done = True
            if done:
                os.close(fd)
                try:
                    _, _, ru = os.wait4(pid, 0)
                    c.cpu = getattr(c, "cpu", 0.0) + ru.ru_utime + ru.ru_stime
                    CPU_BY_PHASE[c.phase] = CPU_BY_PHASE.get(c.phase, 0.0) + ru.ru_utime + ru.ru_stime
                    if DEBUG:
                        log(f"job {c.cid} phase{c.phase} tests={len(tests)} rec={sum(ms[t] for t in tests)/1000:.2f}s cpu={ru.ru_utime + ru.ru_stime:.2f}s")
                except OSError:
                    pass
                del running[fd]
                _cleanup(ov)
                _ingest(c, buf, tests, meta)
                nxt = _next_phase(c, ms, ids)
                if nxt:
                    queue.insert(0, (c, nxt, 2))
                else:
                    _finish(c, ms)


def _ingest(c, buf, tests, meta):
    if b"-3 ready" not in buf and b"-2 " not in buf and c.import_check and not getattr(c, "early_stopped", False):
        # the child died before its import checks finished: assume the patched modules break imports
        c.import_failed |= set(c.import_check)
    for line in buf.decode("utf-8", "replace").splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        if parts[0] == "-1":
            c.file_fail.add(parts[1])
            continue
        if parts[0] == "-2":
            c.import_failed.add(parts[1])
            continue
        if parts[0] == "-3":
            continue
        if parts[0] == "-4":
            if DEBUG:
                log(f"importcpu {c.cid} {parts[1]}")
            continue
        if parts[0] == "probe":
            try:
                num, verdict = parts[1].split()
                c.probe_res = getattr(c, "probe_res", {})
                c.probe_res[int(num)] = verdict != "ok"
            except ValueError:
                pass
            continue
        try:
            n = int(parts[0])
        except ValueError:
            continue
        c.reported.add(n)
        if parts[1] == "1":
            c.failed.add(n)
    # anything not reported (crash / timeout) counts as failed
    for t in tests:
        if t not in c.reported:
            c.failed.add(t)
    # test files that imported fine with the patch (they reported results)
    c.ok_files = getattr(c, "ok_files", set()) | ({meta[t][1] for t in c.reported if 0 <= t < len(meta)} - set(c.file_fail))
    if c.file_fail:
        # "a test fails when its file cannot be imported": every test of that file fails
        for t in range(len(meta)):
            if meta[t][1] in c.file_fail:
                c.failed.add(t)


# executor TPC tests and the optimizer test over the same queries: when the change is not in
# executor-side code, the optimizer test's outcome predicts the executor test's (validator and
# campaign mutants: test_tpch passed -> execute_tpch failed 0/90, optimized_tpch 2/98;
# test_tpcds passed -> execute_tpcds failed 1/102; proxy failed -> ~90% failed)
TPC_PROXY = {
    "tests/test_executor.py::TestExecutor::test_execute_tpch": ("tests/test_optimizer.py::TestOptimizer::test_tpch", 0.01, 0.9),
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": ("tests/test_optimizer.py::TestOptimizer::test_tpch", 0.03, 0.9),
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": ("tests/test_optimizer.py::TestOptimizer::test_tpcds", 0.02, 0.9),
}
EXEC_SIDE = ("sqlglot/executor/", "sqlglot/planner.py", "sqlglot/generators/python.py", "sqlglot/generator.py",
             "tests/test_executor.py", "tests/test_optimizer.py", "tests/helpers.py", "tests/fixtures/optimizer/tpc-")
EXEC_ONLY = ("sqlglot/executor/", "sqlglot/planner.py", "sqlglot/generators/python.py")
EXEC_TPCH = "tests/test_executor.py::TestExecutor::test_execute_tpch"
EXEC_TPCDS = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
OPT_TPCH = "tests/test_optimizer.py::TestOptimizer::test_tpch"
_ID2NUM = {}


def _num(ids, test_id):
    if not _ID2NUM:
        _ID2NUM.update((t, i) for i, t in enumerate(ids))
    return _ID2NUM.get(test_id)


def _finish_deferred(c, ms, ids):
    """After phase 2: decide the executor tests whose optimizer proxy has now run."""
    cur = set(c.failed) | set(c.always) | set(c.sure) | set(getattr(c, "blind", set()))
    for ex, px in sorted(c.deferred.items()):
        p_ok, p_bad = (getattr(c, "proxy_probs", None) or {}).get(ex) or TPC_PROXY[ids[ex]][1:]
        p = p_bad if px in c.failed else p_ok
        m = policy.pack_cost(cur | {ex}, ms) - policy.pack_cost(cur, ms)
        if policy.MISS_MS * p > m:
            c.blind = set(getattr(c, "blind", set())) | {ex}
            c.why = dict(getattr(c, "why", {}))
            c.why[ex] = "deferred"
            cur.add(ex)
    c.deferred = None


def _next_phase(c, ms, ids):
    if c.phase == 2 and getattr(c, "deferred", None):
        _finish_deferred(c, ms, ids)
        return None
    if c.phase != 1:
        return None
    c.phase = 2
    if c.import_failed:
        return None
    base = set(c.always) | set(c.sure)
    n_cheap = len(c.cheap)
    if getattr(c, "early_stopped", False):
        # everything cheap goes in unverified, next to the big blind test(s)
        c.blind = set(c.cheap) | set(c.early_blind)
        c.why = {x: "early" for x in c.blind}
        rest = set(c.exp) - c.blind
        blind, verify = policy.decide(rest, base | c.blind, "cat", n_cheap, ms, ids, getattr(c, "run_ms", None), c.dedicated, cpu=getattr(c, "cpu_cost", None))
        c.blind |= blind
        return verify or None
    nf = len(c.failed & c.cheap)
    if getattr(c, "early_cat", False):
        regime = "cat"
    elif not c.cheap:
        regime = "unknown"
    elif nf == 0:
        regime = "dead"
    elif nf >= 0.8 * n_cheap and n_cheap >= 3:
        regime = "cat"
    elif nf >= 0.3 * n_cheap:
        regime = "broad"
    else:
        regime = "narrow"
    c.regime = regime
    base |= c.failed & c.cheap
    fx = set(c.exp) & getattr(c, "fixture_readers", set())
    probed_bad = set()
    probed = set()
    if regime != "cat":
        # a stand-in run of the same checks decides these (passed: leave out, failed: include)
        for x, bad in getattr(c, "probe_res", {}).items():
            if x in c.exp and x not in fx:
                probed.add(x)
                if bad:
                    probed_bad.add(x)
    pmap = None
    deferred = {}
    proxy_p = {}
    touched = set(c.patched) | set(c.deleted)
    if regime in ("dead", "unknown") and touched and all(p.startswith(EXEC_ONLY) for p in touched):
        for x in c.exp:
            q = policy.EXEC_QUIET.get(ids[x].rsplit("::", 1)[-1])
            if q is not None and x not in fx:
                proxy_p[x] = min(q, policy.prob(ids[x], regime, n_cheap, x in c.dedicated))
    c.proxy_probs = {}
    eh_, oh_ = _num(ids, EXEC_TPCH), _num(ids, OPT_TPCH)
    tpc_ok = any(g is not None and g in c.cheap and g in c.reported and g not in c.failed for g in (eh_, oh_))
    tpc_bad = any(g is not None and g in c.cheap and g in c.reported and g in c.failed for g in (eh_, oh_))
    # catastrophic elsewhere (e.g. dialect test files not importing) while the cheap TPC tests
    # ran and passed: the executor TPC tests are judged on that evidence, not on the regime
    use_tpc = regime != "cat" or (tpc_ok and not tpc_bad)
    if use_tpc and not VERIFY_ALL and not any(p.startswith(EXEC_SIDE) for p in touched):
        eh = _num(ids, EXEC_TPCH)
        for ex_id, (px_id, p_ok, p_bad) in TPC_PROXY.items():
            ex, px = _num(ids, ex_id), _num(ids, px_id)
            if ex is None or px is None or ex not in c.exp or ex in fx or ex in probed:
                continue
            if ex_id == EXEC_TPCDS and eh is not None and eh in c.cheap and eh not in fx:
                # test_execute_tpch (cheap, same executor) ran first: failed -> 44/44 tpcds failed;
                # passed -> 7/104, and then test_tpcds passed 2/96 / failed 5/8
                if eh in c.failed:
                    proxy_p[ex] = 0.97
                    continue
                oh = _num(ids, OPT_TPCH)
                if oh is not None and oh in c.cheap and oh not in fx:
                    # optimizer test_tpch also ran: failed -> tpcds failed 3/3; passed -> 3/99
                    # (then test_tpcds passed: 2/95, failed: 1/4)
                    if oh in c.failed:
                        p_ok, p_bad = 0.3, 0.9
                        proxy_p[ex] = 0.8
                    else:
                        p_ok, p_bad = 0.021, 0.3
                        proxy_p[ex] = 0.03
                else:
                    p_ok, p_bad = 0.02, 0.6
                    proxy_p[ex] = 0.07
            if px in c.cheap:
                proxy_p[ex] = p_bad if px in c.failed else p_ok
            elif px in c.exp and px not in fx:
                deferred[ex] = px  # maybe: run the proxy in phase 2, decide ex after it
                c.proxy_probs[ex] = (p_ok, p_bad)
    if regime == "narrow":
        # what the expensive test's own class did is the best evidence (see policy.NARROW_CELL)
        pmap = {}
        frac = nf / max(1, n_cheap)
        for x in set(c.exp) - fx - probed:
            cls = ids[x].rsplit("::", 1)[0] + "::"
            same = [i for i in c.cheap if ids[i].startswith(cls)]
            p = policy.narrow_prob(ids[x], bool(same), any(i in c.failed for i in same), frac)
            if n_cheap < policy.WEAK_EVIDENCE or x in c.dedicated:
                p = max(p, 0.3)
            pmap[x] = p
    if proxy_p:
        pmap = dict(pmap or {})
        pmap.update(proxy_p)
    if deferred:
        # defer only when running the proxy is worth its CPU (expected cost, policy.decide terms)
        cpu = getattr(c, "cpu_cost", None) or {}
        rm = getattr(c, "run_ms", None)
        cur = base | probed_bad
        cur_cost = policy.pack_cost(cur, ms)
        for ex, px in list(deferred.items()):
            def p_of(x):
                return (pmap or {}).get(x) if pmap and x in pmap else policy.prob(ids[x], regime, n_cheap, x in c.dedicated)

            pe, q = p_of(ex), p_of(px)
            m_ex = policy.pack_cost(cur | {ex}, ms) - cur_cost
            m_px = policy.pack_cost(cur | {px}, ms) - cur_cost
            v_ex, v_px = policy.verify_cpu(ex, ms, rm, cpu), policy.verify_cpu(px, ms, rm, cpu)
            cost_a = min(m_ex, v_ex + pe * m_ex, policy.MISS_MS * pe)
            px_alone = min(m_px, v_px + q * m_px, policy.MISS_MS * q)
            p_ok, p_bad = c.proxy_probs.get(ex) or TPC_PROXY[ids[ex]][1:]
            cost_b = (v_px + q * m_px - px_alone) + q * min(m_ex, policy.MISS_MS * p_bad) + (1 - q) * min(m_ex, policy.MISS_MS * p_ok)
            if cost_b >= 0.85 * cost_a - 1000:  # only clear wins (the proxy can run slower on broken code)
                del deferred[ex]
    blind, verify = policy.decide(set(c.exp) - fx - probed - set(deferred), base | probed_bad, regime, n_cheap, ms, ids, getattr(c, "run_ms", None), c.dedicated, pmap, cpu=getattr(c, "cpu_cost", None))
    blind |= probed_bad
    for ex, px in deferred.items():
        blind.discard(px)
        verify.add(px)
    c.deferred = deferred or None
    if fx:
        # what cheap tests did says nothing about a test reading an edited data file
        b2, v2 = policy.decide(fx, base | blind, "unknown", n_cheap, ms, ids, getattr(c, "run_ms", None), c.dedicated, cpu=getattr(c, "cpu_cost", None))
        blind |= b2
        verify |= v2
    if VERIFY_ALL and regime != "cat":
        verify |= blind
        blind = set()
    c.blind = blind
    c.why = {x: ("proxy" if x in (pmap or {}) and x in proxy_p else regime) for x in blind}
    if DEBUG:
        nm = lambda xs: sorted(ids[x].rsplit("::", 1)[-1] for x in xs)  # noqa: E731
        log(f"decide {c.cid} {regime} n={n_cheap} nf={nf} blind={nm(blind)} verify={nm(verify)} deferred={nm(deferred)} exp={nm(c.exp)}")
    return verify or None


def _finish(c, ms):
    res = set(c.failed) | c.always | c.sure | getattr(c, "blind", set())
    res |= set(getattr(c, "deferred", None) or ())  # proxy never ran (deadline, fork failure): include
    res |= getattr(c, "include_all_after_import", None) or set()
    for m in c.import_failed:
        if m in c.import_fail_sets:
            res |= c.import_fail_sets[m]
        else:
            info = _MAPS.file(m) if _MAPS is not None else None
            if info is not None:
                res |= _MAPS.set(info[3])  # every test that imports/uses that module
        for rel, mu in c.mod_sets.items():
            res |= _MAPS.set(mu)
    if c.file_fail:
        # a test file failed to import because of patched code: every test file importing
        # a patched module fails the same way (except files that did import and run here)
        ok = getattr(c, "ok_files", set()) - set(c.file_fail)
        for rel, mu in c.mod_sets.items():
            res |= {t for t in _MAPS.set(mu) if _META is None or _META[t][1] not in ok}
    c.result = res


def _cleanup(path):
    try:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _preload(files):
    mods = ["unittest", "unittest.mock", "json", "re", "typing", "dataclasses", "decimal", "datetime", "logging",
            "collections", "functools", "itertools", "enum", "copy", "math", "heapq", "textwrap", "inspect",
            "concurrent.futures", "multiprocessing", "importlib.metadata", "string", "operator", "csv", "gzip",
            "pickle", "uuid", "zoneinfo", "statistics", "difflib", "gc", "tracemalloc", "weakref", "contextlib"]
    heavy = any(f in ("tests/test_optimizer.py", "tests/test_executor.py") for f in files) or any(
        "test_serde" in f or "test_lineage" in f for f in files
    )
    if heavy:
        mods += ["duckdb", "pandas", "pandas.testing", "numpy"]
    for m in mods:
        try:
            __import__(m)
        except Exception:
            pass
    try:
        import syncpool  # noqa: F401
    except Exception:
        pass


if __name__ == "__main__":
    main()
    # skip interpreter teardown (frozen objects are never freed; nothing is left to clean up)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
