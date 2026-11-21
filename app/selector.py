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
    global _MAPS
    _MAPS = maps
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
    jobs = []
    for c in commits:
        if c.unknown:
            c.result = set(all_tests)
            continue
        c.sure = {t for t in c.sure if ms[t] < 3000}
        cand = c.cands - c.always - c.sure
        cheap = {t for t in cand if ms[t] < EXPENSIVE * 1000}
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
        if cheap and policy.CPU_RATIO * sum(run_ms.get(t, ms[t]) for t in cheap) > policy.pack_cost(cheap | exp_ | c.sure | c.always, ms):
            # running them would cost more than just including them
            if not c.import_check:
                c.result = cheap | exp_ | c.sure | c.always
                continue
            # but first make sure the patched modules import (else far more tests fail)
            c.include_all_after_import = cheap | exp_
            c.cheap, c.exp = set(), set()
        if not cheap and not exp_ and not c.import_check:
            c.result = set(c.always) | c.sure
            continue
        jobs.append(c)

    ids = [t["id"] for t in tests]
    for c in jobs:
        c.early_blind = policy.early_stop_ok(c.cheap, c.exp, ms, ids, getattr(c, "run_ms", None), c.dedicated)
    try:
        run_jobs(jobs, meta, ms, maps, ids)
    except BaseException as e:  # noqa: BLE001
        log("run_jobs failed:", repr(e))
        for c in jobs:
            if c.result is None:
                c.result = set(c.cands) | set(c.sure) | set(c.always) | set(getattr(c, "blind", ()))

    out = sys.stdout
    for c in commits:
        res = c.result if c.result is not None else set(c.cands) | c.always | c.sure
        res = {t for t in res if isinstance(t, int) and 0 <= t < ntests}
        try:
            workers = pack(sorted(res), ms)
        except Exception:  # noqa: BLE001
            workers = [sorted(res)] if res else []
        out.write(json.dumps({"commit": c.cid, "workers": workers}, separators=(",", ":")) + "\n")
    out.flush()
    if DEBUG:
        for c in commits:
            log(c.cid, "cand", len(c.cands), "res", len(c.result or ()), c.notes)
    log(f"done {len(commits)} commits in {time.time() - t_start:.1f}s")


_MAPS = None


def run_jobs(jobs, meta, ms, maps, ids=None):
    global _MAPS
    _MAPS = maps
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
            runner.precompute_executor_rows(SRC)
        except Exception as e:  # noqa: BLE001
            log("precompute failed", repr(e))

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
                import zlib as _z

                # deterministic shuffle of test classes: early results then represent the whole set
                tl = [meta[t] for t in sorted(tests, key=lambda t: (_z.crc32(f"{meta[t][1]}::{meta[t][2]}".encode()), meta[t][1], meta[t][2]))]
                tm = float(os.environ.get("BR_TIMEOUT_MULT", "5"))
                timeouts = {t: min(float(os.environ.get("BR_TIMEOUT_CAP", "45")), max(5.0, ms[t] / 1000 * tm + 3)) for t in tests}
                runner.TPC_FILTER = getattr(c, "tpc_filter", None)
                runner.OPT_FILTER = getattr(c, "opt_filter", None)
                runner.child_main(ov, tl, w, timeouts, 0, sorted(m for m in c.import_check if m.startswith("sqlglot/") or m.startswith("tests/")), set(c.patched) | set(c.deleted))
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
        if not c.cheap and not c.import_check:
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
                    if c.phase == 1 and getattr(c, "early_blind", None) and (b" 1\n" in buf.replace(b"-3 ready\n", b"") or b"-1 " in buf or b"-2 " in buf):
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
    if c.file_fail:
        # "a test fails when its file cannot be imported": every test of that file fails
        for t in range(len(meta)):
            if meta[t][1] in c.file_fail:
                c.failed.add(t)


def _next_phase(c, ms, ids):
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
        rest = set(c.exp) - c.blind
        blind, verify = policy.decide(rest, base | c.blind, "cat", n_cheap, ms, ids, getattr(c, "run_ms", None), c.dedicated)
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
    blind, verify = policy.decide(set(c.exp) - fx, base, regime, n_cheap, ms, ids, getattr(c, "run_ms", None), c.dedicated)
    if fx:
        # what cheap tests did says nothing about a test reading an edited data file
        b2, v2 = policy.decide(fx, base | blind, "unknown", n_cheap, ms, ids, getattr(c, "run_ms", None), c.dedicated)
        blind |= b2
        verify |= v2
    c.blind = blind
    return verify or None


def _finish(c, ms):
    res = set(c.failed) | c.always | c.sure | getattr(c, "blind", set())
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
        # a patched module fails the same way
        for rel, mu in c.mod_sets.items():
            res |= _MAPS.set(mu)
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
