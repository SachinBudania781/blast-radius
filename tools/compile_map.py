"""Compile raw per-test traces into per-line candidate tables for the selector.

Output (marshal): {
  "files": {relpath: (line_set_ids: bytes-like list, wide_ids list, readers_id, moduse_id)},
  "sets": [tuple(test numbers), ...],
  "file_readers": {relpath: set_id}   # non-python files read/listed by tests
}
"""

import ast
import glob
import json
import marshal
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))
from astinfo import analyze  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/sqlglot"
MAPDIR = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/brmap2"
TESTS = sys.argv[3] if len(sys.argv) > 3 else "/home/claude/kit/sample/tests.json"
CLOSURES = sys.argv[4] if len(sys.argv) > 4 else "/home/claude/mapper/closures.txt"
OUT = sys.argv[5] if len(sys.argv) > 5 else "/home/claude/blast-radius/maps/map.marshal"

T = json.load(open(TESTS))["tests"]
ID = {x["id"]: x["number"] for x in T}
NT = len(T)

res, classes, imp = {}, {}, {}
for f in glob.glob(os.path.join(MAPDIR, "map-*.pkl")):
    d = pickle.load(open(f, "rb"))
    res.update(d.get("results", {}))
    classes.update(d.get("classes", {}))
    for k, v in d.get("import_lines", {}).items():
        imp.setdefault(k, set()).update(v)

cov = {}
tcls = {}
tfiles = {}
bycls = {}
for tid, r in res.items():
    if tid in ID:
        bycls.setdefault(tid.rsplit("::", 1)[0], []).append(ID[tid])
for tid, r in res.items():
    if tid not in ID:
        continue
    n = ID[tid]
    for fl in r["lines"]:
        cov.setdefault(fl, set()).add(n)
    for fl in r["setup_lines"]:
        cov.setdefault(fl, set()).update(bycls[tid.rsplit("::", 1)[0]])
    tcls[n] = r["classes"]
    tfiles[n] = set(r["files"])
# files read while setUpClass ran (in the setup phase of the class's first test) belong to
# every test of the class; we did not record phases for files, so propagate that test's files.
# Those propagated reads are "indirect": the selector runs such tests instead of assuming they fail.
direct_files = {n: set(fs) for n, fs in tfiles.items()}
for tid, r in res.items():
    if tid in ID and r["setup_lines"]:
        for m in bycls[tid.rsplit("::", 1)[0]]:
            tfiles.setdefault(m, set()).update(r["files"])
            direct_files.setdefault(m, set()).difference_update(r["files"]) if m != ID[tid] else None
        direct_files[ID[tid]] = set()  # the class's first test: cannot tell setup from call reads
missing = set(range(NT)) - set(tcls)
if missing:
    print("WARNING tests without trace:", len(missing), file=sys.stderr)

# test-file collection closures
closure = {}
for line in open(CLOSURES):
    f, _, js = line.strip().partition(" ")
    try:
        closure[f] = set(json.loads(js)) if js else set()
    except Exception:
        closure[f] = None  # unknown -> treat as importing everything
tests_by_file = {}
for x in T:
    tests_by_file.setdefault(x["file"], set()).add(x["number"])

# class usage: class qualname -> tests using an instance of it or a subclass
uses_class = {}
for n, cl in tcls.items():
    for k in cl:
        info = classes.get(k)
        mro = info["mro"] if info else [k]
        for c in mro:
            uses_class.setdefault(c, set()).add(n)

# modules: tests that import (collection closure), use a class from, or run a line in a module
def modname(rel):
    m = rel[:-3].replace("/", ".")
    if m.endswith(".__init__"):
        m = m[: -len(".__init__")]
    return m


lines_by_file = {}
for (f, ln), s in cov.items():
    lines_by_file.setdefault(f, set()).update(s)
cls_by_module = {}
for qn, s in uses_class.items():
    mod = qn.split(":", 1)[0]
    cls_by_module.setdefault(mod, set()).update(s)

ALL = frozenset(range(NT))


def moduse(rel):
    m = modname(rel)
    s = set(lines_by_file.get(rel, ()))
    s |= cls_by_module.get(m, set())
    for f, cl in closure.items():
        if cl is None or m in cl or any(x.startswith(m + ".") for x in cl):
            s |= tests_by_file.get(f, set())
    if rel.startswith("tests/") and rel in tests_by_file:
        s |= tests_by_file[rel]
    return s


# readers: tests that opened a file or listed a directory
readers = {}
for n, fs in tfiles.items():
    for p in fs:
        readers.setdefault(p, set()).add(n)

sets = []
set_index = {}


def sid(s):
    key = tuple(sorted(s))
    i = set_index.get(key)
    if i is None:
        i = len(sets)
        sets.append(key)
        set_index[key] = i
    return i


EMPTY = sid(())

# Static name references across all python files: name -> list of (file, line, scope_kind, class_qualname)
py_files = sorted(
    p[len(SRC) + 1 :]
    for p in glob.glob(SRC + "/sqlglot/**/*.py", recursive=True) + glob.glob(SRC + "/tests/**/*.py", recursive=True)
)
infos = {}
for rel in py_files:
    try:
        infos[rel] = analyze(open(os.path.join(SRC, rel), encoding="utf-8").read())
    except Exception as e:  # noqa: BLE001
        print("skip", rel, e, file=sys.stderr)


def scope_chain(a, ln):
    idx = a["line_scope"][ln] if ln < len(a["line_scope"]) else 0
    out = []
    while idx is not None:
        out.append(a["scopes"][idx])
        idx = a["scopes"][idx].parent
    return out


def stmt_cov(rel, a, lo, hi):
    s = set()
    for x in range(lo, hi + 1):
        s |= cov.get((rel, x), set())
    return s


name_refs = {}  # name -> {owner: set of (rel, line)}
for rel, a in infos.items():
    for ln, names in a["refs"].items():
        for owner, nm in names:
            name_refs.setdefault(nm, {}).setdefault(owner, set()).add((rel, ln))

# class family short names: for class qualname C, names of C and all its subclasses
family_names = {}
for qn, info in classes.items():
    short = qn.split(":", 1)[1].split(".")[-1]
    for base in info["mro"]:
        family_names.setdefault(base, set()).add(short)


def ref_locs(nm, owners=None, module_level=False):
    d = name_refs.get(nm, {})
    out = set()
    for owner, locs in d.items():
        if owners is not None:
            if owner in owners or owner == '"':
                out |= locs
        elif module_level:
            if owner not in ("self", "cls", "klass", "super", "type()"):
                out |= locs
    return out


def class_qual(rel, scope):
    return f"{modname(rel)}:{scope.qual}"


def deftime_tests_for_ref(rel, a, ln, depth, seen_names):
    """Tests affected through a reference at (rel, ln)."""
    chain = scope_chain(a, ln)
    st = a["line_stmt"][ln] if ln < len(a["line_stmt"]) else None
    lo, hi = st if st else (ln, ln)
    s = stmt_cov(rel, a, lo, hi)
    inner = chain[0]
    runtime = any(sc.kind in ("func", "lambda") for sc in chain[:-1]) and ln in a["runtime"]
    if runtime:
        return s
    if inner.kind == "class":
        s |= uses_class.get(class_qual(rel, inner), set())
        return s
    if inner.kind == "module" and depth > 0:
        for nm in a["defined"].get(lo, ()):  # names defined by this module-level statement
            if nm not in seen_names:
                seen_names.add(nm)
                s |= refs_tests(nm, depth - 1, seen_names)
    return s


_refs_cache = {}


def refs_tests(nm, depth, seen_names, owners=None):
    key = (nm, depth, tuple(sorted(owners)) if owners is not None else None)
    if key in _refs_cache:
        return _refs_cache[key]
    s = set()
    for rel, ln in ref_locs(nm, owners=owners, module_level=owners is None):
        a = infos.get(rel)
        if a is None:
            continue
        s |= deftime_tests_for_ref(rel, a, ln, depth, seen_names)
        if len(s) >= NT:
            break
    _refs_cache[key] = s
    return s


import_exec = set(imp.keys())
for f in glob.glob(os.path.join(MAPDIR, "map-*.pkl")):
    d = pickle.load(open(f, "rb"))
    import_exec |= set(d.get("collection", {}).get("lines", set()))

_attr_cache = {}


def attr_runtime_readers(nm, fam_quals=()):
    """Tests that run a line reading class attribute nm at runtime, or None when nm is also
    read at import time, read via getattr with a string, or used by bare name inside the
    class body of the same class family (a derived class attribute)."""
    key = (nm, tuple(sorted(fam_quals)))
    if key in _attr_cache:
        return _attr_cache[key]
    d = name_refs.get(nm, {})
    out = set()
    ok = True
    for owner, locs in d.items():
        for rel, ln in locs:
            a = infos.get(rel)
            if a is None:
                continue
            st = a["line_stmt"][ln] if ln < len(a["line_stmt"]) else None
            lo, hi = st if st else (ln, ln)
            if owner == '"':
                ok = False
                break
            if owner is None:
                # bare names inside methods/modules are locals/globals, not this attribute;
                # inside a class body of the same family they derive another attribute
                chain = scope_chain(a, ln)
                if chain and chain[0].kind == "class" and class_qual(rel, chain[0]) in fam_quals:
                    ok = False
                    break
                continue
            chain = scope_chain(a, ln)
            runtime = ln in a["runtime"] and any(sc.kind in ("func", "lambda") for sc in chain[:-1])
            if not runtime or any((rel, x) in import_exec for x in range(lo, hi + 1)):
                ok = False
                break
            out |= stmt_cov(rel, a, lo, hi)
        if not ok:
            break
    # dynamic getattr sites in sqlglot only fetch *_sql methods, classes by name, or
    # instance fields (checked by hand at the pinned commit), never class data attributes
    _attr_cache[key] = out if ok else None
    return _attr_cache[key]


files_out = {}
for rel, a in infos.items():
    nl = a["nlines"]
    narrow = [EMPTY] * (nl + 1)
    wide = [EMPTY] * (nl + 1)
    rd = readers.get(rel, set())
    mu = moduse(rel)
    # lambda body spans: (lambda_line, body_lo, body_hi)
    lam = []
    for sc in a["scopes"]:
        if sc.kind == "lambda":
            lam.append((sc.start, sc.body_start, sc.end))
    func_cache = {}
    for ln in range(1, nl + 1):
        st = a["line_stmt"][ln] if ln < len(a["line_stmt"]) else None
        if st is None:
            # blank / comment / docstring continuation lines
            n_s = set(rd)
            narrow[ln] = sid(n_s)
            wide[ln] = narrow[ln]
            continue
        lo, hi = st
        chain = scope_chain(a, ln)
        inner = chain[0]
        s = set(rd)
        w = set()
        if lo in a["docstring_lines"] and all(x in a["docstring_lines"] for x in range(lo, hi + 1)):
            # docstrings have no runtime effect here (doctests are not collected by pytest)
            narrow[ln] = sid(s)
            wide[ln] = narrow[ln]
            continue
        # lambda body?
        lam_hit = None
        for (ll, blo, bhi) in lam:
            if blo <= ln <= bhi and ln > ll:
                if lam_hit is None or (bhi - blo) < (lam_hit[2] - lam_hit[1]):
                    lam_hit = (ll, blo, bhi)
        in_func = any(sc.kind in ("func",) for sc in chain[:-1]) and ln in a["runtime"]
        imp_mods = set()
        if any(sc.kind == "func" for sc in chain[:-1]):
            for x in range(lo, hi + 1):
                imp_mods |= {m for m in imp.get((rel, x), ())}
        if lam_hit and not in_func:
            s |= stmt_cov(rel, a, lam_hit[1], lam_hit[2])
            w |= stmt_cov(rel, a, lam_hit[0], lam_hit[2])
        elif in_func:
            s |= stmt_cov(rel, a, lo, hi)
            # function-level (innermost def)
            fsc = next(sc for sc in chain if sc.kind == "func")
            key = (fsc.start, fsc.end)
            if key not in func_cache:
                func_cache[key] = stmt_cov(rel, a, fsc.start, fsc.end)
            w |= func_cache[key]
        else:
            # def-time statement (module/class level, or a def/class header)
            s |= stmt_cov(rel, a, lo, hi)
            header_of = None
            for sc in a["scopes"][1:]:
                if sc.start == lo and sc.kind in ("func", "class"):
                    header_of = sc
                    break
            if header_of is not None and header_of.kind == "func":
                s |= stmt_cov(rel, a, header_of.start, header_of.end)
                # functions called at import time: their callers' modules
                for x in range(header_of.start, header_of.end + 1):
                    imp_mods |= set(imp.get((rel, x), ()))
                par = a["scopes"][header_of.parent]
                if par.kind == "class":
                    # renaming/decorating a method changes the class
                    cq = class_qual(rel, par)
                    s |= uses_class.get(cq, set()) & stmt_cov(rel, a, header_of.start, header_of.end) if False else set()
                    s |= refs_tests(header_of.name, 0, set(), owners=family_names.get(cq, {par.name}))
                else:
                    s |= refs_tests(header_of.name, 1, set())
            elif header_of is not None and header_of.kind == "class":
                s |= uses_class.get(class_qual(rel, header_of), set())
                s |= uses_class.get(class_qual(rel, header_of).rsplit(".", 1)[0] if "." in header_of.qual else "", set()) if False else set()
                s |= stmt_cov(rel, a, header_of.start, header_of.end)
                s |= refs_tests(header_of.name, 1, set())
            elif inner.kind == "class":
                cq = class_qual(rel, inner)
                uc = uses_class.get(cq, set())
                fam = family_names.get(cq, {inner.name})
                names = a["defined"].get(lo, set())
                is_def = any(sc.start == lo and sc.kind in ("func", "class") for sc in a["scopes"][1:])
                narrowed = None
                fam_quals = {q for q, info in classes.items() if cq in info["mro"]} | {cq}
                if names and not is_def:
                    narrowed = set()
                    for nm in names:
                        r = attr_runtime_readers(nm, fam_quals)
                        if r is None:
                            narrowed = None
                            break
                        narrowed |= r
                if narrowed is None:
                    s |= uc
                else:
                    s |= uc & narrowed
                for nm in names:
                    # explicit Class.NAME references (self.NAME is covered by class usage)
                    s |= refs_tests(nm, 0, set(), owners=fam)
            else:
                names = a["defined"].get(lo, set())
                if names:
                    for nm in names:
                        s |= refs_tests(nm, 1, set())
                else:
                    s |= mu
            w |= s
        for m in imp_mods:
            mr = m if m.endswith(".py") else None
            if mr:
                s |= moduse(mr)
        if len(s) > NT:
            s = set(ALL)
        narrow[ln] = sid(s)
        wide[ln] = sid(s | w)
    # top-level-ish function spans for local syntax checks: (start incl. decorators, end, indent)
    fspans = []
    src_lines = open(os.path.join(SRC, rel), encoding="utf-8").read().split("\n")
    for sc in a["scopes"][1:]:
        if sc.kind == "func":
            par = a["scopes"][sc.parent]
            if par.kind in ("module", "class"):
                first = src_lines[sc.start - 1]
                indent = len(first) - len(first.lstrip())
                fspans.append((sc.start, sc.end, indent))
    files_out[rel] = (
        narrow,
        wide,
        sid(rd),
        sid(mu),
        tuple(sorted(a["string_spans"])),
        tuple(sorted(a["backslash_lines"])),
        tuple(sorted(fspans)),
    )

# non-python files and directories read by tests
file_readers = {p: sid(s) for p, s in readers.items() if not p.endswith(".py")}
direct = {}
for n, fs in direct_files.items():
    for p in fs:
        direct.setdefault(p, set()).add(n)
file_direct = {p: sid(s) for p, s in direct.items() if not p.endswith(".py")}
dir_listers = file_readers  # same table: directories appear as paths too

os.makedirs(os.path.dirname(OUT), exist_ok=True)
blob = {
    "files": {k: (bytes(0), v[0], v[1], v[2], v[3]) for k, v in files_out.items()},
    "sets": sets,
    "file_readers": file_readers,
    "ntests": NT,
}
# split: one small file per python file + sets file, so the selector loads only what it needs
outdir = os.path.dirname(OUT)
for old in glob.glob(os.path.join(outdir, "f_*.m")):
    os.remove(old)
import array


def _pack(ids):
    return array.array("I", ids).tobytes()


index = {}
for i, (k, v) in enumerate(sorted(files_out.items())):
    fn = f"f_{i:03d}.m"
    index[k] = fn
    with open(os.path.join(outdir, fn), "wb") as fh:
        marshal.dump((bytes(_pack(v[0])), bytes(_pack(v[1])), v[2], v[3], v[4], v[5], v[6]), fh)
# sets as fixed-width bitsets (one bytes blob), decoded lazily by the selector
nb = (NT + 7) // 8
blobs = bytearray()
for st in sets:
    b = bytearray(nb)
    for x in st:
        b[x >> 3] |= 1 << (x & 7)
    blobs += b
with open(os.path.join(outdir, "sets.bin"), "wb") as fh:
    fh.write(bytes(blobs))
with open(os.path.join(outdir, "index.m"), "wb") as fh:
    marshal.dump({"files": index, "file_readers": file_readers, "file_direct": file_direct, "ntests": NT}, fh)
print("files", len(files_out), "sets", len(sets), file=sys.stderr)
sz = sum(os.path.getsize(p) for p in glob.glob(os.path.join(outdir, "*.m")))
print("bytes", sz, file=sys.stderr)
