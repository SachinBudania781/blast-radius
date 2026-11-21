"""Static facts about one Python source file, keyed by line number."""

import ast
import io
import tokenize


class Scope:
    __slots__ = ("kind", "name", "qual", "start", "end", "body_start", "parent")

    def __init__(self, kind, name, qual, start, end, body_start, parent):
        self.kind = kind  # "func" | "class" | "lambda" | "module"
        self.name = name
        self.qual = qual
        self.start = start
        self.end = end
        self.body_start = body_start
        self.parent = parent


def analyze(source: str):
    tree = ast.parse(source)
    nlines = source.count("\n") + 2
    scopes = []  # list of Scope
    # per line: innermost scope index, innermost simple-statement span, runtime flag
    line_scope = [0] * (nlines + 1)
    line_stmt = [None] * (nlines + 1)
    module = Scope("module", "<module>", "", 1, nlines, 1, None)
    scopes.append(module)
    defined = {}  # line -> set of names bound at def-time on that statement (module/class level)
    refs_names = {}  # line -> set of identifiers referenced (Name ids and Attribute attrs)
    docstring_lines = set()

    def mark_scope(lo, hi, idx):
        for ln in range(lo, min(hi, nlines) + 1):
            line_scope[ln] = idx

    def header_end(node):
        # last line of a compound statement's header (before its body)
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            lo = first.lineno
            decos = getattr(first, "decorator_list", None)
            if decos:
                lo = min([lo] + [d.lineno for d in decos])
            return max(node.lineno, lo - 1)
        return node.end_lineno

    def visit_body(stmts, scope_idx):
        for i, st in enumerate(stmts):
            visit_stmt(st, scope_idx, first=(i == 0))

    def visit_stmt(st, scope_idx, first=False):
        lo = st.lineno
        decos = getattr(st, "decorator_list", None) or []
        if decos:
            lo = min([lo] + [d.lineno for d in decos])
        if first and isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
            for ln in range(st.lineno, st.end_lineno + 1):
                docstring_lines.add(ln)
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # header: decorators + signature
            hend = header_end(st)
            for ln in range(lo, hend + 1):
                line_stmt[ln] = (lo, hend)
                line_scope[ln] = scope_idx
            kind = "class" if isinstance(st, ast.ClassDef) else "func"
            parent = scopes[scope_idx]
            qual = f"{parent.qual}.{st.name}" if parent.qual else st.name
            if parent.kind == "func":
                qual = f"{parent.qual}.<locals>.{st.name}"
            idx = len(scopes)
            scopes.append(Scope(kind, st.name, qual, lo, st.end_lineno, hend + 1, scope_idx))
            mark_scope(hend + 1, st.end_lineno, idx)
            if scopes[scope_idx].kind in ("module", "class"):
                defined.setdefault(lo, set()).add(st.name)
            for d in decos:
                collect_refs(d)
            for n in ast.walk(st.args) if kind == "func" else []:
                pass
            if kind == "func":
                for dflt in list(st.args.defaults) + [d for d in st.args.kw_defaults if d is not None]:
                    collect_refs(dflt)
                    collect_lambdas(dflt, scope_idx)
            else:
                for b in st.bases + [k.value for k in st.keywords]:
                    collect_refs(b)
            visit_body(st.body, idx)
            return
        compound = isinstance(
            st,
            (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
            + ((ast.TryStar,) if hasattr(ast, "TryStar") else ()),
        )
        if compound:
            hend = header_end(st)
            for ln in range(st.lineno, hend + 1):
                line_stmt[ln] = (st.lineno, hend)
            collect_header_refs(st)
            for field in ("body", "orelse", "finalbody"):
                b = getattr(st, field, None)
                if b:
                    visit_body(b, scope_idx)
            for h in getattr(st, "handlers", []) or []:
                for ln in range(h.lineno, header_end(h) + 1):
                    line_stmt[ln] = (h.lineno, header_end(h))
                if h.type is not None:
                    collect_refs(h.type)
                visit_body(h.body, scope_idx)
            for c in getattr(st, "cases", []) or []:
                visit_body(c.body, scope_idx)
            # else-lines (the 'else:' keyword line) keep the compound header span
            for ln in range(st.lineno, st.end_lineno + 1):
                if line_stmt[ln] is None:
                    line_stmt[ln] = (st.lineno, hend)
            return
        # simple statement
        for ln in range(st.lineno, st.end_lineno + 1):
            line_stmt[ln] = (st.lineno, st.end_lineno)
        collect_refs(st)
        if scopes[scope_idx].kind in ("module", "class"):
            names = set()
            for node in ast.walk(st):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for a in node.names:
                        names.add((a.asname or a.name).split(".")[0])
            if names:
                defined.setdefault(st.lineno, set()).update(names)
        collect_lambdas(st, scope_idx)

    def collect_header_refs(st):
        for field in ("test", "iter", "target", "items", "subject"):
            v = getattr(st, field, None)
            if v is None:
                continue
            for x in v if isinstance(v, list) else [v]:
                collect_refs(x)
                collect_lambdas(x, None)

    def owner_of(v):
        if isinstance(v, ast.Name):
            return v.id
        if isinstance(v, ast.Attribute):
            return v.attr
        if isinstance(v, ast.Call):
            f = v.func
            if isinstance(f, ast.Name) and f.id in ("super", "type"):
                return "super" if f.id == "super" else "type()"
            return "()"
        return "?"

    def collect_refs(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                if not isinstance(n.ctx, ast.Load):
                    continue
                refs_names.setdefault(n.lineno, set()).add((None, n.id))
            elif isinstance(n, ast.Attribute):
                if not isinstance(n.ctx, ast.Load):
                    continue
                refs_names.setdefault(n.end_lineno, set()).add((owner_of(n.value), n.attr))
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
                refs_names.setdefault(n.lineno, set()).add(('"', n.value))

    def collect_lambdas(node, scope_idx):
        # lambdas / comprehensions inside def-time statements run later (lambdas) -> runtime scope
        for n in ast.walk(node):
            if isinstance(n, ast.Lambda):
                parent_idx = scope_idx if scope_idx is not None else 0
                idx = len(scopes)
                p = scopes[parent_idx]
                scopes.append(Scope("lambda", "<lambda>", f"{p.qual}.<lambda>", n.lineno, n.end_lineno, n.body.lineno, parent_idx))
                # lambda body lines: only lines fully inside the lambda body
                for ln in range(n.body.lineno, n.body.end_lineno + 1):
                    if ln != n.lineno or n.body.lineno != n.lineno or True:
                        pass
                LAMBDAS.append((n.body.lineno, n.body.end_lineno, idx))

    LAMBDAS = []
    visit_body(tree.body, 0)

    # runtime lines: inside a function scope body (or lambda body spanning its own lines)
    def is_runtime_scope(idx):
        while idx is not None:
            s = scopes[idx]
            if s.kind in ("func", "lambda"):
                return True
            if s.kind == "module":
                return False
            idx = s.parent
        return False

    runtime = set()
    for ln in range(1, nlines + 1):
        if is_runtime_scope(line_scope[ln]) and line_stmt[ln] is not None:
            # function header lines belong to the parent scope (handled since line_scope set to parent)
            runtime.add(ln)
    lambda_lines = {}
    for lo, hi, idx in LAMBDAS:
        for ln in range(lo, hi + 1):
            lambda_lines.setdefault(ln, idx)

    # token facts
    string_lines = set()  # lines inside a multi-line string token (not first line)
    backslash_lines = set()  # physical lines ending with a backslash continuation
    comment_only_lines = set()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except Exception:
        toks = []
    string_spans = []
    for tok in toks:
        if tok.type in (tokenize.STRING,) or (hasattr(tokenize, "FSTRING_START") and tok.type == tokenize.FSTRING_START):
            pass
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            string_spans.append((tok.start[0], tok.end[0]))
    if hasattr(tokenize, "FSTRING_START"):
        stack = []
        for tok in toks:
            if tok.type == tokenize.FSTRING_START:
                stack.append(tok.start[0])
            elif tok.type == tokenize.FSTRING_END and stack:
                a0 = stack.pop()
                if tok.end[0] > a0:
                    string_spans.append((a0, tok.end[0]))
    for tok in toks:
        if tok.type == tokenize.STRING or (hasattr(tokenize, "FSTRING_START") and tok.type in (tokenize.FSTRING_MIDDLE,)):
            if tok.end[0] > tok.start[0]:
                for ln in range(tok.start[0], tok.end[0] + 1):
                    string_lines.add(ln)
        elif tok.type == tokenize.COMMENT:
            if source.splitlines()[tok.start[0] - 1].strip().startswith("#"):
                comment_only_lines.add(tok.start[0])
    for i, line in enumerate(source.splitlines(), 1):
        if line.rstrip("\n").endswith("\\") and i not in string_lines:
            backslash_lines.add(i)

    return {
        "scopes": scopes,
        "line_scope": line_scope,
        "line_stmt": line_stmt,
        "runtime": runtime,
        "lambda_lines": lambda_lines,
        "defined": defined,
        "refs": refs_names,
        "docstring_lines": docstring_lines,
        "string_lines": string_lines,
        "backslash_lines": backslash_lines,
        "comment_only_lines": comment_only_lines,
        "string_spans": string_spans,
        "nlines": nlines,
    }
