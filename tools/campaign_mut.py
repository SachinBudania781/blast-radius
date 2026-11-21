import difflib, io, os, random, re, tokenize
SRC = os.environ.get("BR_SRC", "/home/claude/sqlglot")
REL = [("==", "!="), ("!=", "=="), ("<", "<="), ("<=", "<"), (">", ">="), (">=", ">"), ("is", "=="),
       ("and", "or"), ("or", "and"), ("True", "False"), ("False", "True"), ("+", "-"), ("-", "+"),
       ("in", "not in")]


def mutate_line(line):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(line.strip() + "\n").readline))
    except Exception:
        return None
    opts = []
    for tk in toks:
        if tk.type in (tokenize.OP, tokenize.NAME):
            for a, b in REL:
                if tk.string == a:
                    opts.append((tk, b))
        if tk.type == tokenize.NUMBER and re.fullmatch(r"\d+", tk.string):
            opts.append((tk, str(int(tk.string) + 1)))
        if tk.type == tokenize.NAME and tk.string == "not":
            opts.append((tk, ""))
    if not opts:
        return None
    tk, rep = random.choice(opts)
    indent = len(line) - len(line.lstrip())
    s = line.strip()
    c0, c1 = tk.start[1], tk.end[1]
    if tk.start[0] != 1:
        return None
    new = s[:c0] + rep + s[c1:]
    if rep == "":
        new = (s[:c0] + s[c1:]).replace("  ", " ")
    return line[:indent] + new


def make_patch(rel, ln):
    src = open(os.path.join(SRC, rel), encoding="utf-8").read().split("\n")
    old = src[ln - 1]
    new = mutate_line(old)
    if new is None or new == old:
        return None
    mod = list(src)
    mod[ln - 1] = new
    try:
        compile("\n".join(mod), rel, "exec")
    except Exception:
        return None
    a = [x + "\n" for x in src[:-1]] + ([src[-1]] if src[-1] else [])
    b = [x + "\n" for x in mod[:-1]] + ([mod[-1]] if mod[-1] else [])
    diff = "".join(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", n=3))
    return f"diff --git a/{rel} b/{rel}\n" + diff, old.strip(), new.strip()


