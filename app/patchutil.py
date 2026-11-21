"""Parse and apply unified diffs (git format) against the pinned sqlglot tree."""

import re

_HUNK = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FilePatch:
    __slots__ = ("old_path", "new_path", "hunks", "is_new", "is_deleted", "binary", "rename")

    def __init__(self):
        self.old_path = None
        self.new_path = None
        self.hunks = []  # list of (old_start, old_len, new_start, new_len, lines[bytes])
        self.is_new = False
        self.is_deleted = False
        self.binary = False
        self.rename = False

    @property
    def path(self):
        return self.new_path if self.new_path is not None else self.old_path


def _strip_prefix(p):
    p = p.strip()
    if p.startswith(b'"') and p.endswith(b'"'):
        p = p[1:-1]
    if p == b"/dev/null":
        return None
    if p[:2] in (b"a/", b"b/"):
        p = p[2:]
    return p.decode("utf-8", "surrogateescape")


def parse(data: bytes):
    files = []
    cur = None
    lines = data.split(b"\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith(b"diff --git "):
            cur = FilePatch()
            files.append(cur)
            m = re.match(rb"diff --git (\S+) (\S+)$", line)
            if m:
                cur.old_path = _strip_prefix(m.group(1))
                cur.new_path = _strip_prefix(m.group(2))
            i += 1
            continue
        if cur is None:
            i += 1
            continue
        if line.startswith(b"new file mode"):
            cur.is_new = True
        elif line.startswith(b"deleted file mode"):
            cur.is_deleted = True
        elif line.startswith(b"rename from ") or line.startswith(b"rename to ") or line.startswith(b"copy "):
            cur.rename = True
            if line.startswith(b"rename from "):
                cur.old_path = line[12:].decode("utf-8", "surrogateescape")
            elif line.startswith(b"rename to "):
                cur.new_path = line[10:].decode("utf-8", "surrogateescape")
        elif line.startswith(b"Binary files") or line.startswith(b"GIT binary patch"):
            cur.binary = True
        elif line.startswith(b"--- "):
            p = _strip_prefix(line[4:].split(b"\t")[0])
            if p is None:
                cur.is_new = True
            else:
                cur.old_path = p
        elif line.startswith(b"+++ "):
            p = _strip_prefix(line[4:].split(b"\t")[0])
            if p is None:
                cur.is_deleted = True
            else:
                cur.new_path = p
        elif line.startswith(b"@@"):
            m = _HUNK.match(line)
            if m:
                os_, ol, ns, nl = m.groups()
                os_ = int(os_)
                ol = 1 if ol is None else int(ol)
                ns = int(ns)
                nl = 1 if nl is None else int(nl)
                body = []
                i += 1
                o_seen = n_seen = 0
                while i < n and (o_seen < ol or n_seen < nl):
                    b = lines[i]
                    if b.startswith(b"\\"):
                        body.append(b)
                        i += 1
                        continue
                    tag = b[:1]
                    if tag == b" " or b == b"":
                        o_seen += 1
                        n_seen += 1
                        body.append(b" " + b[1:] if b else b" ")
                    elif tag == b"-":
                        o_seen += 1
                        body.append(b)
                    elif tag == b"+":
                        n_seen += 1
                        body.append(b)
                    else:
                        break
                    i += 1
                while i < n and lines[i].startswith(b"\\"):
                    body.append(lines[i])
                    i += 1
                cur.hunks.append((os_, ol, ns, nl, body))
                continue
        i += 1
    if cur is not None and cur.old_path is None and cur.new_path is None:
        pass
    if cur is not None and cur.is_deleted:
        cur.new_path = None
    return files


def apply(original: bytes, fp: FilePatch):
    """Return patched bytes, or None if the hunks do not match."""
    if fp.is_new:
        original = b""
    src = original.split(b"\n")
    # keep track of a trailing newline: split leaves '' at the end
    had_trailing_nl = original.endswith(b"\n")
    if had_trailing_nl:
        src = src[:-1]
    out = []
    pos = 0  # index into src (0-based)
    new_no_nl = False
    for os_, ol, ns, nl, body in fp.hunks:
        start = os_ - 1 if ol > 0 else os_
        if start < pos or start > len(src):
            return None
        out.extend(src[pos:start])
        pos = start
        prev_tag = None
        for b in body:
            if b.startswith(b"\\"):
                if prev_tag in (b"+", b" "):
                    new_no_nl = True
                continue
            tag, text = b[:1], b[1:]
            prev_tag = tag
            if tag == b" ":
                if pos >= len(src) or src[pos] != text:
                    if not (pos < len(src) and src[pos].rstrip(b"\r") == text.rstrip(b"\r")):
                        return None
                out.append(src[pos])
                pos += 1
            elif tag == b"-":
                if pos >= len(src) or src[pos] != text:
                    if not (pos < len(src) and src[pos].rstrip(b"\r") == text.rstrip(b"\r")):
                        return None
                pos += 1
            elif tag == b"+":
                out.append(text)
        if prev_tag == b"+" or prev_tag == b" ":
            pass
    last_touched_end = pos >= len(src)
    out.extend(src[pos:])
    if fp.is_deleted:
        return None
    data = b"\n".join(out)
    if out:
        if new_no_nl:
            pass
        elif last_touched_end or had_trailing_nl:
            data += b"\n"
    return data


def changes(fp: FilePatch):
    """Per hunk: list of (old_line_numbers_removed, inserted_before_old_line, added_texts, removed_texts)."""
    out = []
    for os_, ol, ns, nl, body in fp.hunks:
        o = os_ if ol > 0 else os_ + 1
        removed = []
        removed_txt = []
        added = []  # (insert position as old line number, text)
        for b in body:
            if b.startswith(b"\\"):
                continue
            tag, text = b[:1], b[1:]
            if tag == b" ":
                o += 1
            elif tag == b"-":
                removed.append(o)
                removed_txt.append(text)
                o += 1
            elif tag == b"+":
                added.append((o, text))
        out.append((removed, removed_txt, added))
    return out
