"""Render a Vietnamese thesis-style .docx from one plain content file.

Usage:
    ~/.venvs/thesis-docs/bin/python render_doc.py CONTENT.md OUTPUT.docx [--data DATA.json] [--pdf]

Only the content file changes between documents. Its syntax (full example in
templates/mau_tai_lieu.md):

    ---                         front matter, optional
    title: TIÊU ĐỀ TÀI LIỆU
    subtitle: Dòng phụ đề, có thể lặp lại nhiều dòng subtitle
    data: so_lieu.json          optional, same as --data
    chapter_new_page: true      each chapter after the first starts a new page
    ---
    # Tên chương                CHƯƠNG n. TÊN CHƯƠNG
    ## Tên mục                  n.m. Tên mục
    ### Tên tiểu mục            n.m.k. Tên tiểu mục
    ::heading Lời mở đầu        centred, unnumbered heading
    Đoạn văn. Các dòng liền nhau được nối thành một đoạn; dòng trống ngắt đoạn.
    ::table Chú thích bảng      caption goes ABOVE the table
    ::widths 5, 3, 2,5          optional, cm per column; separate with ", " or ";" (2,5 is a decimal)
    ::font 10                   optional, point size of the body cells
    | Cột 1 | Cột 2 | Cột 3 |   first row is the header
    |---|---|---|               separator rows are ignored
    | A | **0,812** | 0,790 |   **x** renders the cell bold
    ::note Ghi chú dưới bảng    directly after a table: the table note; elsewhere: an italic note
    ::pagebreak
    ::refs                      every following non-empty line is one reference, until the next directive
    Tác giả, "Tên bài," Hội nghị, năm.
    ::check D["a"]["b"] < 0.5 | thông báo khi sai
    <!-- comment -->            ignored

Values from the data file are inserted with {{path|format}}, e.g.
{{stage1.macro_auroc|f4}} -> 0,7643. Formats: f1..f6 (decimal comma), sf1..sf6
(signed), pct / pct0..pct3 (percent), n (thousands dot), raw. A path is dotted
keys and list indices; anything else is evaluated as a Python expression over D.

Every ::check must hold or nothing is written. The rendered text is also scanned
for symbols the thesis style forbids (bullets, arrows, dashes, emoji, markdown
leftovers); any hit aborts with the offending line.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fmt import Thesis, f, sf, pct, n  # noqa: E402

FORBIDDEN_ASCII = set("*#`")


def _fmt_value(value, spec):
    if spec in ("", "raw"):
        return str(value)
    m = re.fullmatch(r"(sf|f|pct)(\d?)", spec)
    if m:
        kind, digits = m.group(1), m.group(2)
        if kind == "f":
            return f(value, int(digits or 4))
        if kind == "sf":
            return sf(value, int(digits or 4))
        return pct(value, int(digits or 1))
    if spec == "n":
        return n(value)
    raise ValueError(f"unknown format '{spec}'")


def _resolve(expr, data):
    if re.fullmatch(r"[\w\-]+(\.[\w\-]+)*", expr):
        cur = data
        for key in expr.split("."):
            if isinstance(cur, list):
                cur = cur[int(key)]
            elif isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                raise KeyError(f"'{expr}': key '{key}' not found in data")
        return cur
    names = {k: _Attr.wrap(v) for k, v in data.items() if isinstance(k, str) and k.isidentifier()}
    return eval(expr, {**names, "D": data, "f": f, "sf": sf, "pct": pct, "n": n, "abs": abs, "min": min, "max": max,
                       "sum": sum, "len": len, "round": round})


class _Attr(dict):
    """Dict whose keys also read as attributes, so {{a.b / c.d}} works in expressions."""
    @classmethod
    def wrap(cls, v):
        if isinstance(v, dict):
            return cls({k: cls.wrap(x) for k, x in v.items()})
        if isinstance(v, list):
            return [cls.wrap(x) for x in v]
        return v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def substitute(text, data, lineno):
    def repl(m):
        expr, _, spec = m.group(1).partition("|")
        try:
            return _fmt_value(_resolve(expr.strip(), data), spec.strip())
        except Exception as exc:  # noqa: BLE001 -- reported with the line number
            raise SystemExit(f"line {lineno}: cannot fill {{{{{m.group(1)}}}}}: {exc}")
    return re.sub(r"\{\{(.+?)\}\}", repl, text)


def parse_front_matter(lines):
    meta = {"subtitle": []}
    if not lines or lines[0].strip() != "---":
        return meta, 0
    for i in range(1, len(lines)):
        s = lines[i].strip()
        if s == "---":
            return meta, i + 1
        if ":" in s:
            k, v = (x.strip() for x in s.split(":", 1))
            if k == "subtitle":
                meta["subtitle"].append(v)
            else:
                meta[k] = v
    raise SystemExit("front matter opened with --- but never closed")


def split_row(line):
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def cell(value):
    m = re.fullmatch(r"\*\*(.+)\*\*", value)
    return (m.group(1), True) if m else value


def check_symbols(text, where):
    bad = sorted({ch for ch in text if ord(ch) >= 0x2000 or ch in FORBIDDEN_ASCII})
    if bad:
        shown = ", ".join(f"'{ch}' U+{ord(ch):04X}" for ch in bad)
        raise SystemExit(f"{where}: forbidden symbol(s) {shown} in: {text[:120]}")


def render(content_path, output_path, data_path=None):
    raw = Path(content_path).read_text(encoding="utf-8")
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    lines = raw.split("\n")
    meta, start = parse_front_matter(lines)
    data_file = data_path or meta.get("data")
    data = {}
    if data_file:
        p = Path(data_file)
        if not p.is_absolute():
            p = Path(content_path).resolve().parent / p
        data = json.loads(p.read_text(encoding="utf-8"))

    T = Thesis()
    chapter_new_page = meta.get("chapter_new_page", "true").lower() != "false"
    if meta.get("title"):
        T.title(substitute(meta["title"], data, 1))
    for sub in meta["subtitle"]:
        T.para(substitute(sub, data, 1), indent=False).alignment = 1

    failures = []
    para_buf, para_line = [], 0
    i = start

    def flush():
        nonlocal para_buf
        if para_buf:
            text = substitute(" ".join(para_buf), data, para_line)
            check_symbols(text, f"line {para_line}")
            T.para(text)
            para_buf = []

    def clean(text, lineno):
        text = substitute(text, data, lineno)
        check_symbols(text, f"line {lineno}")
        return text

    while i < len(lines):
        line, lineno = lines[i], i + 1
        s = line.strip()
        if not s:
            flush(); i += 1; continue
        if s.startswith("### "):
            flush(); T.sub_heading(clean(s[4:], lineno)); i += 1; continue
        if s.startswith("## "):
            flush(); T.section_heading(clean(s[3:], lineno)); i += 1; continue
        if s.startswith("# "):
            flush(); T.chapter_heading(clean(s[2:], lineno), new_page=chapter_new_page); i += 1; continue
        if s.startswith("::heading"):
            flush(); T.unnumbered_heading(clean(s[len("::heading"):].strip(), lineno)); i += 1; continue
        if s == "::pagebreak":
            flush(); T.doc.add_page_break(); i += 1; continue
        if s.startswith("::check"):
            flush()
            expr, _, msg = s[len("::check"):].partition("|")
            try:
                ok = bool(eval(expr.strip(), {"D": data, "abs": abs, "min": min, "max": max, "sum": sum,
                                              "len": len, "all": all, "any": any}))
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"{msg.strip()} (error: {exc})"
            if not ok:
                failures.append(f"line {lineno}: {msg.strip() or expr.strip()}")
            i += 1; continue
        if s.startswith("::note"):
            flush(); T.note(clean(s[len("::note"):].strip(), lineno)); i += 1; continue
        if s == "::refs":
            flush(); refs = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith(("::", "#")):
                r = lines[i].strip()
                if r:
                    r = re.sub(r"^(\[\d+\]|-)\s*", "", r)
                    refs.append(clean(r, i + 1))
                i += 1
            T.references(refs); continue
        if s.startswith("::table"):
            flush()
            caption = clean(s[len("::table"):].strip(), lineno)
            widths, font = None, 10.5
            i += 1
            while i < len(lines) and lines[i].strip().startswith(("::widths", "::font")):
                opt = lines[i].strip()
                if opt.startswith("::widths"):
                    widths = [float(x.replace(",", ".")) for x in re.split(r"[;\s]+|,\s+", opt[8:].strip()) if x]
                else:
                    font = float(opt[6:].strip().replace(",", "."))
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = lines[i].strip()
                if not re.fullmatch(r"\|?[\s:\-|]+\|?", r):
                    # fill values first: a {{x|f4}} placeholder contains a pipe of its own
                    filled = substitute(r, data, i + 1)
                    row = [cell(c) for c in split_row(filled)]
                    for c in row:
                        check_symbols(c[0] if isinstance(c, tuple) else c, f"line {i + 1}")
                    rows.append(row)
                i += 1
            if len(rows) < 2:
                raise SystemExit(f"line {lineno}: table needs a header row and at least one data row")
            width = len(rows[0])
            for k, r in enumerate(rows[1:], start=1):
                if len(r) != width:
                    raise SystemExit(f"line {lineno}: table row {k} has {len(r)} cells, header has {width}")
            if widths and len(widths) != width:
                raise SystemExit(f"line {lineno}: ::widths lists {len(widths)} columns, table has {width}")
            note = None
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].strip().startswith("::note"):
                note = clean(lines[i].strip()[len("::note"):].strip(), i + 1)
                i += 1
            header = [h[0] if isinstance(h, tuple) else h for h in rows[0]]
            T.table(caption, header, rows[1:], widths=widths, font_size=font, note=note)
            continue
        if s.startswith("::"):
            raise SystemExit(f"line {lineno}: unknown directive {s.split()[0]}")
        if not para_buf:
            para_line = lineno
        para_buf.append(s)
        i += 1
    flush()

    if failures:
        print("CHECK FAILED, nothing written:")
        for x in failures:
            print("  -", x)
        raise SystemExit(2)
    T.save(output_path)
    print("saved", output_path)


def to_pdf(docx_path):
    # The snap LibreOffice cannot read /tmp, so convert inside $HOME.
    src = Path(docx_path).resolve()
    work = src.parent if str(src).startswith(str(Path.home())) else Path.home() / ".cache" / "thesis-docs"
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / src.name
    if tmp != src:
        shutil.copy(src, tmp)
    subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(work), str(tmp)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    pdf = tmp.with_suffix(".pdf")
    if tmp != src:
        shutil.move(pdf, src.with_suffix(".pdf"))
        os.remove(tmp)
        pdf = src.with_suffix(".pdf")
    print("saved", pdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render a thesis-style .docx from a content file.")
    ap.add_argument("content")
    ap.add_argument("output")
    ap.add_argument("--data", default=None)
    ap.add_argument("--pdf", action="store_true", help="also write a PDF next to the .docx")
    a = ap.parse_args()
    render(a.content, a.output, a.data)
    if a.pdf:
        to_pdf(a.output)
