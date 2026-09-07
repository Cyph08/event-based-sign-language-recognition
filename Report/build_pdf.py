"""Markdown -> styled HTML -> PDF via headless Chrome, with University front matter.

Two passes. Pass 1 lays the document out so we can discover which printed page each
heading and caption lands on; pass 2 rebuilds with a real Contents, List of Tables and
List of Figures carrying those page numbers. Page numbers are then stamped onto the
pages themselves (roman for front matter, arabic from the Introduction onward), because
Chrome does not implement CSS margin boxes.

    python build_pdf.py Final
"""
import base64, io, os, re, subprocess, sys
import markdown
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = sys.argv[1] if len(sys.argv) > 1 else "Final"
SRC = os.path.join(HERE, f"{STEM}.md")
HTML = os.path.join(HERE, f"{STEM}.html")
PDF = os.path.join(HERE, f"{STEM}.pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

RAW = open(SRC, encoding="utf-8").read()


# ---------------------------------------------------------------- structure
def outline(md):
    """Headings, table captions and figure captions, in document order."""
    heads, tables, figs = [], [], []
    md = md[md.index("\n## Abstract"):]      # skip the title page's own headings
    for ln in md.split("\n"):
        s = ln.strip()
        m = re.match(r"^(#{2,3})\s+(.*)", s)
        if m:
            title = m.group(2).strip()
            if title in ("Contents", "List of Tables", "List of Figures"):
                continue
            heads.append((len(m.group(1)), title))
            continue
        m = re.match(r"^\*\*(Table|Figure) (\d+):\*{0,2}\s*(.*)", s)
        if m:
            cap = re.sub(r"[*`]+", "", m.group(3)).strip()
            cap = cap.split(". ")[0].rstrip(".")
            (tables if m.group(1) == "Table" else figs).append(
                (f"{m.group(1)} {m.group(2)}", cap[:78]))
    return heads, tables, figs


HEADS, TABLES, FIGS = outline(RAW)


def toc_html(pages=None):
    rows = []
    for lvl, title in HEADS:
        pg = (pages or {}).get(title, "")
        cls = "toc1" if lvl == 2 else "toc2"
        rows.append(f'<div class="{cls}"><span class="t">{title}</span>'
                    f'<span class="d"></span><span class="p">{pg}</span></div>')
    return '<h2 class="fm">Contents</h2>' + "".join(rows)


def list_html(kind, items, pages=None):
    rows = []
    for label, cap in items:
        pg = (pages or {}).get(label, "")
        rows.append(f'<div class="toc2"><span class="t">{label} &nbsp;{cap}</span>'
                    f'<span class="d"></span><span class="p">{pg}</span></div>')
    return f'<h2 class="fm">{kind}</h2>' + "".join(rows)


# ---------------------------------------------------------------- rendering
def embed(m):
    alt, path = m.group(1), m.group(2)
    full = os.path.join(HERE, path.replace("/", os.sep))
    if not os.path.exists(full):
        return m.group(0)
    b64 = base64.b64encode(open(full, "rb").read()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}">'


CSS = """
@page { size: A4; margin: 22mm 18mm 20mm; }
body { font-family: 'Georgia','Times New Roman',serif; font-size: 10.5pt;
       line-height: 1.55; color: #111; }
h1 { font-size: 20pt; line-height: 1.3; margin: 0 0 6pt; font-weight: 600; border: none; }
h2 { font-size: 14pt; margin: 20pt 0 7pt; font-weight: 600;
     border-bottom: 1px solid #bbb; padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 5pt; font-weight: 600; page-break-after: avoid; }
h2.fm { border-bottom: none; margin-top: 0; }
p { margin: 0 0 8pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0 14pt;
        font-size: 8.6pt; font-family: 'Helvetica','Arial',sans-serif;
        page-break-inside: avoid; }
th { background: #eef2f6; border-bottom: 1.5px solid #444; padding: 4pt 5pt;
     text-align: left; font-weight: 600; }
td { border-bottom: 1px solid #dde3e9; padding: 3.5pt 5pt; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }
code { font-family: 'Consolas','Courier New',monospace; font-size: 8.8pt;
       background: #f3f4f6; padding: 1px 3px; border-radius: 2px; }
pre { background: #f7f8fa; border: 1px solid #dfe3e8; border-left: 3px solid #666;
      padding: 8pt 10pt; font-size: 8pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { display: block; max-width: 100%; margin: 12pt auto 4pt; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #ccc; margin: 16pt 0; }
ul, ol { margin: 0 0 9pt; padding-left: 20pt; }
li { margin-bottom: 3.5pt; text-align: justify; }
blockquote { border-left: 3px solid #bbb; margin: 8pt 0; padding-left: 12pt; color: #444; }
.pagebreak { page-break-before: always; }

/* title page */
.titlepage { text-align: center; page-break-after: always; padding-top: 38mm; }
.titlepage h1 { font-size: 21pt; margin-bottom: 10pt; }
.titlepage h2 { font-size: 13pt; font-weight: 400; font-style: italic;
                border: none; margin: 0 0 40pt; color: #333; }
.titlepage p { text-align: center; margin: 0 0 6pt; }
.titlepage .decl { font-size: 10pt; letter-spacing: 0.3pt; line-height: 1.9;
                   margin-bottom: 26pt; }
.titlepage .yr { font-size: 12pt; margin-bottom: 34pt; }
.titlepage .nm { font-size: 14pt; font-weight: 600; margin-bottom: 20pt; }

/* contents / lists of tables and figures */
.toc1, .toc2 { display: flex; align-items: baseline; font-size: 10pt;
               margin: 2.5pt 0; }
.toc1 { font-weight: 600; margin-top: 7pt; }
.toc2 { padding-left: 14pt; font-weight: 400; }
.toc1 .d, .toc2 .d { flex: 1; border-bottom: 1px dotted #999; margin: 0 5pt 0 6pt;
                     transform: translateY(-2px); }
.toc1 .p, .toc2 .p { font-variant-numeric: tabular-nums; }
.wc { margin-top: 22pt; font-size: 10pt; font-weight: 600; }
"""


def render(pages=None):
    md = RAW
    md = md.replace("<!--CONTENTS-->", "@@TOC@@")
    md = md.replace("<!--LISTOFTABLES-->", "@@LOT@@")
    md = md.replace("<!--LISTOFFIGURES-->", "@@LOF@@")
    md = md.replace("<!--PAGEBREAK-->", '<div class="pagebreak"></div>')
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed, md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "attr_list"])

    wc = word_count()
    body = body.replace("<p>@@TOC@@</p>",
                        toc_html(pages) + f'<div class="wc">Word Count: {wc:,}</div>')
    body = body.replace("<p>@@LOT@@</p>", list_html("List of Tables", TABLES, pages))
    body = body.replace("<p>@@LOF@@</p>", list_html("List of Figures", FIGS, pages))

    # title page: everything from the marker to the first pagebreak
    body = body.replace("<!--TITLEPAGE-->", '<div class="titlepage">', 1)
    body = body.replace('<div class="pagebreak"></div>', "</div>", 1)
    body = re.sub(r"<p>A DISSERTATION SUBMITTED.*?</p>",
                  lambda m: '<p class="decl">' + m.group(0)[3:-4] + "</p>", body, flags=re.S)
    body = body.replace("<p>2026</p>", '<p class="yr">2026</p>', 1)
    body = body.replace("<p><strong>Deepen Khandelwal</strong></p>",
                        '<p class="nm">Deepen Khandelwal</p>', 1)

    open(HTML, "w", encoding="utf-8").write(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{STEM}</title><style>{CSS}</style></head><body>{body}</body></html>")


def word_count():
    start = RAW.index("\n## 1. Introduction")
    body, incode, stop = [], False, False
    for ln in RAW[start:].split("\n"):
        s = ln.strip()
        if s.startswith("```"):
            incode = not incode; continue
        if incode: continue
        if re.match(r"^##\s+(References|Appendix)", s): stop = True
        if stop or s.startswith(("|", ">", "<!--")) or not s: continue
        if re.match(r"^!\[", s) or re.match(r"^\*\*(Table|Figure)\s", s): continue
        if s.startswith("#"): continue
        body.append(s)
    return len(re.sub(r"[*_`]", "", " ".join(body)).split())


def chrome():
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", f"file:///{HTML.replace(os.sep,'/')}"],
                   check=True, capture_output=True, timeout=300)


# ------------------------------------------------- locate items in the layout
def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


""" Anchor on prose unique to the body, never on a heading: every heading also
appears in the Contents, so matching a heading finds the Contents page first. """
BODY_ANCHOR = "Conventional cameras sample the world on a fixed clock"


def locate():
    """Map each heading/caption to the printed page it starts on."""
    r = PdfReader(PDF)
    ptext = [norm(pg.extract_text() or "") for pg in r.pages]
    body_start = next((i for i, t in enumerate(ptext) if norm(BODY_ANCHOR) in t), 1)

    # Contents / List of Tables / List of Figures name every section before it
    # appears, so they must not be treated as the location of anything.
    global NAV
    NAV = [t.startswith(("contents", "listoftables", "listoffigures"))
           or "wordcount" in t[:200] for t in ptext]

    found = {}
    targets = ([(t, t) for _, t in HEADS]
               + [(lbl, f"{lbl}: {cap}") for lbl, cap in TABLES + FIGS])
    for key, probe in targets:
        n = norm(probe)[:46]
        if not n:
            continue
        # body pages first; the Contents and the two lists live before body_start
        hit = next((i for i in range(body_start, len(ptext)) if n in ptext[i]), None)
        if hit is not None:
            found[key] = str(hit - body_start + 1)
            continue
        # Front-matter item. Skip the navigation pages, which name every section
        # before it appears, then take the FIRST match -- not the last, because
        # the Copyright text contains "declarations" and would otherwise capture
        # the Declaration entry.
        pre = [i for i in range(1, body_start)
               if n in ptext[i] and not NAV[i]]
        if pre:
            found[key] = _roman(pre[0] + 1)
    return found, body_start, len(r.pages)


def _roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, sym in vals:
        while n >= v:
            out += sym; n -= v
    return out


# ------------------------------------------------------- stamp page numbers
def stamp(body_start):
    r = PdfReader(PDF)
    w = PdfWriter()
    for i, page in enumerate(r.pages):
        if i == 0:                                  # no number on the title page
            w.add_page(page); continue
        label = _roman(i + 1) if i < body_start else str(i - body_start + 1)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.setFont("Times-Roman", 9.5)
        c.drawCentredString(A4[0] / 2, 12 * 2.3, label)
        c.save()
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        w.add_page(page)
    with open(PDF, "wb") as f:
        w.write(f)


if not os.path.exists(CHROME):
    sys.exit(f"Chrome not found at {CHROME}")

render()                       # pass 1: discover the layout
chrome()
pages, body_start, total = locate()
render(pages)                  # pass 2: real page numbers in the front matter
chrome()
pages, body_start, total = locate()
stamp(body_start)

print(f"HTML : {HTML}")
print(f"PDF  : {PDF}  ({os.path.getsize(PDF)/1024:.0f} KB, {total} pages)")
print(f"words: {word_count():,}   front matter: {body_start} pages")
print(f"toc  : {len(HEADS)} headings, {len(TABLES)} tables, {len(FIGS)} figures")
