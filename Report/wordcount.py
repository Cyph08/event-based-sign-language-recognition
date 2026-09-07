"""Body word count for the rubric: Introduction..Conclusion, excluding tables,
code, captions, headings, references and appendices."""
import re, sys

def count(path="Final.md", verbose=True):
    txt = open(path, encoding="utf-8").read()
    start = txt.index("\n## 1. Introduction")
    lines = txt[start:].split("\n")
    body, incode, stop = [], False, False
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            incode = not incode; continue
        if incode: continue
        if re.match(r"^##\s+(References|Appendix)", s): stop = True
        if stop: continue
        if s.startswith(("|", ">", "<!--")): continue
        if re.match(r"^!\[", s): continue
        if re.match(r"^\*\*(Table|Figure)\s", s): continue
        if s.startswith("#"): continue
        if not s: continue
        body.append(s)
    t = re.sub(r"[*_`]", "", " ".join(body))
    n = len(t.split())
    if verbose:
        print(f"body words: {n}   (rubric 7,000-9,000)   headroom: {9000-n:+d}")
    return n

if __name__ == "__main__":
    count(sys.argv[1] if len(sys.argv) > 1 else "Final.md")
