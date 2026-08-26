"""Phase 1: absorb v02.1 theory sections into fa/thesis_master.md chapter 2."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V02 = ROOT / "docs" / "thesis" / "thesis_v02.1.cloude.md"
MAS = ROOT / "docs" / "thesis" / "fa" / "thesis_master.md"

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def clean(txt: str) -> str:
    txt = re.sub(r"\\\n?", "", txt)                      # escaped line breaks
    txt = re.sub(r"\\([=_#*<>|\[\]])", r"\1", txt)        # escaped symbols
    txt = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                 lambda m: m.group(1), txt)               # links -> label
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    return txt


def ren_head(txt: str, base: int) -> str:
    """Normalize v02 headings '## **2-x.y Title**' -> '## ۲-base-x.y. Title'."""
    def rep(m):
        lvl, num, title = m.group(1), m.group(2), m.group(3).strip("*").strip()
        depth = num.count(".")
        prefix = f"{base}-" if depth == 0 else f"{base}-{num.split('.')[0]}."
        tail = "" if depth == 0 else "." if depth == 1 else ""
        return "#" * len(m.group(1)) + " " + (prefix + num.split(".")[-1]).translate(FA_DIGITS) + ". " + title
    return re.sub(r"^(#{2,4})\s+\*\*(\d+(?:-\d+)*(?:\.\d+)*)\.?\s*(.+?)\*\*\s*$",
                  rep, txt, flags=re.M)


def slice_v(start_pat: str, end_pat: str) -> str:
    t = V02.read_text(encoding="utf-8")
    s = re.search(start_pat, t, re.M)
    e = re.search(end_pat, t, re.M)
    assert s and e and s.start() < e.end(), (start_pat, end_pat)
    return clean(t[s.start():e.start()])


def slice_mas(a: str, b: str) -> str:
    t = MAS.read_text(encoding="utf-8")
    s = re.search(re.escape(a), t, re.M)
    assert s, ("start missing", a[:50])
    e = re.search(re.escape(b), t[s.end():], re.M)
    assert e, ("end missing", b[:50])
    return t[s.start():s.end() + e.start()]


def wrap_ascii(txt: str) -> str:
    """Fence runs of >=3 non-Persian lines (ASCII diagrams)."""
    out, buf = [], []
    FA = re.compile(r"[\u0600-\u06FF]")
    for ln in txt.splitlines():
        if ln.strip() and not FA.search(ln) and not ln.lstrip().startswith(("#", "|")):
            buf.append(ln)
        else:
            if len(buf) >= 3:
                out.append("```text")
                out += buf
                out.append("```")
            else:
                out += buf
            buf = []
            out.append(ln)
    out += buf
    return "\n".join(out)


if __name__ == "__main__":
    print("lib ok")
