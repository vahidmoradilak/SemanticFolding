"""Persian typography normalisation pass over thesis_master.md (task 9).

Protects fenced code blocks + inline backticks, then applies:
  * Arabic Yeh/Kaf -> Persian Yeh/Kaf
  * ZWNJ after prefixes می/نمی before Persian letters
  * common broken compounds (پایان نامه، هم چنین، ...)
"""
import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "docs" / "thesis" / "fa" / "thesis_master.md"
text = p.read_text(encoding="utf-8")

# ── protect code regions ────────────────────────────────────────────────────
protected = []

def stash(m):
    protected.append(m.group(0))
    return f"\x00{len(protected)-1}\x00"

text = re.sub(r"```.*?```", stash, text, flags=re.S)      # fenced blocks
text = re.sub(r"`[^`\n]+`", stash, text)                   # inline code

n_yk = len(re.findall(r"[يك]", text))
text = text.replace("ي", "ی").replace("ك", "ک")

# prefix + space -> prefix + ZWNJ (before Persian letters only)
pat_prefix = re.compile(r"(می|نمی)(?: |\u00a0)(?=[\u067E\u0686\u0698\u06CC\u06A9\u0627-\u064A])")
n_mi = len(pat_prefix.findall(text))
text = pat_prefix.sub(lambda m: m.group(1) + "\u200c", text)

compounds = {
    "پایان نامه": "پایان‌نامه",
    "هم چنین": "همچنین",
    "هر کدام": "هرکدام",
    "اثر انگشت": "اثرانگشت",
    "واژه نامه": "واژه‌نامه",
    "بی معنا": "بی‌معنا",
    "بی ضرر": "بی‌ضرر",
    "غیر فعال": "غیرفعال",
}
n_comp = 0
for a, b in compounds.items():
    n_comp += text.count(a)
    text = text.replace(a, b)

text = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)

p.write_text(text, encoding="utf-8")
print(f"yeh/kaf fixed: {n_yk} | mi-ZWNJ fixed: {n_mi} | compounds fixed: {n_comp}")
