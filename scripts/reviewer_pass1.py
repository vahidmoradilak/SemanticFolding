"""Reviewer pass #1: mechanical fixes over thesis_master.md"""
import re
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "docs" / "thesis" / "fa" / "thesis_master.md"
text = p.read_text(encoding="utf-8")

protected = []
def stash(m):
    protected.append(m.group(0)); return f"\x00{len(protected)-1}\x00"
text = re.sub(r"```.*?```", stash, text, flags=re.S)
text = re.sub(r"`[^`\n]+`", stash, text)

# 1) ZWNJ directly before short function words is always wrong
FUNCS = ["را", "از", "است", "هستند", "هست", "بود", "به", "در", "با", "که",
         "تا", "نیز", "هم", "یک", "این", "آن"]
n = 0
for f in FUNCS:
    pat = "\u200c" + f
    c = text.count(pat)
    if c:
        n += c
        text = text.replace(pat, " " + f)

# 2) targeted ezafe / wording fixes
fixes = {
    "۱۳ مجموعه‌داده": "۱۲ مجموعه‌داده",
    "بخش‌بندی مفهومی‌پیشنهاد": "بخش‌بندی مفهومی پیشنهاد",
    "تراکمی‌اجراشده": "تراکمیِ اجراشده",
    "تراکمی‌واقعی": "تراکمیِ واقعی",
    "(all-MiniLM-L6-v2`، 384بعدی،": "(all-MiniLM-L6-v2` با ۳۸۴ بُعد،",
    " روی بخشی از داده تیون ": " روی بخشی از داده تنظیم ",
    "معیارها (ابعاد گرید، σ هموارسازی، top-percent، نوع وزن‌دهی، تعداد انتشار) روی بخشی از داده تیون":
        "معیارها (ابعاد گرید، σ هموارسازی، top-percent، نوع وزن‌دهی، تعداد انتشار) روی بخشی از داده تنظیم",
}
for a, b in fixes.items():
    if a in text:
        n += text.count(a)
        text = text.replace(a, b)

# generic leftover: 'داده تیون'
text = text.replace("داده تیون", "داده تنظیم")

text = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)
p.write_text(text, encoding="utf-8")
print(f"mechanical fixes applied: {n}")
