"""Renumber all section headings inside thesis chapter 2 properly."""
import re
from pathlib import Path

FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
p = Path(__file__).resolve().parents[1] / "docs" / "thesis" / "fa" / "thesis_master.md"
t = p.read_text(encoding="utf-8")

s = re.search(r"(?m)^# فصل ۲:", t).start()
e = re.search(r"(?m)^# فصل ۳:", t).start()
ch2 = t[s:e]

n2 = n3 = 0
out = []
for ln in ch2.splitlines():
    m = re.match(r"^(#{2,3})\s+(.*)$", ln)
    if not m:
        out.append(ln); continue
    hashes, title = m.group(1), m.group(2)
    # strip any existing numeric prefix like '۲-۳-۴.' or '13-1.'
    title = re.sub(r"^[\d۰-۹]+(?:[-٫.][\d۰-۹]+)*\.?\s*", "", title).strip()
    if hashes == "##":
        if "جمع‌بندی فصل" in title or "جایگاه پژوهش حاضر" == title:
            pass
        n2 += 1; n3 = 0
        label = f"۲-{n2}."
    else:
        # 'کارهای پیشین...' should be a sub-section of the related-work chapter
        n3 += 1
        label = f"۲-{n2}-{n3}."
    out.append(f"{hashes} {label} {title}".replace("۲-", "۲-").translate(FA)
               if False else f"{hashes} {label} {title}")

ch2_new = "\n".join(out)
t = t[:s] + ch2_new + t[e:]
p.write_text(t, encoding="utf-8")
print("renumbered:", len(re.findall(r'(?m)^#{2,3} ', ch2_new)), "headings")
