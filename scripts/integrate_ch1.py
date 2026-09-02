"""Phase 2: rebuild chapter 1 of thesis_master.md based on v02.1 (absorb) + ours."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V02 = ROOT / "docs" / "thesis" / "thesis_v02.1.cloude.md"
MAS = ROOT / "docs" / "thesis" / "fa" / "thesis_master.md"
BAK = MAS.with_suffix(".pre_ch1.md")

CITES = [
    (r"\(Webber,? 2015\)", "[13]"),
    (r"\(Formal et al\.?,? 2021\)", "[37], [38]"),
    (r"\(Karpukhin et al\.?,? 2020\)", "[61]"),
    (r"\(Hawkins( and| &)? Ahmad,? 20\d\d\)", "[10]"),
    (r"\(Cortical\.io,? 2017?\)", "[36]"),
]


def clean(txt):
    txt = re.sub(r"\\([=_#*<>|\[\]()])", r"\1", txt)
    txt = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", txt)
    return re.sub(r"[ \t]+\n", "\n", txt)


def map_cites(txt):
    for a, b in CITES:
        txt = re.sub(a, b, txt)
    return txt


def v_section(sec_no):
    t = V02.read_text(encoding="utf-8")
    m = re.search(rf"(?ms)^## \*\*{sec_no}\..*?(?=^## \*\*1-|\Z)", t)
    assert m, sec_no
    return clean(map_cites(m.group(0)))


def m_section(a, b=None):
    t = MAS.read_text(encoding="utf-8")
    s = re.search(re.escape(a), t)
    assert s, a[:40]
    e = re.search(re.escape(b), t[s.end():]) if b else None
    end = s.end() + (e.start() if e else len(t) - s.end())
    return t[s.start():end].rstrip() + "\n"


# ── v02.1 chapter-1 sections ───────────────────────────────────────────────
v_intro = v_section("1-1")
v_nece = v_section("1-2")
v_prob = v_section("1-3")
v_goal = v_section("1-4")
v_method = v_section("1-5")

# ── legacy (ours) blocks ───────────────────────────────────────────────────
o_subq = m_section("سه زیرمسئلهٔ اصلی از این تعریف قابل استخراج است:", "\n## ۱-۳.")
o_goal = m_section("## ۱-۳. اهداف پژوهش", "\n## ۱-۴.")
o_chal = m_section("## ۱-۵. چالش‌های پیش رو", "\n## ۱-۶.")
o_nov = m_section("## ۱-۶. نوآوری‌ها و دستاوردها", "\n## ۱-۷.")
o_struct = m_section("## ۱-۷. ساختار پایان‌نامه", "\n---")

# ── compose ────────────────────────────────────────────────────────────────
head = "# فصل ۱: مقدمه\n\n"

new = head
new += v_intro.replace("## **1-1.", "## ۱-۱.").replace("**", "") .rstrip() + "\n\n"

new += v_nece.replace("## **1-2.", "## ۱-۲.").replace("**", "").rstrip()
new += "\n\n" + o_chal.split("\n", 1)[1].rstrip() + "\n\n"          # چالش‌ها به‌عنوان ادامهٔ ضرورت

new += v_prob.replace("## **1-3.", "## ۱-۳.").replace("**", "").rstrip()
new += "\n\n### ۱-۳-۱. زیرمسئله‌های عملیاتی پژوهش\n\n" + \
       o_subq.split("\n", 1)[1].rstrip() + "\n\n"

new += v_goal.replace("## **1-4.", "## ۱-۴.").replace("**", "").rstrip()
goal_body = o_goal.split("\n", 1)[1].rstrip()
new += "\n\n**اهداف جزئی این پایان‌نامه:**\n\n" + goal_body + "\n\n"

new += v_method.replace("## **1-5.", "## ۱-۵.").replace("**", "").rstrip() + "\n\n"

new += "## ۱-۶. نوآوری‌ها و دستاوردهای پژوهش\n\n" + \
       o_nov.split("\n", 1)[1].rstrip() + "\n\n"

new += "## ۱-۷. ساختار پایان‌نامه\n\n" + \
       o_struct.split("\n", 1)[1].rstrip() + "\n"

# ── splice ────────────────────────────────────────────────────────────────
mas = MAS.read_text(encoding="utf-8")
BAK.write_text(mas, encoding="utf-8")
s = re.search(r"(?m)^# فصل ۱:[^\n]*", mas).start()
e = re.search(r"(?m)^# فصل ۲:", mas).start()
mas_new = mas[:s] + new.strip() + "\n\n---\n\n" + mas[e:]
MAS.write_text(mas_new, encoding="utf-8")

print(f"chapter1 rebuilt: {len(new)} chars | backup -> {BAK.name}")
