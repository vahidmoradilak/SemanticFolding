"""Phase 1 executor: rebuild chapter 2 of thesis_master.md from v02.1 + legacy."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from integrate_lib import clean, ren_head, slice_mas, slice_v, wrap_ascii  # noqa

ROOT = Path(__file__).resolve().parents[1]
MAS = ROOT / "docs" / "thesis" / "fa" / "thesis_master.md"
BAK = MAS.with_suffix(".pre_ch2.md")

# ── pull v02.1 parts (each ends before next sibling heading) ───────────────
V = {
    "ir":    slice_v(r"^## \*\*2-2\. بازیابی اطلاعات", r"^## \*\*2-3\."),
    "qa":    slice_v(r"^## \*\*2-3\. پاسخ", r"^## \*\*2-4\."),
    "mul":   slice_v(r"^## \*\*2-4\. بازیابی چندزبانه", r"^## \*\*2-5\."),
    "tfidf": slice_v(r"^## \*\*2-5\. نمایش متون", r"^## \*\*2-6\."),
    "sd":    slice_v(r"^## \*\*2-6\. نمایش‌های برداری", r"^## \*\*2-7\."),
    "sf":    slice_v(r"^## \*\*2-7\. Semantic Folding", r"^## \*\*2-8\."),
    "dim":   slice_v(r"^## \*\*2-8\. کاهش بُعد", r"^## \*\*2-9\."),
    "fp":    slice_v(r"^## \*\*2-9\. اثرانگشت‌های متنی", r"^## \*\*2-10\."),
    "spl":   slice_v(r"^## \*\*2-10\. SPLADE", r"^### \*\*منابع اصلی پیشنهادی"),
    "met":   slice_v(r"^## \*\*2-11\. معیارهای ارزیابی", r"^## \*\*2-12\."),
    "ds":    slice_v(r"^## \*\*2-12\. مجموعه‌داده‌های ارزیابی", r"^## \*\*2-13\."),
    "rw":    slice_v(r"^## \*\*2-13\. پیشینه پژوهش", r"^## \*\*2-14\."),
}

# ── legacy blocks from current master ──────────────────────────────────────
L_htm = slice_mas("## ۲-۴. مبانی زیست‌الهامی", "\n## ۲-۵.")
L_tab21 = slice_mas("Word2Vec و GloVe فضاهای پیوسته",
                    "با سازوکار شبکهٔ واژگان")
L_tab21 += "* با سازوکار شبکهٔ واژگان (فصل ۳) بهبود می‌یابد.\n"
L_prior = slice_mas("## ۲-۶. کارهای پیشین", "## ۲-۷. جمع‌بندی فصل")
L_place = re.search(r"\*\*جایگاه این پژوهش:\*\*.*?\n", MAS.read_text(encoding="utf-8"), re.S).group(0)
L_sum = slice_mas("## ۲-۷. جمع‌بندی فصل", "\n---")

# ── transforms on v-parts ──────────────────────────────────────────────────
def prep(txt, base):
    txt = ren_head(clean(wrap_ascii(txt)), base)
    return txt.rstrip() + "\n"

parts = {k: prep(v, i + 2) for i, (k, v) in enumerate(V.items())}

# citation mapping inside inserted text
CITES = [
    (r"\(Webber,? 2015\)|\(Webber et al\.?,? 2015\)", "[13]"),
    (r"\(Hawkins( and| &)? Ahmad,? 2016\)", "[10]"),
    (r"\(Karpukhin et al\.?,? 2020\)", "[61]"),
    (r"\(Formal et al\.?,? 2021\)", "[37], [38]"),
    (r"\(Robertson (&|and) Zaragoza,? 2009\)", "[39]"),
    (r"\(Asai et al\.?,? 2021\)", "[62]"),
    (r"\(Cormack et al\.?,? 2009\)", "[63]"),
    (r"Cortical\.io", "Cortical.io [36]"),
    (r"\(Clark et al\.?,? 2021\)", "[59]"),
    (r"\(Zhang et al\.?,? 202[34]\)", "[60]"),
]
for k in list(parts):
    t = parts[k]
    for a, b in CITES:
        t = re.sub(a, b, t)
    t = t.replace("CAREN_Belebele", "CAREN (custom_ar_en)")
    parts[k] = t

# drop the 'منابع اصلی پیشنهادی' leftover heading if any slipped into spl part
parts["spl"] = re.sub(r"منابع اصلی پیشنهادی برای پایان‌نامه\n?", "", parts["spl"])

# ── compose new chapter 2 ──────────────────────────────────────────────────
intro = ("## ۲-۱. مقدمه\n\nدر این فصل ابتدا چارچوب‌های پایهٔ بازیابی اطلاعات و "
         "پاسخ‌گویی به سوال مرور می‌شود؛ سپس به بازیابی چندزبانه/میان‌زبانی و "
         "مجموعه‌داده‌های استاندارد آن پرداخته می‌شود. بخش‌های میانی، نمایش متون "
         "(واژگانی، تنک، متراکم)، نظریهٔ خوشه‌بندی معنایی و اجزای سازندهٔ "
         "اثرانگشت‌ها را تشریح می‌کنند و در پایان، معیارهای ارزیابی، مجموعه‌داده‌ها "
         "و مرور ساختارمند پیشینه ارائه می‌گردد.\n\n")

head2 = "# فصل ۲: مبانی نظری و پیشینه پژوهش\n\n"

new_ch2 = (
    head2 + intro
    + parts["ir"] + "\n"
    + parts["qa"] + "\n"
    + parts["mul"] + "\n"
    + parts["tfidf"] + "\n"
    + parts["sd"] + "\n"
    + L_tab21 + "\n"
    + L_htm + "\n"
    + parts["sf"]
    + "- **استقلال زبانی / تحمل ابهام / حساب معنایی / سرعت:** ویژگی‌های رفتاری این بازنمایی که در پروپوزال نیز بر آن‌ها تأکید شده (نمونه‌ها: «فلسفه» در چند زبان؛ apple؛ یوزپلنگ/پورشه/ببر).\n\n"
    + parts["dim"] + "\n"
    + parts["fp"] + "\n"
    + parts["spl"] + "\n"
    + parts["met"] + "\n"
    + parts["ds"] + "\n"
    + parts["rw"] + "\n"
    + L_place + "\n"
    + L_prior.replace("## ۲-۶.", "## ۲-۱۵-الف.") + "\n"
    + L_sum.replace("## ۲-۷.", "## ۲-۱۵.")
)

# fix duplicated 'کارهای پیشین' numbering style
new_ch2 = new_ch2.replace("## ۲-۱۵-الف. کارهای پیشین مبتنی بر خوشه‌بندی معنایی",
                          "### ۲-۱۵-۱. کارهای پیشین مبتنی بر خوشه‌بندی معنایی")
new_ch2 = new_ch2.replace("## ۲-۱۵. جمع‌بندی فصل", "## ۲-۱۶. جمع‌بندی فصل")

# ── splice into master ─────────────────────────────────────────────────────
mas_text = MAS.read_text(encoding="utf-8")
BAK.write_text(mas_text, encoding="utf-8")
s = re.search(r"(?m)^# فصل ۲:[^\n]*", mas_text).start()
e = re.search(r"(?m)^# فصل ۳:[^\n]*", mas_text).start()
new_all = mas_text[:s] + new_ch2.strip() + "\n\n---\n\n" + mas_text[e:]
MAS.write_text(new_all, encoding="utf-8")

h_count = len(re.findall(r"(?m)^#{2,3} ", new_ch2))
print(f"chapter2 rebuilt: {len(new_ch2)} chars | subsections={h_count} | backup -> {BAK.name}")
