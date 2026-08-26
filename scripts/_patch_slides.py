from pathlib import Path

p = Path(__file__).resolve().parents[1] / "scripts" / "build_defense_slides.py"
t = p.read_text(encoding="utf-8")

old_b = '"**انگلیسی عمومی:** Belebele، NarrativeQA، PubMedQA، PopQA، MuSiQue",'
new_b = old_b + '\n    "**چندزبانه استاندارد:** Mr.TyDi-ar و MIRACL-ar (پروتکل pooled-100)",'
assert old_b in t
t = t.replace(old_b, new_b)

old_c = "چارچوب بنچمارک ۱۲ مجموعه‌داده‌ای"
assert old_c in t
t = t.replace(old_c, "چارچوب بنچمارک ۱۴ مجموعه‌داده‌ای")

anchor = "# 13 ── WordNet lesson"
new_slide = '''# 12b ── multilingual standard benchmarks
s = add_slide("بنچمارک‌های چندزبانه استاندارد عربی (TyDi / MIRACL)")
bullets(s, [
    "پروتکل pooled-100: هر پرسش = سند مرتبط رسمی + ۹۹ نگتیو از کورپوس کامل",
    "TyDi-ar: SF=0.544 | BM25=0.881 | E5-multi=0.989",
    "MIRACL-ar: SF=0.542 | BM25=0.815 | E5=0.860",
    "**تفسیر صادقانه:** در ویکی‌پدیا بازدامنه، تراکمی چندزبانه غالب است؛",
    "نقطه قوت SF در دامنه بسته تخصصی (قرآن/دوزبانه دینی/SciFact) تأیید مجدد شد",
], size=17)

'''
assert anchor in t
t = t.replace(anchor, new_slide + anchor)

# also update neural slide to mention E5 sweep
old_n = '"**SciFact (اجرا مستقیم):** SF+SPLADE 0.966 > BM25 0.947 > MiniLM-L6-v2 0.865",'
new_n = ('"**SciFact:** SF+SPLADE 0.966 > BM25 0.947 > MiniLM 0.865",\n'
         '    "E5-multi روی ۸ دیتاست اجرا شد — در ۶ مورد Best-SF جلوتر (جزئیات جدول ۴-۸)",')
assert old_n in t
t = t.replace(old_n, new_n)

p.write_text(t, encoding="utf-8")
print("slides script patched")
