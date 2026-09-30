"""Quality pass on editable thesis SVGs (in place, git-tracked).
1. Persian-capable font stack (Vazirmatn > Tahoma > Noto Sans Arabic > DejaVu Sans).
2. Minimum body font 12px (11px labels in gauss.svg -> 12px).
3. Hairline strokes in gauss.svg (0.4/0.6px) thickened for print survival.
"""
import glob, re

OLD_FONT = 'font-family="DejaVu Sans, Arial, sans-serif"'
NEW_FONT = 'font-family="Vazirmatn, Tahoma, \'Noto Sans Arabic\', \'DejaVu Sans\', Arial, sans-serif"'

for f in sorted(glob.glob("docs/thesis/filesCloude/thesis_figures/svg_editable_text/*.svg")):
    t = open(f, encoding="utf-8").read()
    orig = t
    t = t.replace(OLD_FONT, NEW_FONT)
    t = re.sub(r'font-size="11"', 'font-size="12"', t)
    if f.endswith("gauss.svg"):
        t = re.sub(r'stroke-width="0\.4"', 'stroke-width="0.8"', t)
        t = re.sub(r'stroke-width="0\.6"', 'stroke-width="0.9"', t)
    if t != orig:
        open(f, "w", encoding="utf-8").write(t)
        print("updated:", f.split("\\")[-1])
print("done")
