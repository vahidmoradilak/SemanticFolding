from pathlib import Path
import re

p = Path(__file__).resolve().parents[1] / "scripts" / "integrate_ch2.py"
t = p.read_text(encoding="utf-8")

t = t.replace('mas_text.index("# فصل ۲: مبانی نظری و پیشینه پژوهش")',
              're.search(r"(?m)^# فصل ۲:[^\\n]*", mas_text).start()')
t = t.replace('mas_text.index("# فصل ۳: روش پژوهش")',
              're.search(r"(?m)^# فصل ۳:[^\\n]*", mas_text).start()')

p.write_text(t, encoding="utf-8")
print("patched anchors")
assert "re.search" in t and "index(" not in t
