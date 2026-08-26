from pathlib import Path
import re

p = Path(__file__).resolve().parents[1] / "scripts" / "bench_pooled.py"
t = p.read_text(encoding="utf-8")

# stringify every numeric P[...] passed as a CLI arg value
t = re.sub(r'(P\["(?:grid|seed|nn|md|mwl|mf|sigma|tp)"\])',
           lambda m: "str(" + m.group(1) + ")", t)
# avoid double-wrap inside already-written str(P[..])
t = t.replace('str(str(', 'str(')

p.write_text(t, encoding="utf-8")
leftover = re.findall(r'(?<!str\()P\["(?:grid|seed|nn|md|mwl|mf|sigma|tp)"\]\)', t)
print("done; unwrapped leftovers:", len(leftover))
