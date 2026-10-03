"""One-shot patcher: adds --n-enroll (strict k-window enrollment calibration)
to scripts/run_experiment.py. Backs up first; restores on any failure.
Run from the WheelsEye_V2 root with calibration_enroll.py in the same dir:
    python patch_enroll.py
"""
import ast, re, shutil, sys
from pathlib import Path

RE = Path("scripts/run_experiment.py")
BAK = Path("scripts/run_experiment.py.pre_enroll.bak")
MOD = Path("calibration_enroll.py")

def fail(msg):
    if BAK.exists(): shutil.copy2(BAK, RE); print("restored original run_experiment.py")
    sys.exit(f"PATCH FAILED: {msg}")

if not RE.exists() or not MOD.exists(): sys.exit("run from V2 root with calibration_enroll.py present")
shutil.copy2(RE, BAK); shutil.copy2(MOD, Path("common") / MOD.name)
src = RE.read_text(); lines = src.splitlines(keepends=True); changes = []

# 1. import + dispatcher after the existing calibration import
i = next((k for k,l in enumerate(lines) if "from common.calibration import apply_alert_baseline" in l), None)
if i is None: fail("calibration import line not found")
lines.insert(i+1,
 "from common.calibration_enroll import apply_enrollment_baseline\n"
 "_N_ENROLL = 0  # set from --n-enroll; 0 = whole-session alert baseline (original behaviour)\n"
 "def _calibrate(df, features, *a, **k):\n"
 "    if _N_ENROLL:\n"
 "        return apply_enrollment_baseline(df, features, n_enroll=_N_ENROLL)[0]\n"
 "    return apply_alert_baseline(df, features, *a, **k)\n")
changes.append("dispatcher added")

# 2. route existing call sites through the dispatcher (not the import/def lines)
n=0
for k,l in enumerate(lines):
    if "apply_alert_baseline(" in l and "import" not in l and "def _calibrate" not in l and "return apply_alert_baseline" not in l:
        lines[k] = l.replace("apply_alert_baseline(", "_calibrate("); n+=1
if n == 0: fail("no apply_alert_baseline( call site found")
changes.append(f"{n} call site(s) routed")

# 3. argparse flag next to --calibrate
j = next((k for k,l in enumerate(lines) if re.search(r'add_argument\(\s*["\']--calibrate["\']', l)), None)
if j is None: fail("--calibrate add_argument not found")
m = re.match(r'(\s*)(\w+)\.add_argument', lines[j]); indent, pvar = m.group(1), m.group(2)
# skip to end of that add_argument statement (may span lines)
e = j
while lines[e].count("(") > lines[e].count(")") or (e>j and lines[e-1].count("(")>lines[e-1].count(")") and lines[e].strip().startswith(("help","type","action","default"))):
    e += 1
lines.insert(e+1, f'{indent}{pvar}.add_argument("--n-enroll", type=int, default=0,\n'
                  f'{indent}    help="strict enrollment: baseline from first K Awake windows only (0 = whole session)")\n')
changes.append("--n-enroll flag added")

# 4. set the module global right after parse_args()
p = next((k for k,l in enumerate(lines) if re.search(r'(\w+)\s*=\s*\w+\.parse_args\(\)', l)), None)
if p is None: fail("parse_args() line not found")
avar = re.search(r'(\w+)\s*=\s*\w+\.parse_args\(\)', lines[p]).group(1)
ind = re.match(r'(\s*)', lines[p]).group(1)
lines.insert(p+1, f'{ind}globals()["_N_ENROLL"] = int(getattr({avar}, "n_enroll", 0) or 0)\n')
changes.append("global wired after parse_args")

# 5. result filename suffix after the smooth suffix line
q = next((k for k,l in enumerate(lines) if '_smooth{' in l and 'if args.smooth else' in l), None)
if q is None: fail("smooth suffix line not found (filename tagging)")
ind = re.match(r'(\s*)', lines[q]).group(1)
lines.insert(q+1, f'{ind}+ (f"_enroll{{args.n_enroll}}" if getattr(args, "n_enroll", 0) else "")\n')
changes.append("filename suffix added")

new = "".join(lines)
try: ast.parse(new)
except SyntaxError as ex: fail(f"syntax error after patch: {ex}")
RE.write_text(new)
print("OK:", "; ".join(changes)); print(f"backup at {BAK}")
