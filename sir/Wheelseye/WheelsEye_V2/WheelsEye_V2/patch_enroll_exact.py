"""Exact-string patch for scripts/run_experiment.py (WheelsEye_V2): adds --n-enroll.
Run from the V2 root with calibration_enroll.py present. Backs up; aborts untouched on mismatch."""
import ast, shutil, sys
from pathlib import Path
RE, BAK = Path("scripts/run_experiment.py"), Path("scripts/run_experiment.py.pre_enroll.bak")
if not RE.exists() or not Path("calibration_enroll.py").exists(): sys.exit("run from V2 root with calibration_enroll.py present")
src = RE.read_text()
if "apply_enrollment_baseline" in src: sys.exit("already patched -- nothing to do (file untouched)")
edits = [
 ('from common.calibration import apply_alert_baseline\n',
  'from common.calibration import apply_alert_baseline\nfrom common.calibration_enroll import apply_enrollment_baseline\n'),
 ('        df = apply_alert_baseline(df, flat_features, mode="replace")\n',
  '        if args.n_enroll:\n'
  '            df, _enroll_mask = apply_enrollment_baseline(df, flat_features, n_enroll=args.n_enroll)\n'
  '        else:\n'
  '            df = apply_alert_baseline(df, flat_features, mode="replace")\n'),
 ('    parser.add_argument("--seed", type=int, default=0)\n',
  '    parser.add_argument("--seed", type=int, default=0)\n'
  '    parser.add_argument("--n-enroll", type=int, default=0,\n'
  '                        help="strict enrollment calibration: baseline from the first K Awake windows only (0 = whole session)")\n'),
 ('           + (f"_smooth{args.smooth}" if args.smooth else "")\n',
  '           + (f"_smooth{args.smooth}" if args.smooth else "")\n'
  '           + (f"_enroll{args.n_enroll}" if args.n_enroll else "")\n'),
]
for old, new in edits:
    if src.count(old) != 1: sys.exit(f"ABORT (file untouched): expected exactly one match for:\n{old}")
new_src = src
for old, new in edits: new_src = new_src.replace(old, new)
try:
    ast.parse(new_src)
except SyntaxError as ex:
    sys.exit(f"ABORT (file untouched): patched source would not parse: {ex}")
shutil.copy2(RE, BAK); shutil.copy2("calibration_enroll.py", Path("common") / "calibration_enroll.py")
RE.write_text(new_src)
print(f"OK: 4 edits applied; backup at {BAK}")
