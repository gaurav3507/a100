"""Adds --cal-mean-only to scripts/run_experiment.py (requires the earlier --n-enroll patch).
Run from V2 root with calibration_enroll.py present. Aborts untouched on mismatch; refuses to double-apply."""
import ast, shutil, sys
from pathlib import Path
RE, BAK = Path("scripts/run_experiment.py"), Path("scripts/run_experiment.py.pre_meanonly.bak")
if not RE.exists() or not Path("calibration_enroll.py").exists(): sys.exit("run from V2 root with calibration_enroll.py present")
src = RE.read_text()
if "cal_mean_only" in src: sys.exit("already patched -- nothing to do (file untouched)")
if "apply_enrollment_baseline" not in src: sys.exit("ABORT: apply patch_enroll_exact.py first")
edits = [
 ('            df, _enroll_mask = apply_enrollment_baseline(df, flat_features, n_enroll=args.n_enroll)\n',
  '            df, _enroll_mask = apply_enrollment_baseline(df, flat_features, n_enroll=args.n_enroll, mean_only=args.cal_mean_only)\n'),
 ('    parser.add_argument("--n-enroll", type=int, default=0,\n',
  '    parser.add_argument("--cal-mean-only", action="store_true",\n'
  '                        help="enrollment calibration subtracts the enrollment mean only (no std division)")\n'
  '    parser.add_argument("--n-enroll", type=int, default=0,\n'),
 ('           + (f"_enroll{args.n_enroll}" if args.n_enroll else "")\n',
  '           + (f"_enroll{args.n_enroll}" if args.n_enroll else "")\n'
  '           + ("_mean" if getattr(args, "cal_mean_only", False) else "")\n'),
]
for old, new in edits:
    if src.count(old) != 1: sys.exit(f"ABORT (file untouched): expected exactly one match for:\n{old}")
new_src = src
for old, new in edits: new_src = new_src.replace(old, new)
try: ast.parse(new_src)
except SyntaxError as ex: sys.exit(f"ABORT (file untouched): {ex}")
shutil.copy2(RE, BAK); shutil.copy2("calibration_enroll.py", Path("common") / "calibration_enroll.py")
RE.write_text(new_src); print(f"OK: 3 edits applied; backup at {BAK}")
