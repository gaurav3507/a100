"""Add --num-workers / --prefetch to scripts/run_experiment.py (WheelsEye_V2).

The two DataLoader dicts default to num_workers=0, so batches are built on the main
process and the GPU starves on the sequence models. This patch adds worker options,
defaulting to 0 so behaviour is UNCHANGED unless --num-workers is passed.

Run from the V2 root:  python patch_dataloader_v1.py
Aborts untouched on any mismatch; refuses to double-apply.
"""
import ast, shutil, sys
from pathlib import Path

RE  = Path("scripts/run_experiment.py")
BAK = Path("scripts/run_experiment.py.pre_workers.bak")

if not RE.exists():
    sys.exit("run from the WheelsEye_V2 root (scripts/run_experiment.py not found)")
src = RE.read_text()
if "num_workers" in src:
    sys.exit("already patched -- nothing to do (file untouched)")

WORKER_KW = ("num_workers=args.num_workers,\n"
             "                    pin_memory=(args.num_workers > 0),\n"
             "                    persistent_workers=(args.num_workers > 0),\n"
             "                    prefetch_factor=(args.prefetch if args.num_workers > 0 else None)")

edits = [
 ("    loaders = dict(batch_size=args.batch_size, collate_fn=window_collate)\n",
  f"    loaders = dict(batch_size=args.batch_size, collate_fn=window_collate,\n                    {WORKER_KW})\n"),
 ("    loaders = dict(batch_size=args.batch_size, collate_fn=sequence_collate)\n",
  f"    loaders = dict(batch_size=args.batch_size, collate_fn=sequence_collate,\n                    {WORKER_KW})\n"),
 ('    parser.add_argument("--seed", type=int, default=0)\n',
  '    parser.add_argument("--seed", type=int, default=0)\n'
  '    parser.add_argument("--num-workers", type=int, default=0,\n'
  '                        help="DataLoader worker processes; 0 keeps the original single-process behaviour")\n'
  '    parser.add_argument("--prefetch", type=int, default=4,\n'
  '                        help="batches prefetched per worker (ignored when --num-workers 0)")\n'),
]
for old, _ in edits:
    if src.count(old) != 1:
        sys.exit(f"ABORT (file untouched): expected exactly one match for:\n{old!r}")

new = src
for old, rep in edits:
    new = new.replace(old, rep, 1)
try:
    ast.parse(new)
except SyntaxError as ex:
    sys.exit(f"ABORT (file untouched): patched source would not parse: {ex}")

shutil.copy2(RE, BAK)
RE.write_text(new)
print(f"OK: 3 edits applied; backup at {BAK}")
print("default is unchanged (num_workers=0); pass --num-workers 8 to enable parallel loading")
