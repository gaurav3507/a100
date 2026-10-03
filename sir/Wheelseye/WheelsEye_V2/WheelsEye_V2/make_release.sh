#!/usr/bin/env bash
# make_release.sh -- build the clean WheelsEye repository for public release.
#
# Run from the WheelsEye_V2 root on the DGX, with WheelsEye_repo_package.zip
# placed in the same directory. Produces ../WheelsEye_release/ and
# ../WheelsEye_release.zip, verified, ready for manual upload to GitHub.
#
# Includes:  V2 source (common/, scripts/*.py, track_a_heavy/, track_b_light/, tests/),
#            the analysis package, result JSONs (the evidence, ~2 MB), feature lists,
#            the controls results, the paper source.
# Excludes:  data (parquet/h5/tar.gz), CSV dumps, .bak files, "(1)" duplicates,
#            root-level copies of analysis scripts, virtualenvs, logs.
set -euo pipefail

SRC="$(pwd)"
REL="$(dirname "$SRC")/WheelsEye_release"
PKG="$SRC/WheelsEye_repo_package.zip"

[ -f "$PKG" ] || { echo "ERROR: put WheelsEye_repo_package.zip in $SRC first"; exit 1; }
[ -d "$SRC/scripts" ] && [ -d "$SRC/common" ] || { echo "ERROR: run from the WheelsEye_V2 root"; exit 1; }

echo "== building release in $REL"
rm -rf "$REL"; mkdir -p "$REL"

# 1. the analysis package (README, REPRODUCE, analysis/, tools/, docs/, paper/, results/)
tmp=$(mktemp -d); unzip -q "$PKG" -d "$tmp"; cp -r "$tmp"/WheelsEye/. "$REL"/; rm -rf "$tmp"

# 2. V2 source -- copy only tracked-worthy files
for d in common track_a_heavy track_b_light tests; do
  if [ -d "$SRC/$d" ]; then
    mkdir -p "$REL/$d"
    ( cd "$SRC/$d" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' ! -name '*.bak' -exec cp --parents {} "$REL/$d/" \; )
  fi
done
mkdir -p "$REL/scripts"
find "$SRC/scripts" -maxdepth 1 -name '*.py' ! -name '*.bak' -exec cp {} "$REL/scripts/" \;

# 3. evidence: result JSONs (small) + the controls run + feature lists
mkdir -p "$REL/results/json" "$REL/results/controls"
find "$SRC/data/processed/results" -maxdepth 1 -name '*.json' -exec cp {} "$REL/results/json/" \;
[ -f "$SRC/data/processed/results/multiseed_table_rows.tex" ] && cp "$SRC/data/processed/results/multiseed_table_rows.tex" "$REL/results/"
for f in results.json features_used.txt controls_snippets.tex; do
  [ -f "$SRC/revision_results_v2exact/$f" ] && cp "$SRC/revision_results_v2exact/$f" "$REL/results/controls/"
done
for f in v2_feature_names.txt v2_drift_names.txt enrollment_curve.csv; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$REL/results/"
done

# 4. gitignore: allow the result JSONs (they ARE the evidence), keep data out
cat > "$REL/.gitignore" <<'EOF'
data/
features_v2/
*.parquet
*.h5
*.tar.gz
*.csv
!results/enrollment_curve.csv
*.pt
*.pth
*.onnx
*.bak
*.log
.venv/
venv/
__pycache__/
*.pyc
paper/*.aux
paper/*.out
paper/*.bbl
paper/*.blg
paper/*.synctex.gz
EOF

# 5. sanity: nothing that should never ship
echo "== forbidden-file scan"
bad=$(find "$REL" -type f \( -name '*.bak' -o -name '*(1)*' -o -name '*(2)*' -o -name '*(3)*' \
      -o -name '*.parquet' -o -name '*.h5' -o -name '*.tar.gz' -o -name 'predictions.csv' -o -name 'v2_windows.csv' \) | wc -l)
[ "$bad" -eq 0 ] && echo "   clean" || { echo "   FOUND $bad forbidden files:"; find "$REL" -type f \( -name '*.bak' -o -name '*(1)*' -o -name '*.parquet' -o -name '*.h5' -o -name '*.tar.gz' -o -name 'predictions.csv' -o -name 'v2_windows.csv' \); exit 1; }

# 6. verify structure and numbers
echo "== verify_repo.py (structure)"
( cd "$REL" && python tools/verify_repo.py ) || { echo "STRUCTURE CHECK FAILED"; exit 1; }
echo "== verify_repo.py (numbers vs paper)"
( cd "$REL" && python tools/verify_repo.py --results results/json | tail -n 12 )

# 7. size and zip
echo "== size"; du -sh "$REL"; find "$REL" -type f | wc -l | xargs echo "   files:"
( cd "$(dirname "$REL")" && rm -f WheelsEye_release.zip && zip -qr WheelsEye_release.zip WheelsEye_release )
echo "== done: $(dirname "$REL")/WheelsEye_release.zip"
echo "   upload the CONTENTS of WheelsEye_release/ to the repository root (drag-and-drop on GitHub, or git push with a token)."
