#!/usr/bin/env bash
#
# Download every matrix listed in results_analysis.csv.bak using ssgetpy.
#
#   bash download_matrices.sh                 # 8 parallel downloads into /data/matrices
#   bash download_matrices.sh -j 16
#   bash download_matrices.sh -d /data/matrices -c /path/to/results.csv
#
# ssgetpy fetches each matrix and extracts it to <dest>/<name>/<name>.mtx;
# matrices that are already there are skipped by ssgetpy itself, so the script
# can simply be re-run to continue. Runs in the micromamba env reorder_env and
# works from fish.

set -uo pipefail

here="$(cd -- "$(dirname -- "$0")" && pwd)"
CSV="$here/../results/results_analysis.csv.bak"
DEST="/data/matrices"
JOBS=8
ENV_NAME="reorder_env"

while getopts ":c:d:j:e:h" opt; do
  case "$opt" in
    c) CSV="$OPTARG" ;;
    d) DEST="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    e) ENV_NAME="$OPTARG" ;;
    h) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: -$OPTARG" >&2; exit 2 ;;
  esac
done

MAMBA="${MAMBA_BIN:-$(command -v micromamba || true)}"
if [[ -z "$MAMBA" && -x "$HOME/micromamba/micromamba" ]]; then
  MAMBA="$HOME/micromamba/micromamba"
fi
[[ -n "$MAMBA" ]] || { echo "micromamba not found (set MAMBA_BIN)" >&2; exit 1; }

PY="$("$MAMBA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')" || exit 1

LIST="$(mktemp)"
OUT="$(mktemp)"
trap 'rm -f "$LIST" "$OUT"' EXIT

# unique matrix names of the results CSV, biggest matrices first
"$PY" -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
names = {r["matrix"].removesuffix(".mtx"): int(r["nnz"]) for r in rows}
print("\n".join(n for n, _ in sorted(names.items(), key=lambda kv: -kv[1])))
' "$CSV" > "$LIST" || exit 1
TOTAL=$(wc -l < "$LIST")

# one ssgetpy download per matrix, run in parallel; ssgetpy skips matrices that
# are already downloaded, the worker prints one "ok <group>/<name>" line each
WORKER='
import sys
import ssgetpy, ssgetpy.matrix

class _Bar:                                # silent stand-in for tqdm, so that
    def __init__(self, *a, **k): pass      # parallel jobs do not interleave bars
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def update(self, *a, **k): pass
    def close(self): pass

ssgetpy.matrix.tqdm = _Bar

dest, name = sys.argv[1], sys.argv[2]
for m in ssgetpy.search(name=name, limit=100000):
    if m.name == name:
        m.download(destpath=dest, extract=True)
        print("ok   " + m.group + "/" + name, flush=True)
        break
else:
    print("MISS " + name + ": not in the SuiteSparse index", file=sys.stderr, flush=True)
    sys.exit(1)
'

echo "==> $TOTAL matrices from $CSV"
echo "==> downloading into $DEST, $JOBS parallel ssgetpy jobs in env $ENV_NAME"

xargs -d '\n' -n 1 -P "$JOBS" "$PY" -c "$WORKER" "$DEST" < "$LIST" 2>&1 | tee "$OUT"
rc=${PIPESTATUS[0]}

ok=$(grep -c '^ok ' "$OUT")
miss=$(grep -c '^MISS' "$OUT")
echo "==> downloaded $ok/$TOTAL matrices ($miss not found, rc=$rc)"
[[ "$ok" -eq "$TOTAL" ]] || echo "==> run the script again to retry the missing ones"
