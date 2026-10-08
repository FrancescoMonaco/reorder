#!/usr/bin/env bash
#
# fetch_datasets.sh — download the .mtx matrices named in
# datasets/matrices_list_filtered.txt (or given as arguments) from the
# SuiteSparse Matrix Collection into datasets/<name>/<name>.mtx.
#
# Usage:
#   ./fetch_datasets.sh [options] [name ...]
#
# With no names, every entry of datasets/matrices_list_filtered.txt is
# fetched. Absolute paths, plain names, blank lines, and '#' comments are
# all accepted — only the file stem is used.
#
# Options:
#   -g LIST     comma-separated SuiteSparse group(s) to try, in order
#               (default: HB; override with SS_GROUPS env var)
#   -f          re-download even if the .mtx already exists
#   -h          show this help
#
# Requirements: curl, tar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS="$ROOT/datasets"
LIST_FILE="$DATASETS/matrices_list_filtered.txt"
BASE_URL="${SS_BASE_URL:-https://suitesparse-collection-website.herokuapp.com/MM}"
SS_GROUP_LIST="${SS_GROUPS:-HB}"
FORCE=0

usage() {
    sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" | grep -v '^set -euo pipefail$' | sed 's/^# \{0,1\}//'
}

while getopts ":g:fh" opt; do
    case "$opt" in
        g) SS_GROUP_LIST="$OPTARG" ;;
        f) FORCE=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: '$1' is required but not installed." >&2
        exit 1
    }
}
need curl
need tar

names=()
if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        names+=("$(basename "$arg" .mtx)")
    done
else
    [ -f "$LIST_FILE" ] || {
        echo "Error: list file not found: $LIST_FILE" >&2
        exit 1
    }
    while IFS= read -r line || [ -n "$line" ]; do
        line="$(printf '%s' "$line" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [ -z "$line" ] && continue
        case "$line" in \#*) continue ;; esac
        names+=("$(basename "$line" .mtx)")
    done <"$LIST_FILE"
fi
[ "${#names[@]}" -gt 0 ] || {
    echo "Error: no matrix names given and list file is empty." >&2
    exit 1
}

mkdir -p "$DATASETS"
ok=0
skipped=0
failed=0
for name in "${names[@]}"; do
    dest="$DATASETS/$name/$name.mtx"
    if [ -f "$dest" ] && [ "$FORCE" -eq 0 ]; then
        echo "skip  $name (already exists, use -f to re-download)"
        skipped=$((skipped + 1))
        continue
    fi
    tmp="$(mktemp -d)"
    got=0
    used_group=""
    IFS=',' read -ra groups <<<"$SS_GROUP_LIST"
    for group in "${groups[@]}"; do
        if curl -fsSL --retry 2 --max-time 300 \
            -o "$tmp/$name.tar.gz" "$BASE_URL/$group/$name.tar.gz"; then
            got=1
            used_group="$group"
            break
        fi
    done
    if [ "$got" -eq 0 ]; then
        echo "FAIL  $name (not found in groups: $SS_GROUP_LIST)"
        failed=$((failed + 1))
        rm -rf "$tmp"
        continue
    fi
    mtx_path="$(tar tzf "$tmp/$name.tar.gz" | grep -m1 '\.mtx$' || true)"
    if [ -z "$mtx_path" ]; then
        echo "FAIL  $name (tarball contains no .mtx)"
        failed=$((failed + 1))
        rm -rf "$tmp"
        continue
    fi
    mkdir -p "$DATASETS/$name"
    tar xzf "$tmp/$name.tar.gz" -C "$tmp" "$mtx_path"
    mv "$tmp/$mtx_path" "$dest"
    rm -rf "$tmp"
    echo "ok    $name (group $used_group)"
    ok=$((ok + 1))
done

echo "done: $ok fetched, $skipped skipped, $failed failed"
[ "$failed" -eq 0 ]
