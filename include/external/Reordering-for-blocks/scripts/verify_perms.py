#!/usr/bin/env python3
"""Verify that all .perm files contain valid permutations.

A valid permutation line of size N must contain exactly the integers 1..N
with no duplicates and no missing values.
"""

import os
import sys
from pathlib import Path
from collections import Counter


def verify_perm_line(values, filename, line_idx):
    """Check that a list of ints is a valid 1-based permutation. Returns list of error strings."""
    errors = []
    n = len(values)
    if n == 0:
        errors.append(f"  line {line_idx}: empty")
        return errors

    min_val, max_val = min(values), max(values)
    if min_val < 1:
        errors.append(f"  line {line_idx}: min value {min_val} < 1")
    if max_val > n:
        errors.append(f"  line {line_idx}: max value {max_val} > n={n}")
    if max_val != n or min_val != 1:
        errors.append(f"  line {line_idx}: range [{min_val}, {max_val}] != [1, {n}]")

    counts = Counter(values)
    dupes = {v: c for v, c in counts.items() if c > 1}
    if dupes:
        top5 = sorted(dupes.items(), key=lambda x: -x[1])[:5]
        errors.append(f"  line {line_idx}: {len(dupes)} duplicate value(s), e.g. {top5}")

    unique = set(values)
    if len(unique) != n:
        missing_count = n - len(unique)
        errors.append(f"  line {line_idx}: {missing_count} missing value(s) out of {n}")

    return errors


def verify_file(filepath):
    """Verify a single .perm file. Returns (ok, errors_list)."""
    errors = []
    try:
        with open(filepath, 'r') as f:
            raw = f.read().strip()
    except Exception as e:
        return False, [f"  read error: {e}"]

    if not raw:
        return False, ["  file is empty"]

    lines = raw.split('\n')
    if len(lines) not in (1, 2):
        errors.append(f"  unexpected number of lines: {len(lines)} (expected 1 or 2)")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            errors.append(f"  line {i}: blank")
            continue
        try:
            values = list(map(int, line.split()))
        except ValueError as e:
            errors.append(f"  line {i}: non-integer values ({e})")
            continue
        errors.extend(verify_perm_line(values, filepath, i))

    return len(errors) == 0, errors


def main():
    root = Path(__file__).resolve().parent.parent
    perm_dirs = [root / "perms", root / "perms_random"]

    total = 0
    ok_count = 0
    bad_files = []

    for perm_dir in perm_dirs:
        if not perm_dir.exists():
            continue
        for filepath in sorted(perm_dir.rglob("*.perm")):
            total += 1
            ok, errors = verify_file(filepath)
            if ok:
                ok_count += 1
            else:
                rel = filepath.relative_to(root)
                bad_files.append((rel, errors))

    # Summary
    print(f"Scanned {total} .perm files")
    print(f"  Valid:   {ok_count}")
    print(f"  Invalid: {len(bad_files)}")

    if bad_files:
        print(f"\n{'='*60}")
        print("INVALID FILES:")
        print(f"{'='*60}")
        for rel, errors in bad_files:
            print(f"\n{rel}")
            for e in errors:
                print(e)
        sys.exit(1)
    else:
        print("\nAll permutation files are valid.")


if __name__ == "__main__":
    main()
