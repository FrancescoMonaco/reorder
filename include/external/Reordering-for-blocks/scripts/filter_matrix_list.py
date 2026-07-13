#!/usr/bin/env python3
"""
Pre-filter matrices_list_mtx.txt based on filter_config.yaml.

Reads .mtx file headers (without loading full matrices) to extract
dimensions and nnz, then applies the same filters used in analysis/plotting.
Writes a filtered list that can be used by sbatchman launch configs.

Usage:
    python scripts/filter_matrix_list.py
    python scripts/filter_matrix_list.py --input datasets/matrices_list_mtx.txt datasets/large-matrices/matrices_list.txt
    python scripts/filter_matrix_list.py --config scripts/filter_config.yaml
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import yaml


def read_mtx_header(path):
    """Read .mtx header to extract (rows, cols, nnz) without loading the matrix."""
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('%') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[1]), int(parts[2])
                break
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
    return None


def get_family(mtx_path):
    """Extract the SuiteSparse family from a matrix path."""
    parts = str(mtx_path).replace('\\', '/').split('/')
    if len(parts) >= 3:
        return parts[-3]
    return None


def get_matrix_name(mtx_path):
    """Extract the matrix name (stem) from a path."""
    return Path(mtx_path).stem


def main():
    parser = argparse.ArgumentParser(description="Pre-filter matrix list for sbatchman launch")
    parser.add_argument('--config', default='scripts/filter_config.yaml',
                        help='Path to filter_config.yaml')
    parser.add_argument('--input', nargs='*', default=None,
                        help='Input matrix list file(s). Auto-discovers datasets/*/matrices_list.txt if not specified.')
    parser.add_argument('--output', default='datasets/matrices_list_filtered.txt',
                        help='Output filtered matrix list')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    filters = config.get('filters', {})

    # Collect input lists: explicit args, or auto-discover per-category matrices_list.txt
    if args.input:
        input_files = args.input
    else:
        datasets_dir = Path('datasets')
        # Each mtxman category has datasets/<category>/matrices_list.txt
        input_files = sorted(str(p) for p in datasets_dir.glob('*/matrices_list.txt'))
        if not input_files:
            input_files = [config.get('data', {}).get('matrices_list', 'datasets/matrices_list_mtx.txt')]

    # Read and combine all input lists
    all_paths = []
    for input_path in input_files:
        with open(input_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        print(f"  {input_path}: {len(paths)} matrices")
        all_paths.extend(paths)

    print(f"Combined input: {len(all_paths)} matrices from {len(input_files)} file(s)")

    # Read headers for all matrices
    matrices = []
    for mtx_path in all_paths:
        header = read_mtx_header(mtx_path)
        if header is None:
            print(f"  Skipping (unreadable): {mtx_path}", file=sys.stderr)
            continue
        rows, cols, nnz = header
        matrices.append({
            'path': mtx_path,
            'rows': rows,
            'cols': cols,
            'nnz': nnz,
            'name': get_matrix_name(mtx_path),
            'family': get_family(mtx_path),
        })

    print(f"Readable: {len(matrices)} matrices")
    kept = matrices[:]

    # Filter: square_only
    if filters.get('square_only', False):
        before = len(kept)
        kept = [m for m in kept if m['rows'] == m['cols']]
        print(f"Square only: {before} -> {len(kept)}")

    # Filter: min_size
    min_size = filters.get('min_size')
    if min_size is not None and min_size > 0:
        before = len(kept)
        kept = [m for m in kept if m['rows'] >= min_size]
        print(f"Min size (>={min_size}): {before} -> {len(kept)}")

    # Filter: max_size
    max_size = filters.get('max_size')
    if max_size is not None:
        before = len(kept)
        kept = [m for m in kept if m['rows'] <= max_size and m['cols'] <= max_size]
        print(f"Max size (<={max_size}): {before} -> {len(kept)}")

    # Filter: filter_diagonal
    if filters.get('filter_diagonal', False):
        before = len(kept)
        kept = [m for m in kept if not (m['rows'] == m['cols'] and m['nnz'] == m['rows'])]
        print(f"Remove diagonal: {before} -> {len(kept)}")

    # Filter: max_sparsity_factor (nnz < factor * rows means too sparse)
    sparsity_factor = filters.get('max_sparsity_factor')
    if sparsity_factor is not None:
        before = len(kept)
        kept = [m for m in kept if m['nnz'] >= sparsity_factor * m['rows']]
        print(f"Sparsity filter (nnz >= {sparsity_factor}*N): {before} -> {len(kept)}")

    # Filter: one_per_family
    if filters.get('one_per_family', False):
        keep_full = set(filters.get('keep_full_families', []) or [])
        before = len(kept)

        # Group by family
        families = defaultdict(list)
        for m in kept:
            families[m['family']].append(m)

        selected = []
        for family, members in sorted(families.items()):
            if family in keep_full:
                selected.extend(members)
            else:
                # Pick the first one alphabetically by name
                members.sort(key=lambda m: m['name'])
                selected.append(members[0])

        kept = selected
        print(f"One per family: {before} -> {len(kept)}")

    # Deduplicate by matrix name (in case same matrix appears from multiple categories)
    before = len(kept)
    seen_names = set()
    deduped = []
    for m in kept:
        if m['name'] not in seen_names:
            seen_names.add(m['name'])
            deduped.append(m)
    kept = deduped
    if before != len(kept):
        print(f"Deduplicate: {before} -> {len(kept)}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for m in kept:
            f.write(m['path'] + '\n')

    print(f"\nOutput: {len(kept)} matrices written to {args.output}")


if __name__ == '__main__':
    main()
