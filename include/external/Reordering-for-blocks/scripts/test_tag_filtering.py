#!/usr/bin/env python3
"""
Test: compare tag-filtered jobs_list() calls vs single fetch-all approach.
Verifies same results but measures time difference.
"""
import sys
import time
from sbatchman import jobs_list

# All relevant tags discovered from the directory structure
PERM_TAGS = [
    'SB_amd', 'SB_degree', 'SB_gray', 'SB_rcm', 'SB_metis',
    'SB_rabbit', 'SB_patoh', 'SB_slashburn',
    'GROOT_reorder', 'SPARTA_reorder', 'TCA_reorder', 'random1D', 'random2D',
]

RANDOM_PERM_TAGS = [f'{t}_RANDOM' for t in PERM_TAGS if t not in ('random1D', 'random2D')]

ANALYSIS_TAGS = ['ANALYSIS_NO_REORDER', 'ANALYSIS_ROW', 'ANALYSIS_SYMMETRIC']
RANDOM_ANALYSIS_TAGS = ['ANALYSIS_RANDOM_ROW', 'ANALYSIS_RANDOM_SYMMETRIC']

SPMM_TAGS = [
    'CUSPARSE_SPMM_CSR_NO_REORDER', 'CUSPARSE_SPMM_CSR_ROW', 'CUSPARSE_SPMM_CSR_SYMMETRIC',
    'CUSPARSE_SPMM_BSR_NO_REORDER', 'CUSPARSE_SPMM_BSR_ROW', 'CUSPARSE_SPMM_BSR_SYMMETRIC',
    'FLASHSPARSE_SPMM_NO_REORDER', 'FLASHSPARSE_SPMM_ROW', 'FLASHSPARSE_SPMM_SYMMETRIC',
    'DTC_SPMM_NO_REORDER', 'DTC_SPMM_ROW', 'DTC_SPMM_SYMMETRIC',
    'ASPT_SPMM_NO_REORDER', 'ASPT_SPMM_ROW', 'ASPT_SPMM_SYMMETRIC',
    'SMAT_SPMM_NO_REORDER', 'SMAT_SPMM_ROW', 'SMAT_SPMM_SYMMETRIC',
    'ACCSPMM_SPMM_NO_REORDER', 'ACCSPMM_SPMM_SYMMETRIC',
]

RANDOM_SPMM_TAGS = ['CUSPARSE_SPMM_BSR_SYMMETRIC_RANDOM']

ALL_RELEVANT_TAGS = (
    PERM_TAGS + RANDOM_PERM_TAGS +
    ANALYSIS_TAGS + RANDOM_ANALYSIS_TAGS +
    SPMM_TAGS + RANDOM_SPMM_TAGS
)


def job_key(job):
    """Unique key for a job to compare across approaches."""
    return (getattr(job, 'job_id', None), job.tag)


def fetch_all_approach():
    """Original: fetch everything, then categorize."""
    t0 = time.perf_counter()
    all_jobs = jobs_list(from_archived=True, status=["COMPLETED"], update_jobs=False)
    elapsed = time.perf_counter() - t0

    # Categorize same way as parse_results.py
    categorized = {
        'analysis': [], 'random_analysis': [],
        'ops': [], 'random_ops': [],
        'perms': [], 'random_perms': [],
    }
    perm_set = set(PERM_TAGS)
    random_perm_set = set(RANDOM_PERM_TAGS + [f'{t}_RANDOM' for t in PERM_TAGS if t not in ('random1D', 'random2D')])

    for j in all_jobs:
        tag = j.tag or ""
        if tag.startswith("ANALYSIS_RANDOM_"):
            categorized['random_analysis'].append(j)
        elif tag.startswith("ANALYSIS_"):
            categorized['analysis'].append(j)
        elif "SPMM" in tag and "RANDOM" in tag:
            categorized['random_ops'].append(j)
        elif "SPMM" in tag:
            categorized['ops'].append(j)
        elif tag in random_perm_set:
            categorized['random_perms'].append(j)
        elif tag in perm_set:
            categorized['perms'].append(j)

    return categorized, elapsed


def tag_filtered_approach():
    """New: fetch per-tag, skip irrelevant jobs entirely."""
    t0 = time.perf_counter()
    categorized = {
        'analysis': [], 'random_analysis': [],
        'ops': [], 'random_ops': [],
        'perms': [], 'random_perms': [],
    }

    common_args = dict(from_archived=True, status=["COMPLETED"], update_jobs=False)

    for tag in ANALYSIS_TAGS:
        categorized['analysis'].extend(
            jobs_list(tag=tag, **common_args))

    for tag in RANDOM_ANALYSIS_TAGS:
        categorized['random_analysis'].extend(
            jobs_list(tag=tag, **common_args))

    for tag in SPMM_TAGS:
        categorized['ops'].extend(
            jobs_list(tag=tag, **common_args))

    for tag in RANDOM_SPMM_TAGS:
        categorized['random_ops'].extend(
            jobs_list(tag=tag, **common_args))

    for tag in PERM_TAGS:
        categorized['perms'].extend(
            jobs_list(tag=tag, **common_args))

    for tag in RANDOM_PERM_TAGS:
        categorized['random_perms'].extend(
            jobs_list(tag=tag, **common_args))

    elapsed = time.perf_counter() - t0
    return categorized, elapsed


def compare(cat_a, cat_b):
    """Compare two categorization results."""
    all_match = True
    for key in cat_a:
        set_a = set(job_key(j) for j in cat_a[key])
        set_b = set(job_key(j) for j in cat_b[key])
        if set_a == set_b:
            print(f"  {key}: MATCH ({len(set_a)} jobs)")
        else:
            all_match = False
            only_a = set_a - set_b
            only_b = set_b - set_a
            print(f"  {key}: MISMATCH!")
            print(f"    count A={len(set_a)}, B={len(set_b)}")
            if only_a:
                print(f"    only in A ({len(only_a)}): {list(only_a)[:5]}")
            if only_b:
                print(f"    only in B ({len(only_b)}): {list(only_b)[:5]}")
    return all_match


if __name__ == "__main__":
    print("=" * 60)
    print("Approach 1: Fetch ALL then categorize")
    print("=" * 60)
    cat_all, t_all = fetch_all_approach()
    print(f"Time: {t_all:.1f}s")
    for k, v in cat_all.items():
        print(f"  {k}: {len(v)} jobs")

    print()
    print("=" * 60)
    print("Approach 2: Tag-filtered fetches")
    print("=" * 60)
    cat_filtered, t_filtered = tag_filtered_approach()
    print(f"Time: {t_filtered:.1f}s")
    for k, v in cat_filtered.items():
        print(f"  {k}: {len(v)} jobs")

    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    match = compare(cat_all, cat_filtered)

    print()
    print(f"Speedup: {t_all/t_filtered:.1f}x ({t_all:.1f}s -> {t_filtered:.1f}s)")
    if match:
        print("RESULT: All categories match!")
    else:
        print("RESULT: MISMATCHES found — investigate before switching!")
