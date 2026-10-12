#!/usr/bin/env python3
"""Unpack the SPRINkLE dataset (Zenodo 20282035) and convert every .npz into
Matrix Market (.mtx), mirroring the layout of the SuiteSparse matrices that
already live in /data/matrices/datasets/, then emit ai_matrices.txt.

The SPRINkLE archive stores matrices as .npz; this repo consumes .mtx via
scipy.io.mmread (matrix_features.py, CLUB/DTC reorder code, ...). This script
detects the npz layout automatically (scipy sparse save_npz, COO CSR-style
arrays, or a single dense array) and writes coordinate .mtx files.

Destination layout (confirmed with the maintainer):
    /data/matrices/datasets/<name>/<name>.mtx      # <name> = npz path, '/' -> '__'
    /data/matrices/datasets/ai_matrices.txt        # sorted absolute .mtx paths

Usage (reorder_env):
    micromamba run -n reorder_env python scripts/sprinkle_to_mtx.py \
        --staging /data/matrices/_staging_SPRINkLE
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse


def load_npz_any(npz_path: Path) -> scipy.sparse.spmatrix:
    """Load an .npz into a scipy sparse matrix, auto-detecting the layout."""
    # 1) scipy.sparse.save_npz format
    try:
        return scipy.sparse.load_npz(str(npz_path))
    except Exception:
        pass

    # 2) Inspect the raw arrays and reconstruct.
    with np.load(str(npz_path), allow_pickle=False) as z:
        keys = set(z.files)

        # csr/csc/bsr save_npz keys
        if {"data", "indices", "indptr", "shape"} <= keys:
            shape = tuple(int(x) for x in z["shape"])
            data, indices, indptr = z["data"], z["indices"], z["indptr"]
            fmt = str(z["format"]) if "format" in keys else "csr"
            if fmt == "csc":
                return scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
            if fmt == "coo":
                return scipy.sparse.coo_matrix((data, indices), shape=shape)
            return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

        # 3) COO given as row/col/data(+shape)
        if {"row", "col", "data"} <= keys:
            shape = tuple(int(x) for x in z["shape"]) if "shape" in keys else None
            return scipy.sparse.coo_matrix((z["data"], (z["row"], z["col"])), shape=shape)

        # 4) Single dense array (any rank). Conv weights are stored as N-D
        #    tensors, e.g. (out, in, kh, kw); flatten every axis after the
        #    first so the result is a 2D matrix (out, in*kh*kw). Linear layers
        #    are already 2D and pass through unchanged.
        for k in sorted(keys):
            arr = z[k]
            if arr.ndim >= 2:
                mat = arr.reshape(arr.shape[0], -1)
                # scipy.sparse has no float16; upcast so mmwrite can serialize it.
                if mat.dtype == np.float16:
                    mat = mat.astype(np.float32)
                return scipy.sparse.coo_matrix(mat)

    raise ValueError(f"Unrecognized npz layout in {npz_path} (keys={sorted(keys)})")


def npz_name_from_relpath(rel: str) -> str:
    """Derive a unique matrix stem from the npz path inside the archive."""
    p = Path(rel)
    return "__".join(list(p.with_suffix("").parts))



def main() -> int:
    ap = argparse.ArgumentParser(description="Convert SPRINkLE npz -> mtx and build ai_matrices.txt.")
    ap.add_argument("--staging", type=Path, default=Path("/data/matrices/_staging_SPRINkLE"))
    ap.add_argument("--zip", type=Path, default=None, help="path to SPRINkLE.zip (default: <staging>/SPRINkLE.zip)")
    ap.add_argument("--dest", type=Path, default=Path("/data/matrices/datasets"))
    ap.add_argument("--list", type=Path, default=None, help="output list (default: <dest>/ai_matrices.txt)")
    ap.add_argument("--limit", type=int, default=0, help="convert only first N npz (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="inspect + list, do not write .mtx")
    a = ap.parse_args()

    zip_path = a.zip or (a.staging / "SPRINkLE.zip")
    list_path = a.list or (a.dest / "ai_matrices.txt")
    if not zip_path.exists():
        raise SystemExit(f"zip not found: {zip_path}")

    # ---- inventory the archive ----
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".npz") and not n.endswith("/")]
    names.sort()
    print(f"found {len(names)} npz files in {zip_path}")
    if not names:
        raise SystemExit("no .npz entries found")

    tops = sorted({n.split('/')[0] for n in names})
    print(f"top-level entries ({len(tops)}): {tops[:10]}{' ...' if len(tops) > 10 else ''}")
    for n in names[:5]:
        print(f"  sample npz: {n}")

    targets = names[: a.limit] if a.limit else names

    written: list[Path] = []
    ok = err = 0
    with zipfile.ZipFile(zip_path) as zf:
        for i, rel in enumerate(targets, 1):
            stem = npz_name_from_relpath(rel)
            out_mtx = a.dest / stem / f"{stem}.mtx"
            try:
                if a.dry_run:
                    print(f"[dry {i}/{len(targets)}] {rel} -> {out_mtx}")
                    written.append(out_mtx)
                    ok += 1
                    continue
                out_mtx.parent.mkdir(parents=True, exist_ok=True)
                tmp_npz = out_mtx.parent / (stem + ".npz")
                with zf.open(rel) as src, open(tmp_npz, "wb") as dst:
                    dst.write(src.read())
                M = load_npz_any(tmp_npz)
                A = M.tocoo()
                A.eliminate_zeros()
                A.sum_duplicates()
                scipy.io.mmwrite(str(out_mtx), A)
                tmp_npz.unlink(missing_ok=True)
                written.append(out_mtx)
                ok += 1
                if i % 10 == 0 or i == len(targets):
                    print(f"[{i}/{len(targets)}] ok={ok} err={err} last={stem} shape={A.shape} nnz={A.nnz}", flush=True)
            except Exception as e:
                err += 1
                print(f"[ERR {i}/{len(targets)}] {rel}: {type(e).__name__}: {e}", flush=True)

    written_sorted = sorted({str(p) for p in written})
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(list_path, "w") as f:
        for p in written_sorted:
            f.write(p + "\n")
    print(f"wrote {list_path} ({len(written_sorted)} entries)")
    print(f"DONE ok={ok} err={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
