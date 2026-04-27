"""
Assess de-novo SBS96 signature extraction against a synthetic ground truth.

For each per-cancer-type cohort the script:
  1. Reads the extracted signature matrix (96 rows × K cols, 'Type' index in COSMIC notation).
  2. Matches every extracted signature to its best-cosine COSMIC reference signature.
  3. Compares the matched COSMIC set to the true active signatures derived from the
     ground-truth exposures file (restricted to the samples that were actually in
     that cohort's input CSV).
  4. Writes one row per cohort to a sorted CSV.

Inputs
------
--results-dir   DIR     Top-level directory with per-cancer-type subdirectories.
--sig-subpath   PATH    Path relative to each cancer-type subdir to the signature
                        CSV (e.g. "De-Novo/Suggested_SBS_De_Novo/MUSE_SBS.csv").
                        For a flat layout where files live directly in results-dir
                        as {cancer_type}.csv, pass an empty string "".
--datasets-dir  DIR     Directory containing the per-cancer-type input CSVs that were
                        fed to the tool (default: datasets/).  Used to identify which
                        samples belong to each cohort.
--ground-truth  FILE    Ground-truth exposures CSV: rows = signatures, cols = samples.
--cosmic        FILE    COSMIC reference TSV (tab-separated, 'Type' first column).
--out           FILE    Output CSV path.
--match-threshold FLOAT Minimum cosine similarity for a COSMIC match to count
                        (default 0.0 — report all, let the table speak).

Example (MUSE-XAE)
-------------------
python scripts/assess_extraction.py \\
    --results-dir Experiments/per_cancer_type \\
    --sig-subpath De-Novo/Suggested_SBS_De_Novo/MUSE_SBS.csv \\
    --ground-truth /cwork/sr110/sig_attribution_paper_code/synthetic_data/SBS_7200/ground.truth.syn.exposures.csv \\
    --out results/muse_xae_per_ct_assessment.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_cosmic(path: str) -> pd.DataFrame:
    """Return (96, N_sigs) DataFrame indexed by Type."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df


def load_sig_file(path: str) -> pd.DataFrame:
    """Return (96, K) DataFrame indexed by Type from an extracted-signature CSV."""
    df = pd.read_csv(path, index_col=0)
    return df


def match_sigs_to_cosmic(
    extracted: pd.DataFrame,
    cosmic: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    For every extracted signature find the best-matching COSMIC signature.

    Returns a DataFrame with columns:
        extracted_sig, cosmic_sig, cosine_sim
    """
    common_types = extracted.index.intersection(cosmic.index)
    if len(common_types) < 96:
        missing = 96 - len(common_types)
        print(f"  WARNING: {missing} COSMIC types absent from extracted file", file=sys.stderr)

    ext_mat = extracted.loc[common_types].values.T   # (K, n_types)
    cos_mat = cosmic.loc[common_types].values.T       # (N_cosmic, n_types)

    sims = cosine_similarity(ext_mat, cos_mat)        # (K, N_cosmic)

    rows = []
    for i, sig_name in enumerate(extracted.columns):
        best_idx = int(np.argmax(sims[i]))
        rows.append({
            "extracted_sig": sig_name,
            "cosmic_sig":    cosmic.columns[best_idx],
            "cosine_sim":    float(sims[i, best_idx]),
        })
    return pd.DataFrame(rows)


def true_active_sigs(gt_exposures: pd.DataFrame, samples: list[str]) -> set:
    """Return signatures with nonzero total exposure across the given samples."""
    available = [s for s in samples if s in gt_exposures.columns]
    if not available:
        return set()
    sub = gt_exposures[available]
    return set(sub.index[sub.sum(axis=1) > 0])


def find_sig_path(results_dir: str, ct_dir: str, sig_subpath: str) -> str:
    if sig_subpath:
        return os.path.join(results_dir, ct_dir, sig_subpath)
    # flat layout: file is {results_dir}/{ct_dir}.csv
    return os.path.join(results_dir, f"{ct_dir}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assess de-novo SBS extraction against synthetic ground truth."
    )
    ap.add_argument("--results-dir", required=True,
                    help="Directory with per-cancer-type output subdirectories")
    ap.add_argument("--sig-subpath", default="",
                    help="Relative path within each cancer-type subdir to the signature CSV "
                         "(leave empty for flat layout: {results_dir}/{cancer_type}.csv)")
    ap.add_argument("--datasets-dir", default="datasets/",
                    help="Directory with the per-cancer-type input CSVs (identifies cohort samples)")
    ap.add_argument("--ground-truth", required=True,
                    help="Ground-truth exposures CSV (rows=signatures, cols=samples)")
    ap.add_argument("--cosmic", default="datasets/COSMIC_SBS_GRCh37_3.4.txt",
                    help="COSMIC SBS96 reference (tab-separated, Type first column)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--match-threshold", type=float, default=0.0,
                    help="Min cosine similarity to count as a match (default 0.0)")
    args = ap.parse_args()

    cosmic    = load_cosmic(args.cosmic)
    gt_exp    = pd.read_csv(args.ground_truth, index_col=0)   # (N_sigs, N_samples)

    output_rows = []

    ct_dirs = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    )

    for ct_dir in ct_dirs:
        sig_path    = find_sig_path(args.results_dir, ct_dir, args.sig_subpath)
        dataset_csv = os.path.join(args.datasets_dir, f"{ct_dir}.csv")

        if not os.path.isfile(sig_path):
            print(f"  SKIP {ct_dir}: signature file not found at {sig_path}", file=sys.stderr)
            continue
        if not os.path.isfile(dataset_csv):
            print(f"  SKIP {ct_dir}: dataset CSV not found at {dataset_csv}", file=sys.stderr)
            continue

        # Samples that were in this cohort
        ds_header = pd.read_csv(dataset_csv, nrows=0)
        cohort_samples = [c for c in ds_header.columns if c != "Type"]

        true_sigs  = true_active_sigs(gt_exp, cohort_samples)
        if not true_sigs:
            print(f"  SKIP {ct_dir}: no samples found in ground-truth exposures", file=sys.stderr)
            continue

        extracted  = load_sig_file(sig_path)
        match_df   = match_sigs_to_cosmic(extracted, cosmic, args.match_threshold)

        # Apply threshold
        if args.match_threshold > 0:
            match_df = match_df[match_df["cosine_sim"] >= args.match_threshold]

        matched_cosmic = set(match_df["cosmic_sig"])
        found  = matched_cosmic & true_sigs
        missed = true_sigs - matched_cosmic
        extra  = matched_cosmic - true_sigs

        sims = match_df["cosine_sim"].values

        output_rows.append({
            "cancer_type":    ct_dir,
            "true_K":         len(true_sigs),
            "extracted_K":    len(extracted.columns),
            "n_matched":      len(found),
            "median_cosine":  round(float(np.median(sims)), 4) if len(sims) else None,
            "min_cosine":     round(float(np.min(sims)),    4) if len(sims) else None,
            "true_sigs":      " ".join(sorted(true_sigs)),
            "matched_sigs":   " ".join(sorted(found)),
            "missed_sigs":    " ".join(sorted(missed)),
            "extra_sigs":     " ".join(sorted(extra)),
        })

    if not output_rows:
        print("No cohorts assessed — check --results-dir and --sig-subpath.", file=sys.stderr)
        sys.exit(1)

    out_df = (
        pd.DataFrame(output_rows)
        .sort_values("cancer_type")
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # Print a readable summary to stdout
    display_cols = ["cancer_type", "true_K", "extracted_K", "n_matched",
                    "median_cosine", "min_cosine", "missed_sigs"]
    print(out_df[display_cols].to_string(index=False))
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
