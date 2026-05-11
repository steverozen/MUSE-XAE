"""
Assess de-novo SBS96 signature extraction against a synthetic ground truth.

For each per-cancer-type cohort the script:
  1. Reads the extracted signature matrix (96 rows × K cols, 'Type' index in COSMIC notation).
  2. Matches extracted signatures to true active signature profiles using Hungarian matching
     (when --ground-truth-sigs is provided) or greedy best-COSMIC-match (fallback).
  3. Compares matched signatures to the true active set derived from the ground-truth
     exposures file (restricted to the samples that were actually in that cohort's input CSV).
  4. Optionally computes exposure RMSE (when --exposures-subpath is provided).
  5. Writes one row per cohort to a sorted CSV.

Inputs
------
--results-dir       DIR    Top-level directory with per-cancer-type subdirectories.
--sig-subpath       PATH   Path relative to each cancer-type subdir to the signature CSV
                           (e.g. "De-Novo/Suggested_SBS_De_Novo/MUSE_SBS.csv").
                           Leave empty for flat layout: {results_dir}/{cancer_type}.csv
--datasets-dir      DIR    Directory with per-cancer-type input CSVs (identifies cohort
                           samples). Default: datasets/
--ground-truth      FILE   Ground-truth exposures CSV (rows=signatures, cols=samples).
--ground-truth-sigs FILE   ground.truth.sigs.csv (Jiang two-column multi-index).
                           When provided, uses Hungarian matching against true sig profiles
                           instead of greedy COSMIC matching.
--exposures-subpath PATH   Path relative to each cancer-type subdir to an exposures CSV
                           (sample_id index, K sig columns). When provided alongside
                           --ground-truth-sigs, computes exposure RMSE.
--cosmic            FILE   COSMIC SBS96 reference (tab-separated, Type first column).
                           Only used in greedy fallback mode.
--out               FILE   Output CSV path.
--match-threshold   FLOAT  Minimum cosine similarity to count as a match (default 0.0).

Examples
--------
# Hungarian mode (recommended — for tools with true ground-truth sigs available)
python scripts/assess_extraction.py \\
    --results-dir /cwork/sr110/mSigLDA/results/per_cancer_type \\
    --sig-subpath sigs.csv \\
    --ground-truth /path/to/ground.truth.syn.exposures.csv \\
    --ground-truth-sigs /path/to/ground.truth.sigs.csv \\
    --exposures-subpath exposures.csv \\
    --out results/lda_per_ct_assessment.csv

# Greedy COSMIC mode (backward-compatible, for MUSE-XAE)
python scripts/assess_extraction.py \\
    --results-dir Experiments/per_cancer_type \\
    --sig-subpath De-Novo/Suggested_SBS_De_Novo/MUSE_SBS.csv \\
    --ground-truth /path/to/ground.truth.syn.exposures.csv \\
    --out results/muse_xae_per_ct_assessment.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_cosmic(path: str) -> pd.DataFrame:
    """Return (96, N_sigs) DataFrame indexed by Type."""
    return pd.read_csv(path, sep="\t", index_col=0)


def load_sig_file(path: str) -> pd.DataFrame:
    """Return (96, K) DataFrame indexed by Type from an extracted-signature CSV."""
    return pd.read_csv(path, index_col=0)


def load_ground_truth_sigs(path: str) -> pd.DataFrame:
    """Load ground.truth.sigs.csv and return (96, K_truth) DataFrame indexed by COSMIC Type.

    The file has a Jiang two-column multi-index (Mutation type, Trinucleotide).
    Rows are converted to COSMIC notation: f"{tri[0]}[{mut}]{tri[2]}".
    """
    df = pd.read_csv(path, index_col=[0, 1])
    df.index = pd.Index(
        [f"{tri[0]}[{mut}]{tri[2]}" for mut, tri in df.index],
        name="Type",
    )
    return df


def true_active_sigs(gt_exposures: pd.DataFrame, samples: list[str]) -> set:
    """Return signature names with nonzero total exposure across the given samples."""
    available = [s for s in samples if s in gt_exposures.columns]
    if not available:
        return set()
    sub = gt_exposures[available]
    return set(sub.index[sub.sum(axis=1) > 0])


def find_sig_path(results_dir: str, ct_dir: str, sig_subpath: str) -> str:
    if sig_subpath:
        return os.path.join(results_dir, ct_dir, sig_subpath)
    return os.path.join(results_dir, f"{ct_dir}.csv")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_sigs_to_cosmic(
    extracted: pd.DataFrame,
    cosmic: pd.DataFrame,
) -> pd.DataFrame:
    """Greedy best-match of each extracted signature to COSMIC reference.

    Returns DataFrame with columns: extracted_sig, cosmic_sig, cosine_sim.
    """
    common_types = extracted.index.intersection(cosmic.index)
    if len(common_types) < 96:
        print(f"  WARNING: {96 - len(common_types)} COSMIC types absent from extracted file",
              file=sys.stderr)

    ext_mat = extracted.loc[common_types].values.T    # (K, n_types)
    cos_mat = cosmic.loc[common_types].values.T        # (N_cosmic, n_types)
    sims = cosine_similarity(ext_mat, cos_mat)         # (K, N_cosmic)

    rows = []
    for i, sig_name in enumerate(extracted.columns):
        best_idx = int(np.argmax(sims[i]))
        rows.append({
            "extracted_sig": sig_name,
            "cosmic_sig":    cosmic.columns[best_idx],
            "cosine_sim":    float(sims[i, best_idx]),
        })
    return pd.DataFrame(rows)


def match_sigs_hungarian(
    extracted: pd.DataFrame,
    truth_sub: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian optimal matching of extracted signatures against true active sig profiles.

    Args:
        extracted:  (96, K_ext) indexed by COSMIC Type
        truth_sub:  (96, K_active) indexed by COSMIC Type, columns = active sig names

    Returns:
        learned_idx — indices into extracted.columns (length min(K_ext, K_active))
        truth_idx   — indices into truth_sub.columns
        cosines     — matched cosine similarities
    """
    common = extracted.index.intersection(truth_sub.index)
    ext_mat  = extracted.loc[common].values.T    # (K_ext, n_types)
    true_mat = truth_sub.loc[common].values.T    # (K_active, n_types)

    sim_matrix = cosine_similarity(ext_mat, true_mat)   # (K_ext, K_active)
    learned_idx, truth_idx = linear_sum_assignment(-sim_matrix)
    cosines = sim_matrix[learned_idx, truth_idx]
    return learned_idx, truth_idx, cosines


# ---------------------------------------------------------------------------
# Exposure RMSE
# ---------------------------------------------------------------------------

def compute_exposure_rmse(
    learned_exp: pd.DataFrame,
    gt_exp: pd.DataFrame,
    cohort_samples: list[str],
    active_sig_names: list[str],
    learned_idx: np.ndarray,
    truth_idx: np.ndarray,
) -> float:
    """Compute RMSE between learned and ground-truth fractional exposures.

    Args:
        learned_exp:      (N, K_ext) DataFrame, sample_id index, values = doc fractions
        gt_exp:           (K_all, N_all) DataFrame, sig index, sample columns
        cohort_samples:   list of sample IDs in this cohort
        active_sig_names: ordered list of active sig names (truth_idx indexes into this)
        learned_idx:      Hungarian learned→truth assignment (into learned_exp columns)
        truth_idx:        Hungarian truth indices (into active_sig_names)
    """
    K_active = len(active_sig_names)
    available = [s for s in cohort_samples if s in gt_exp.columns]
    N = len(available)
    if N == 0:
        return float("nan")

    # Ground truth fractions (K_active, N)
    gt_sub = gt_exp.loc[active_sig_names, available].values.astype(float)
    col_sums = gt_sub.sum(axis=0, keepdims=True) + 1e-12
    truth_frac = gt_sub / col_sums    # (K_active, N)

    # Learned fractions aligned to truth order (K_active, N), zeros for unmatched
    # learned_exp rows = samples, columns = SIG_1..SIG_K
    learned_sub = learned_exp.loc[available].values    # (N, K_ext)
    aligned = np.zeros((K_active, N), dtype=float)
    for li, ti in zip(learned_idx, truth_idx):
        aligned[ti, :] = learned_sub[:, li]

    return float(np.sqrt(np.mean((aligned - truth_frac) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
                    help="Directory with per-cancer-type input CSVs (identifies cohort samples)")
    ap.add_argument("--ground-truth", required=True,
                    help="Ground-truth exposures CSV (rows=signatures, cols=samples)")
    ap.add_argument("--ground-truth-sigs", default=None,
                    help="ground.truth.sigs.csv (Jiang two-column multi-index). "
                         "Enables Hungarian matching against true sig profiles.")
    ap.add_argument("--exposures-subpath", default=None,
                    help="Path relative to each cancer-type subdir to the exposures CSV. "
                         "Enables exposure RMSE (requires --ground-truth-sigs).")
    ap.add_argument("--cosmic", default="datasets/COSMIC_SBS_GRCh37_3.4.txt",
                    help="COSMIC SBS96 reference TSV (used in greedy fallback mode)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--match-threshold", type=float, default=0.0,
                    help="Min cosine similarity to count as a match (default 0.0)")
    args = ap.parse_args()

    gt_exp = pd.read_csv(args.ground_truth, index_col=0)    # (N_sigs, N_samples)

    # Load once before loop
    truth_sigs_df = None
    if args.ground_truth_sigs:
        truth_sigs_df = load_ground_truth_sigs(args.ground_truth_sigs)

    cosmic = None
    if truth_sigs_df is None:
        cosmic = load_cosmic(args.cosmic)

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

        ds_header = pd.read_csv(dataset_csv, nrows=0)
        cohort_samples = [c for c in ds_header.columns if c != "Type"]

        true_sigs = true_active_sigs(gt_exp, cohort_samples)
        if not true_sigs:
            print(f"  SKIP {ct_dir}: no samples found in ground-truth exposures", file=sys.stderr)
            continue

        extracted = load_sig_file(sig_path)

        # --- Hungarian path ---
        if truth_sigs_df is not None:
            active_sig_names = sorted(true_sigs)
            # Filter to sigs present in the ground-truth sigs file
            available_names = [s for s in active_sig_names if s in truth_sigs_df.columns]
            if len(available_names) < len(active_sig_names):
                missing = set(active_sig_names) - set(available_names)
                print(f"  WARNING {ct_dir}: {missing} not in ground-truth-sigs file",
                      file=sys.stderr)
                active_sig_names = available_names

            truth_sub = truth_sigs_df[active_sig_names]   # (96, K_active)
            learned_idx, truth_idx, cosines = match_sigs_hungarian(extracted, truth_sub)

            above = cosines >= args.match_threshold
            found  = set(np.array(active_sig_names)[truth_idx[above]])
            missed = true_sigs - found
            extra  = set()   # Hungarian assigns each extracted sig to a truth sig
            sims   = cosines[above]

            exposure_rmse = None
            if args.exposures_subpath:
                exp_path = os.path.join(args.results_dir, ct_dir, args.exposures_subpath)
                if os.path.isfile(exp_path):
                    learned_exp = pd.read_csv(exp_path, index_col=0)
                    exposure_rmse = compute_exposure_rmse(
                        learned_exp, gt_exp, cohort_samples,
                        active_sig_names, learned_idx, truth_idx,
                    )
                else:
                    print(f"  WARNING {ct_dir}: exposures file not found at {exp_path}",
                          file=sys.stderr)

        # --- Greedy COSMIC fallback path ---
        else:
            match_df = match_sigs_to_cosmic(extracted, cosmic)
            if args.match_threshold > 0:
                match_df = match_df[match_df["cosine_sim"] >= args.match_threshold]

            matched_cosmic = set(match_df["cosmic_sig"])
            found  = matched_cosmic & true_sigs
            missed = true_sigs - matched_cosmic
            extra  = matched_cosmic - true_sigs
            sims   = match_df["cosine_sim"].values
            exposure_rmse = None

        output_rows.append({
            "cancer_type":   ct_dir,
            "true_K":        len(true_sigs),
            "extracted_K":   len(extracted.columns),
            "n_matched":     len(found),
            "median_cosine": round(float(np.median(sims)), 4) if len(sims) else None,
            "mean_cosine":   round(float(np.mean(sims)),   4) if len(sims) else None,
            "min_cosine":    round(float(np.min(sims)),    4) if len(sims) else None,
            "n_above_0.9":   int((cosines >= 0.9).sum())  if truth_sigs_df is not None else None,
            "n_above_0.95":  int((cosines >= 0.95).sum()) if truth_sigs_df is not None else None,
            "exposure_rmse": round(float(exposure_rmse), 6) if exposure_rmse is not None else None,
            "true_sigs":     " ".join(sorted(true_sigs)),
            "matched_sigs":  " ".join(sorted(found)),
            "missed_sigs":   " ".join(sorted(missed)),
            "extra_sigs":    " ".join(sorted(extra)),
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

    display_cols = ["cancer_type", "true_K", "extracted_K", "n_matched",
                    "median_cosine", "min_cosine", "n_above_0.95", "missed_sigs"]
    # Only show n_above_0.95 if Hungarian path was used
    if truth_sigs_df is None:
        display_cols = [c for c in display_cols if c != "n_above_0.95"]
    print(out_df[display_cols].to_string(index=False))
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
