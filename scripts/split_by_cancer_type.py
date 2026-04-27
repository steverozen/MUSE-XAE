"""Split a multi-cancer-type Jiang-format catalog into per-cancer-type MUSE-XAE CSVs.

For each cancer type:
  - Writes a main cohort CSV (MSI/POLE-suspect samples excluded)
  - Writes a hypermutator CSV if any MSI/POLE-suspect samples are found

MSI/POLE-suspect samples are identified by ONE or more of:
  1. --msi-pole-samples  : explicit text file listing sample IDs (one per line)
  2. --burden-threshold  : mutation burden (total counts) above this percentile
                           within the cancer type is flagged as hypermutator

Cancer type is parsed from the column prefix before the first '::'.

Output CSVs are written to --out-dir and named:
  <cancer_type>.csv
  <cancer_type>_hypermutator.csv   (only if hypermutators found)

Output format is MUSE-XAE-ready (single 'Type' column in COSMIC notation,
rows reordered to COSMIC order).

Usage (from MUSE-XAE repo root):
  python scripts/split_by_cancer_type.py \\
      --catalog /path/to/catalog.csv \\
      --cosmic  datasets/COSMIC_SBS_GRCh37_3.4.txt \\
      --out-dir datasets/ \\
      [--msi-pole-samples /path/to/hyper_samples.txt] \\
      [--burden-threshold 95]
"""

import argparse
import os
import pandas as pd


def to_cosmic_type(mut_type: str, trinucleotide: str) -> str:
    return f"{trinucleotide[0]}[{mut_type}]{trinucleotide[2]}"


def load_catalog(path: str, cosmic_index: pd.Index) -> pd.DataFrame:
    """Read a Jiang catalog and return a COSMIC-ordered DataFrame (Type x Sample)."""
    df = pd.read_csv(path)
    df["Type"] = df.apply(
        lambda r: to_cosmic_type(r["Mutation type"], r["Trinucleotide"]), axis=1
    )
    df = df.drop(columns=["Mutation type", "Trinucleotide"]).set_index("Type")
    missing = set(cosmic_index) - set(df.index)
    if missing:
        raise ValueError(f"Catalog missing COSMIC types: {sorted(missing)[:5]} ...")
    return df.loc[cosmic_index]   # (96, N), COSMIC order


def write_muse_csv(df: pd.DataFrame, path: str) -> None:
    """Write a COSMIC-ordered (96, N) DataFrame as a MUSE-XAE input CSV."""
    out = df.reset_index()   # Type column + sample columns
    out.to_csv(path, index=False)
    print(f"  wrote {out.shape[1]-1:4d} samples → {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="Jiang-format multi-cancer catalog CSV")
    ap.add_argument("--cosmic", default="datasets/COSMIC_SBS_GRCh37_3.4.txt",
                    help="COSMIC reference file (tab-separated, Type column)")
    ap.add_argument("--out-dir", default="datasets/", help="Directory for output CSVs")
    ap.add_argument("--msi-pole-samples",
                    help="Text file with one sample ID per line to treat as hypermutators")
    ap.add_argument("--burden-threshold", type=float, default=None,
                    help="Flag samples above this mutation-burden percentile (per cancer type) "
                         "as hypermutators, e.g. 95")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cosmic = pd.read_csv(args.cosmic, sep="\t")
    cosmic_index = pd.Index(cosmic["Type"])

    print(f"Loading catalog: {args.catalog}")
    catalog = load_catalog(args.catalog, cosmic_index)   # (96, N)
    all_samples = list(catalog.columns)
    print(f"  {len(all_samples)} samples, {len(cosmic_index)} mutation types")

    # Build explicit hypermutator set from file
    explicit_hyper: set = set()
    if args.msi_pole_samples:
        with open(args.msi_pole_samples) as fh:
            explicit_hyper = {line.strip() for line in fh if line.strip()}
        print(f"  {len(explicit_hyper)} explicit hypermutator samples loaded")

    # Detect cancer types from column prefix before '::'
    cancer_types = {}
    for s in all_samples:
        ct = s.split("::")[0] if "::" in s else "unknown"
        cancer_types.setdefault(ct, []).append(s)

    print(f"\n{'Cancer type':<25} {'N':>4} {'hyper':>6} {'main':>6}  output")
    for ct, samples in sorted(cancer_types.items()):
        sub = catalog[samples]   # (96, n_ct)

        # Burden-based hypermutator detection (within this cancer type only)
        burden_hyper: set = set()
        if args.burden_threshold is not None:
            burdens = sub.sum(axis=0)
            threshold = burdens.quantile(args.burden_threshold / 100)
            burden_hyper = set(burdens[burdens > threshold].index)

        hyper_samples = sorted((explicit_hyper | burden_hyper) & set(samples))
        main_samples  = [s for s in samples if s not in hyper_samples]

        safe_ct = ct.replace(" ", "_").replace("/", "-")

        # Main cohort
        if main_samples:
            write_muse_csv(sub[main_samples],
                           os.path.join(args.out_dir, f"{safe_ct}.csv"))

        # Hypermutator cohort
        if hyper_samples:
            write_muse_csv(sub[hyper_samples],
                           os.path.join(args.out_dir, f"{safe_ct}_hypermutator.csv"))

        print(f"  {ct:<23} {len(samples):>4} {len(hyper_samples):>6} {len(main_samples):>6}")

    print("\nDone.")


if __name__ == "__main__":
    main()
