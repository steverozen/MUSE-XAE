"""Convert a Jiang et al. 2025 SBS catalog to MUSE-XAE input format.

The Jiang catalog has two index columns:
  "Mutation type"  (e.g. C>A)
  "Trinucleotide"  (e.g. ACA)

MUSE-XAE expects a single "Type" column in COSMIC notation (e.g. A[C>A]A).
The script also verifies that all 96 COSMIC types are present after conversion.

Usage (from the MUSE-XAE repo root):
  python scripts/convert_jiang_sbs.py \
      --input  /cwork/sr110/sig_attribution_paper_code/synthetic_data/SBS/ground.truth.syn.catalog.csv \
      --output datasets/jiang_sbs.csv
"""

import argparse
import pandas as pd

COSMIC_ORDER_FILE = "datasets/COSMIC_SBS_GRCh37_3.4.txt"


def convert(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    # Build COSMIC-style Type column from the two Jiang index columns.
    df["Type"] = df.apply(
        lambda r: f"{r['Trinucleotide'][0]}[{r['Mutation type']}]{r['Trinucleotide'][2]}",
        axis=1,
    )
    df = df.drop(columns=["Mutation type", "Trinucleotide"])

    # Move Type to the first column.
    cols = ["Type"] + [c for c in df.columns if c != "Type"]
    df = df[cols]

    # Sanity-check: all 96 COSMIC types must be present.
    cosmic = pd.read_csv(COSMIC_ORDER_FILE, sep="\t")
    expected = set(cosmic["Type"])
    got = set(df["Type"])
    missing = expected - got
    extra = got - expected
    if missing:
        raise ValueError(f"Missing COSMIC types after conversion: {sorted(missing)[:5]} ...")
    if extra:
        raise ValueError(f"Unexpected types after conversion: {sorted(extra)[:5]} ...")

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)-1} samples → {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
