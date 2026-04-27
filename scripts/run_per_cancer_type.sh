#!/bin/bash
#SBATCH --job-name=muse_xae_per_ct
#SBATCH --output=/cwork/sr110/MUSE-XAE/logs/muse_xae_%A_%a.out
#SBATCH --error=/cwork/sr110/MUSE-XAE/logs/muse_xae_%A_%a.err
#SBATCH --chdir=/cwork/sr110/MUSE-XAE
#SBATCH --partition=common
#SBATCH --account=rozenlab
#SBATCH --cpus-per-task=96
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Per-cancer-type MUSE-XAE de-novo extraction.
# One array task per cancer-type CSV; all tasks run in parallel on separate nodes.
# Each task runs the full K sweep internally (MUSE-XAE selects best K).
#
# Main cohorts:        --min_sig 3  --max_sig 20  (expected K 8-16)
# Hypermutator cohorts: --min_sig 2  --max_sig 10
# augmentation=50 with 800 samples gives 40,000 training vectors per fit,
# keeping per-fit time to ~28 min; 1800 fits / 96 CPUs ~ 9 hours.
#
# Prepare datasets first:
#   python scripts/split_by_cancer_type.py \
#       --catalog /path/to/catalog.csv \
#       [--burden-threshold 95] \
#       [--msi-pole-samples /path/to/hyper_samples.txt]
#
# Then submit:
#   mapfile -t DATASETS < <(ls datasets/*.csv | grep -E '(Breast|ColoRect|Eso|Kidney|Liver|Lung|Ovary|Skin|Stomach)' | sort)
#   sbatch --array=0-$((${#DATASETS[@]}-1)) scripts/run_per_cancer_type.sh

set -euo pipefail

mapfile -t DATASETS < <(ls datasets/*.csv | grep -E '(Breast|ColoRect|Eso|Kidney|Liver|Lung|Ovary|Skin|Stomach)' | sort)

DATASET_PATH="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_NAME=$(basename "$DATASET_PATH" .csv)

echo "Array task ${SLURM_ARRAY_TASK_ID}  dataset: ${DATASET_NAME}"
echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}  Started: $(date)"

if [[ "$DATASET_NAME" == *_hypermutator ]]; then
    MIN_SIG=2
    MAX_SIG=10
else
    MIN_SIG=3
    MAX_SIG=20
fi

pixi run python MUSE-XAE/main.py \
    --dataset      "$DATASET_NAME" \
    --directory    per_cancer_type \
    --min_sig      "$MIN_SIG" \
    --max_sig      "$MAX_SIG" \
    --iter         100 \
    --augmentation 50 \
    --epochs       1000 \
    --n_jobs       "${SLURM_CPUS_PER_TASK}" \
    --batch_size   64 \
    --cosmic_version 3.4

echo "Finished: $(date)"
