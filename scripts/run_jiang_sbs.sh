#!/bin/bash
#SBATCH --job-name=muse_xae_jiang_sbs
#SBATCH --output=/cwork/sr110/MUSE-XAE/logs/muse_xae_jiang_sbs_%j.out
#SBATCH --error=/cwork/sr110/MUSE-XAE/logs/muse_xae_jiang_sbs_%j.err
#SBATCH --chdir=/cwork/sr110/MUSE-XAE
#SBATCH --partition=scavenger
#SBATCH --account=rozenlab
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --requeue

# Run MUSE-XAE de-novo extraction on the Jiang et al. 2025 SBS96 synthetic benchmark.
#
# The ground-truth catalog has 43 signatures; we scan 25-55 to bracket it.
# MUSE-XAE writes all results under ./Experiments/{--directory}/{--dataset}/De-Novo/
#
# Submit from the repo root:
#   mkdir -p logs
#   sbatch scripts/run_jiang_sbs.sh

set -euo pipefail

echo "Starting MUSE-XAE on node $(hostname) at $(date)"
echo "Repo root: $(pwd)"

pixi run python MUSE-XAE/main.py \
    --dataset   jiang_sbs \
    --directory jiang_2025 \
    --min_sig   25 \
    --max_sig   55 \
    --iter      100 \
    --augmentation 100 \
    --epochs    1000 \
    --n_jobs    24 \
    --batch_size 64 \
    --cosmic_version 3.4

echo "Finished at $(date)"
