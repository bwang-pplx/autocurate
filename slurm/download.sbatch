#!/bin/bash
#SBATCH --job-name=dl-fineweb2
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1

# Needs 1 GPU because uv run resolves torch which requires CUDA at import
# The actual download is CPU/network bound

LANG="${1:-dan_Latn}"
mkdir -p logs

echo "$(date) | Downloading fineweb-2 for $LANG on $(hostname)"
uv run prepare_data.py --lang "$LANG"
echo "$(date) | Download complete."
