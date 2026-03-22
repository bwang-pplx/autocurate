#!/bin/bash
#SBATCH --job-name=dl-fineweb2
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --qos=search
# Needs 1 GPU because uv run resolves torch which requires CUDA at import
# The actual download is CPU/network bound

LANG="${1:-dan_Latn}"
mkdir -p logs

echo "$(date) | Node: $(hostname)"
echo "$(date) | Installing dependencies (uv sync)..."
uv sync 2>&1
echo "$(date) | Dependencies ready."
echo "$(date) | Downloading fineweb-2 for $LANG (capped at 10M docs)..."
PYTHONUNBUFFERED=1 uv run prepare.py --lang "$LANG" --max-docs 10000000 2>&1
echo "$(date) | Download complete."
