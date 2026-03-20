#!/bin/bash
#SBATCH --job-name=dl-fineweb2
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu

LANG="${1:-dan_Latn}"
mkdir -p logs

echo "Downloading fineweb-2 for $LANG..."
uv run prepare_data.py --lang "$LANG"
echo "Download complete."
