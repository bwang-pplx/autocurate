#!/bin/bash
# Launch the full data-quality autoresearch pipeline on SLURM.
# Usage: bash slurm/launch.sh [LANG] [ITERATIONS]
#
# Example:
#   bash slurm/launch.sh dan_Latn 100

set -e

LANG="${1:-dan_Latn}"
ITERATIONS="${2:-100}"

echo "=== Data Quality Autoresearch ==="
echo "Language: $LANG"
echo "Iterations: $ITERATIONS"
echo ""

mkdir -p logs

# Step 1: Download data (1 GPU for torch import, mostly CPU/network)
echo "Submitting download job..."
DL_JOB=$(sbatch --parsable slurm/download.sbatch "$LANG")
echo "  Download job: $DL_JOB"

# Step 2: Run the loop (2 GPUs: vLLM + training, starts after download)
echo "Submitting experiment loop job..."
LOOP_JOB=$(sbatch --parsable --dependency=afterok:$DL_JOB slurm/run_loop.sbatch "$LANG" "$ITERATIONS")
echo "  Loop job: $LOOP_JOB"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/download_${DL_JOB}.out   # download progress"
echo "  tail -f logs/loop_${LOOP_JOB}.out      # experiment loop"
echo "  cat results.tsv                         # results table"
