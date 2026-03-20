#!/bin/bash
# Launch the full data-quality autoresearch pipeline on SLURM.
# Usage: bash slurm/launch.sh [--lang dan_Latn] [--iterations 50]

set -e

LANG="${1:-dan_Latn}"
ITERATIONS="${2:-100}"

echo "=== Data Quality Autoresearch ==="
echo "Language: $LANG"
echo "Iterations: $ITERATIONS"
echo ""

# Step 1: Download data (CPU job, run first)
echo "Step 1: Submitting data download job..."
DL_JOB=$(sbatch --parsable slurm/download.sbatch "$LANG")
echo "  Download job: $DL_JOB"

# Step 2: Start vLLM server (1 GPU, starts after download)
echo "Step 2: Submitting vLLM server job..."
VLLM_JOB=$(sbatch --parsable --dependency=afterok:$DL_JOB slurm/vllm_server.sbatch)
echo "  vLLM job: $VLLM_JOB"

# Step 3: Run the experiment loop (1 GPU, starts after vLLM is up)
echo "Step 3: Submitting experiment loop job..."
LOOP_JOB=$(sbatch --parsable --dependency=afterok:$DL_JOB slurm/run_loop.sbatch "$LANG" "$ITERATIONS" "$VLLM_JOB")
echo "  Loop job: $LOOP_JOB"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/loop_\${LOOP_JOB}.out"
