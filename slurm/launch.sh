#!/bin/bash
# Launch the autocurate pipeline on SLURM for a language.
# Usage: bash slurm/launch.sh <LANG> [ITERATIONS]
#
# Example:
#   bash slurm/launch.sh dan_Latn 100
#   bash slurm/launch.sh swe_Latn 50

set -e

LANG="${1:?Usage: bash slurm/launch.sh <LANG> [ITERATIONS]}"
ITERATIONS="${2:-100}"
LANG_CODE=$(echo "$LANG" | cut -d'_' -f1)
FILTER_FILE="filter_${LANG_CODE}.py"

echo "=== Autocurate ==="
echo "Language: $LANG"
echo "Iterations: $ITERATIONS"
echo ""

mkdir -p logs

# Create filter file if it doesn't exist
if [ ! -f "$FILTER_FILE" ]; then
    echo "Creating $FILTER_FILE..."
    cat > "$FILTER_FILE" << PYEOF
"""
Data cleaning and filtering pipeline for ${LANG}.
This file grows automatically — peek.py appends new rules here.
"""

import re

CLEANERS = []

# --- new cleaners are appended above this line by peek.py ---

FILTERS = []

# --- new filters are appended above this line by peek.py ---

def clean(text):
    for fn in CLEANERS:
        text = fn(text)
    return text

def should_keep(text):
    for fn in FILTERS:
        if not fn(text):
            return False
    return True
PYEOF
    echo "Created $FILTER_FILE"
fi

# Step 1: Download data
echo "Submitting download job..."
DL_JOB=$(sbatch --parsable slurm/download.sbatch "$LANG")
echo "  Download job: $DL_JOB"

# Step 2: Run the loop (starts after download)
echo "Submitting experiment loop job..."
LOOP_JOB=$(sbatch --parsable --dependency=afterok:$DL_JOB slurm/run_loop.sbatch "$LANG" "$ITERATIONS")
echo "  Loop job: $LOOP_JOB"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/download_${DL_JOB}.out"
echo "  tail -f logs/loop_${LOOP_JOB}.out"
echo "  cat results_${LANG_CODE}.tsv"
