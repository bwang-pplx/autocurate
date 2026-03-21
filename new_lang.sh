#!/bin/bash
# Set up a new language for data-quality autoresearch.
# Usage: bash new_lang.sh deu_Latn [ITERATIONS]
#
# Creates filter_{lang_code}.py and launches the SLURM pipeline.

set -e

LANG="${1:?Usage: bash new_lang.sh <LANG> [ITERATIONS]}"
ITERATIONS="${2:-100}"
LANG_CODE=$(echo "$LANG" | cut -d'_' -f1)
FILTER_FILE="filter_${LANG_CODE}.py"

if [ -f "$FILTER_FILE" ]; then
    echo "$FILTER_FILE already exists."
else
    echo "Creating $FILTER_FILE..."
    cat > "$FILTER_FILE" << 'PYEOF'
"""
Data cleaning and filtering pipeline for LANG_PLACEHOLDER.
This file grows automatically — peek.py appends new rules here.
DO NOT EDIT MANUALLY.
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
    if len(text.strip()) < 200:
        return False
    for fn in FILTERS:
        if not fn(text):
            return False
    return True
PYEOF
    sed -i "s/LANG_PLACEHOLDER/$LANG/" "$FILTER_FILE"
    echo "Created $FILTER_FILE"
fi

echo "Launching pipeline for $LANG..."
bash slurm/launch.sh "$LANG" "$ITERATIONS"
