"""
Data cleaning and filtering pipeline for Danish (dan_Latn).
This file grows automatically — peek.py appends new rules here.
DO NOT EDIT MANUALLY.

Each rule is a function added by Qwen after analyzing a sample of documents.
Rules are applied in order: first all cleaners, then all filters.
"""

import re

# ===================================================================
# CLEANERS — modify text, applied in order
# ===================================================================

CLEANERS = []

# --- new cleaners are appended above this line by peek.py ---

# ===================================================================
# FILTERS — return False to drop a document, applied after cleaning
# ===================================================================

FILTERS = []

# --- new filters are appended above this line by peek.py ---

# ===================================================================
# Pipeline entry points (called by filter.py)
# ===================================================================

def clean(text):
    """Apply all cleaners in order."""
    for fn in CLEANERS:
        text = fn(text)
    return text

def should_keep(text):
    """Apply all filters. Drop if any returns False."""
    if len(text.strip()) < 200:
        return False
    for fn in FILTERS:
        if not fn(text):
            return False
    return True
