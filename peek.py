"""
Peek at the data: sample documents multiple times, accumulate quality
observations, then synthesize into one fix and apply it.

Usage:
    python peek.py --lang dan_Latn --iteration 1
    python peek.py --lang dan_Latn --iteration 5 --from-filtered
    python peek.py --lang dan_Latn --peeks 20           # more peeks for larger languages
    python peek.py --lang dan_Latn --dry-run             # don't apply, just print

Requires a running vLLM server:
    vllm serve Qwen/Qwen3.5-35B-A3B --tensor-parallel-size 1
"""

import os
import re
import sys
import json
import argparse
import random
import time

import pyarrow.parquet as pq

from prepare_data import get_lang_dir, list_raw_parquet_files

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"
VLLM_URL = "http://localhost:8000/v1/chat/completions"

LANG_NAMES = {
    "dan_Latn": "Danish", "deu_Latn": "German", "fra_Latn": "French",
    "kor_Hang": "Korean", "tur_Latn": "Turkish", "vie_Latn": "Vietnamese",
    "zho_Hans": "Chinese (Simplified)", "spa_Latn": "Spanish",
    "rus_Cyrl": "Russian", "jpn_Jpan": "Japanese",
}

# ---------------------------------------------------------------------------
# Phase 1 prompt: observe quality problems in a sample
# ---------------------------------------------------------------------------

OBSERVE_PROMPT = """You are a data quality researcher analyzing web-crawled text data.

Below are {n} randomly sampled documents from a {lang} web corpus (fineweb-2).
The data has already been through basic quality filtering (dedup, language detection,
boilerplate removal). Your job is to find REMAINING quality problems.

{strategy_instruction}

List every quality problem you observe. For each problem:
- Describe it specifically
- Quote a short example from the documents
- Estimate how many of the {n} documents have this problem (e.g. "8 out of {n}")

Be thorough. List ALL problems, even minor ones. We will aggregate across
multiple samples to find the most common patterns.

{focus_instruction}

Here are the documents:

{documents}
"""

# ---------------------------------------------------------------------------
# Phase 2 prompt: synthesize observations into one fix
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = """You are a data quality researcher. You have observed quality problems
across {n_peeks} random samples of {lang} web text (fineweb-2).

Here are all the observations, collected from different samples:

{all_observations}

Now:
1. **RANK** the problems by how frequently they appeared across samples.
2. **PICK** the single most impactful problem — the one that affects the most documents.
3. **WRITE** exactly ONE Python function to fix it. It must be either:
   - A **cleaner** named `clean_<description>(text: str) -> str` that modifies text
   - A **filter** named `filter_<description>(text: str) -> bool` returning True to keep
   Allowed imports: re, string, unicodedata, collections, html, hashlib, difflib, json, urllib, textwrap, math.
   NO ML, NO network calls, NO numpy/pandas/torch. Pure heuristic only.

IMPORTANT: Output ONLY in this exact format:

## Problem
<one paragraph describing the most common problem>

## Frequency
<how often it appeared across the {n_peeks} samples>

## Type
<either "cleaner" or "filter">

## Function
```python
def <name>(text):
    <implementation>
```

## Expected Impact
<what % of docs affected>
"""

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def get_peek_config(iteration):
    """
    Early iterations: many peeks, many docs per peek — find big patterns.
    Later iterations: fewer peeks, fewer docs — subtler issues.

    Returns (n_peeks, docs_per_peek)
    """
    if iteration is None:
        return 10, 50
    if iteration <= 3:
        return 10, 100   # 10 peeks × 100 docs = 1000 docs seen
    elif iteration <= 8:
        return 8, 50      # 8 × 50 = 400 docs seen
    elif iteration <= 15:
        return 5, 30      # 5 × 30 = 150 docs seen
    else:
        return 3, 20      # 3 × 20 = 60 docs seen, carefully

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_documents(lang, n, from_filtered=False, seed=None):
    """Sample n documents using reservoir sampling."""
    lang_dir = get_lang_dir(lang)

    if from_filtered:
        selected_path = os.path.join(lang_dir, "selected_doc_ids.json")
        if not os.path.exists(selected_path):
            print("No filtered data found. Run filter.py first, or omit --from-filtered.")
            return []
        with open(selected_path) as f:
            selected_ids = set(json.load(f))
    else:
        selected_ids = None

    parquet_files = list_raw_parquet_files(lang)
    if not parquet_files:
        print("No raw data found. Run prepare_data.py --phase download first.")
        return []

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    for filepath in parquet_files:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            ids = rg.column("id").to_pylist()
            urls = rg.column("url").to_pylist()

            for doc_id, text, url in zip(ids, texts, urls):
                if selected_ids is not None and doc_id not in selected_ids:
                    continue
                seen += 1
                if len(reservoir) < n:
                    reservoir.append({"doc_id": doc_id, "text": text, "url": url})
                else:
                    j = rng.randint(0, seen - 1)
                    if j < n:
                        reservoir[j] = {"doc_id": doc_id, "text": text, "url": url}

    return reservoir


def format_documents(docs, max_chars_per_doc=3000):
    """Format documents for the prompt."""
    parts = []
    for i, doc in enumerate(docs):
        text = doc["text"][:max_chars_per_doc]
        url = doc.get("url", "")
        parts.append(f"--- Document {i+1} (url: {url}) ---\n{text}\n")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Query Qwen
# ---------------------------------------------------------------------------

def query_qwen(prompt):
    """Query Qwen via vLLM OpenAI-compatible API."""
    import requests

    response = requests.post(
        VLLM_URL,
        json={
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.7,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------
# Parse and apply fix
# ---------------------------------------------------------------------------

def parse_response(response):
    """Parse Qwen's synthesized response into structured parts."""
    result = {}

    m = re.search(r'## Problem\s*\n(.*?)(?=\n## )', response, re.DOTALL)
    result["problem"] = m.group(1).strip() if m else ""

    m = re.search(r'## Type\s*\n(.*?)(?=\n## )', response, re.DOTALL)
    fix_type = m.group(1).strip().lower() if m else ""
    result["type"] = "cleaner" if "clean" in fix_type else "filter"

    m = re.search(r'## Function\s*\n```python\s*\n(.*?)```', response, re.DOTALL)
    result["code"] = m.group(1).strip() if m else ""

    m = re.search(r'## Expected Impact\s*\n(.*?)$', response, re.DOTALL)
    result["impact"] = m.group(1).strip() if m else ""

    return result


def extract_function_name(code):
    """Extract the function name from a def statement."""
    m = re.search(r'def\s+(\w+)\s*\(', code)
    return m.group(1) if m else None


def validate_code(code):
    """Reject code that uses ML, network, or non-stdlib dependencies.
    Returns (ok, reason)."""
    import ast as _ast

    # Must parse
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Only these imports are allowed (stdlib text processing tools)
    allowed_modules = {
        "re",              # regex
        "string",          # string constants (punctuation, digits, etc.)
        "unicodedata",     # unicode categories, normalization
        "collections",     # Counter, defaultdict
        "functools",       # lru_cache etc.
        "itertools",       # groupby etc.
        "math",            # basic math
        "html",            # unescape HTML entities
        "textwrap",        # text wrapping/dedent
        "difflib",         # sequence matching (detect near-duplicate paragraphs)
        "hashlib",         # hashing (dedup lines/paragraphs)
        "json",            # detect/parse JSON blobs in text
        "urllib",          # parse URLs found in text
    }

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed_modules:
                    return False, f"Banned import: {alias.name}. Only {allowed_modules} allowed."
        if isinstance(node, _ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root not in allowed_modules:
                return False, f"Banned import: from {node.module}. Only {allowed_modules} allowed."

    return True, ""


def apply_fix_to_filter(lang, parsed, iteration=None):
    """Append the proposed fix to filter_{lang_code}.py."""
    lang_code = lang.split("_")[0]
    filter_path = os.path.join(os.path.dirname(__file__), f"filter_{lang_code}.py")

    if not os.path.exists(filter_path):
        print(f"Filter file not found: {filter_path}")
        return False

    func_name = extract_function_name(parsed["code"])
    if not func_name:
        print("Could not extract function name from proposed code.")
        return False

    with open(filter_path, "r") as f:
        content = f.read()

    if func_name in content:
        print(f"Function {func_name} already exists in {filter_path}, skipping.")
        return False

    iter_label = f"iteration {iteration}" if iteration else "peek.py"
    comment = f"# Added by {iter_label}: {parsed['problem'][:80]}"

    if parsed["type"] == "cleaner":
        marker = "# --- new cleaners are appended above this line by peek.py ---"
        register_line = f"CLEANERS.append({func_name})"
    else:
        marker = "# --- new filters are appended above this line by peek.py ---"
        register_line = f"FILTERS.append({func_name})"

    insertion = f"\n{comment}\n{parsed['code']}\n{register_line}\n\n"

    if marker not in content:
        print(f"Marker not found in {filter_path}. Cannot insert.")
        return False

    content = content.replace(marker, insertion + marker)

    with open(filter_path, "w") as f:
        f.write(content)

    print(f"Applied {parsed['type']} '{func_name}' to {filter_path}")
    return True

# ---------------------------------------------------------------------------
# Verify and rollback
# ---------------------------------------------------------------------------

def verify_fix(lang, all_docs):
    """Run the updated filter on all sampled docs to check for crashes.
    Returns (ok, error_message)."""
    import importlib
    import traceback

    lang_code = lang.split("_")[0]
    module_name = f"filter_{lang_code}"

    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        lang_mod = importlib.import_module(module_name)
    except Exception as e:
        return False, f"Import error: {e}\n{traceback.format_exc()}"

    for i, doc in enumerate(all_docs):
        try:
            cleaned = lang_mod.clean(doc["text"])
            lang_mod.should_keep(cleaned)
        except Exception as e:
            return False, f"Crashed on doc {i} ({doc['doc_id']}): {e}\n{traceback.format_exc()}"

    return True, ""


def rollback_fix(lang, parsed):
    """Remove the last applied fix from filter_{lang_code}.py."""
    lang_code = lang.split("_")[0]
    filter_path = os.path.join(os.path.dirname(__file__), f"filter_{lang_code}.py")

    func_name = extract_function_name(parsed["code"])
    if not func_name or not os.path.exists(filter_path):
        return

    with open(filter_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    skip = False
    for line in lines:
        if f"# Added by" in line and parsed["problem"][:40] in line:
            skip = True
            continue
        if skip:
            if line.startswith("def ") and func_name in line:
                continue
            if line.strip().startswith(("CLEANERS.append", "FILTERS.append")) and func_name in line:
                skip = False
                continue
            if line.startswith((" ", "\t")) or line.strip() == "":
                continue
            else:
                skip = False
        new_lines.append(line)

    with open(filter_path, "w") as f:
        f.writelines(new_lines)

    print(f"Rolled back {func_name} from {filter_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peek at data quality with Qwen")
    parser.add_argument("--lang", type=str, default="dan_Latn")
    parser.add_argument("--peeks", type=int, default=None,
                        help="Number of peek rounds (default: auto based on iteration)")
    parser.add_argument("--docs-per-peek", type=int, default=None,
                        help="Docs per peek round (default: auto based on iteration)")
    parser.add_argument("--from-filtered", action="store_true")
    parser.add_argument("--focus", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iteration", type=int, default=None)
    args = parser.parse_args()

    # Determine peek schedule
    default_peeks, default_docs = get_peek_config(args.iteration)
    n_peeks = args.peeks or default_peeks
    docs_per_peek = args.docs_per_peek or default_docs
    base_seed = args.seed if args.seed is not None else int(time.time())
    lang_name = LANG_NAMES.get(args.lang, args.lang)

    print(f"Iteration {args.iteration or '?'}: {n_peeks} peeks × {docs_per_peek} docs = {n_peeks * docs_per_peek} docs total")

    # Strategy instruction
    iteration = args.iteration or 0
    if iteration <= 3:
        strategy = ("Focus on STATISTICAL patterns — problems that appear in many documents. "
                     "Look for repeated boilerplate, common junk patterns, widespread quality issues.")
    elif iteration <= 8:
        strategy = ("The most obvious junk has likely been cleaned. Look for MEDIUM-frequency "
                     "patterns: domain-specific issues, structural problems, content type issues.")
    elif iteration <= 15:
        strategy = ("Focus on SUBTLER quality issues: content that passed basic filters but "
                     "is still low-quality or non-educational.")
    else:
        strategy = ("The easy problems are solved. Look for FINE-GRAINED issues: tone, "
                     "coherence, information density, educational value.")

    focus_instruction = f"Focus your analysis specifically on: {args.focus}" if args.focus else ""

    # ---------------------------------------------------------------
    # Phase 1: Observe — peek multiple times, accumulate observations
    # ---------------------------------------------------------------

    all_observations = []
    all_docs = []

    for peek_idx in range(n_peeks):
        seed = base_seed + peek_idx
        docs = sample_documents(args.lang, n=docs_per_peek,
                                from_filtered=args.from_filtered, seed=seed)
        if not docs:
            print(f"Peek {peek_idx+1}: no docs sampled, skipping")
            continue

        all_docs.extend(docs)

        prompt = OBSERVE_PROMPT.format(
            n=len(docs),
            lang=lang_name,
            documents=format_documents(docs),
            strategy_instruction=strategy,
            focus_instruction=focus_instruction,
        )

        print(f"Peek {peek_idx+1}/{n_peeks} ({len(docs)} docs, seed={seed})... ", end="", flush=True)
        t0 = time.time()
        observation = query_qwen(prompt)
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")

        all_observations.append(f"=== Sample {peek_idx+1} ({len(docs)} docs) ===\n{observation}")

    if not all_observations:
        print("No observations collected.")
        exit(1)

    print(f"\nCollected {len(all_observations)} observations from {len(all_docs)} total docs")

    # ---------------------------------------------------------------
    # Phase 2: Synthesize — merge observations into one fix
    # ---------------------------------------------------------------

    print(f"\nSynthesizing fix from {n_peeks} observations...")
    synth_prompt = SYNTHESIZE_PROMPT.format(
        n_peeks=n_peeks,
        lang=lang_name,
        all_observations="\n\n".join(all_observations),
    )

    t0 = time.time()
    response = query_qwen(synth_prompt)
    elapsed = time.time() - t0
    print(f"Synthesis received in {elapsed:.1f}s\n")

    print("=" * 70)
    print(response)
    print("=" * 70)

    # Parse
    parsed = parse_response(response)
    if not parsed["code"]:
        print("\nERROR: Could not parse a function from Qwen's response.")
        exit(1)

    print(f"\nParsed: type={parsed['type']}, function={extract_function_name(parsed['code'])}")
    print(f"Problem: {parsed['problem'][:120]}")

    # Apply
    if args.dry_run:
        print("\n[dry-run] Would apply this fix but --dry-run was set.")
        exit(0)

    # Validate: must be pure heuristic, no ML/network/non-stdlib
    ok, reason = validate_code(parsed["code"])
    if not ok:
        print(f"Code validation FAILED: {reason}")
        print("Fix rejected — must use only re/string/unicodedata/collections.")
        exit(1)

    ok = apply_fix_to_filter(args.lang, parsed, iteration=args.iteration)
    if not ok:
        print("Fix was NOT applied.")
        exit(1)

    # Verify on ALL docs from ALL peeks
    print(f"\nVerifying fix on {len(all_docs)} docs...")
    ok, err = verify_fix(args.lang, all_docs)
    if ok:
        print("Verification passed.")
    else:
        print(f"Verification FAILED: {err}")
        print("Rolling back fix...")
        rollback_fix(args.lang, parsed)
        exit(1)
