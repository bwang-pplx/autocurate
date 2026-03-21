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

from prepare import get_lang_dir, list_raw_parquet_files

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.5-35B-A3B")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")

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

SYNTHESIZE_PROMPT = """You observed quality problems across {n_peeks} samples of {lang} web text.

Observations:
{all_observations}

Pick the #1 most frequent problem and fix it using one of these templates:

{template_menu}

RULES:
- Do NOT explain your reasoning. Just pick a template and provide parameters.
- Do NOT write regex or code. Just pick a template and list the strings/numbers.

Output EXACTLY this format and nothing else:

## Problem
One sentence describing the problem.

## Template
TEMPLATE_NAME

## Params
```json
{{"param_name": ["value1", "value2"]}}
```

## Expected Impact
X% of docs
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

def sample_documents_multi(lang, n_batches, n_per_batch, from_filtered=False, base_seed=0):
    """Sample multiple batches by picking random files and random row groups.
    Fast — reads only what's needed, not the entire dataset."""
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
        print("No raw data found. Run prepare.py --phase download first.")
        return []

    rng = random.Random(base_seed)
    total_needed = n_batches * n_per_batch
    all_docs = []
    t0 = time.time()

    # Pick random files, read random row groups until we have enough
    file_indices = list(range(len(parquet_files)))
    rng.shuffle(file_indices)

    for fi in file_indices:
        filepath = parquet_files[fi]
        pf = pq.ParquetFile(filepath)
        rg_indices = list(range(pf.num_row_groups))
        rng.shuffle(rg_indices)

        for rg_idx in rg_indices:
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            ids = rg.column("id").to_pylist()
            urls = rg.column("url").to_pylist()

            for doc_id, text, url in zip(ids, texts, urls):
                if selected_ids is not None and doc_id not in selected_ids:
                    continue
                all_docs.append({"doc_id": doc_id, "text": text, "url": url})

            print(f"\r  Sampling: {len(all_docs):,} docs collected ({total_needed:,} needed), {time.time()-t0:.0f}s", end="", flush=True)

            if len(all_docs) >= total_needed * 3:
                break
        if len(all_docs) >= total_needed * 3:
            break

    # Shuffle and split into batches
    rng.shuffle(all_docs)
    batches = []
    for b in range(n_batches):
        start = b * n_per_batch
        end = start + n_per_batch
        batches.append(all_docs[start:end])

    elapsed = time.time() - t0
    print(f"\r  Sampling done: {n_batches} batches × {n_per_batch} docs in {elapsed:.0f}s" + " " * 20)
    return batches


MAX_PROMPT_CHARS = 80_000  # ~20K tokens, safe for 32K context with response room

def format_documents(docs, max_chars_per_doc=None):
    """Format documents for the prompt, auto-fitting to context budget."""
    if max_chars_per_doc is None:
        # Reserve ~5K chars for prompt template + response
        budget = MAX_PROMPT_CHARS
        overhead_per_doc = 60  # "--- Document N (url: ...) ---\n"
        max_chars_per_doc = max(200, (budget // len(docs)) - overhead_per_doc)

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
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=300,
    )
    if response.status_code != 200:
        print(f"  vLLM error {response.status_code}: {response.text[:500]}", flush=True)
        response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    # Strip thinking block if Qwen still outputs one
    if "<think>" in content:
        import re
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)

    return content

# ---------------------------------------------------------------------------
# Parse and apply fix
# ---------------------------------------------------------------------------

def parse_response(response):
    """Parse Qwen's response — template-based or code-based."""
    result = {}

    m = re.search(r'## Problem\s*\n(.*?)(?=\n## )', response, re.DOTALL)
    result["problem"] = m.group(1).strip() if m else ""

    m = re.search(r'## Expected Impact\s*\n(.*?)$', response, re.DOTALL)
    result["impact"] = m.group(1).strip() if m else ""

    # Try template format first
    m = re.search(r'## Template\s*\n(\S+)', response)
    if m:
        result["template"] = m.group(1).strip()
        m2 = re.search(r'## Params\s*\n```json\s*\n(.*?)```', response, re.DOTALL)
        if m2:
            try:
                result["params"] = json.loads(m2.group(1).strip())
            except json.JSONDecodeError:
                result["params"] = None
        else:
            result["params"] = None
        result["code"] = None
        result["type"] = None
        return result

    # Fallback: code format
    m = re.search(r'## Type\s*\n(.*?)(?=\n## )', response, re.DOTALL)
    fix_type = m.group(1).strip().lower() if m else ""
    result["type"] = "cleaner" if "clean" in fix_type else "filter"

    m = re.search(r'## Function\s*\n```python\s*\n(.*?)```', response, re.DOTALL)
    result["code"] = m.group(1).strip() if m else ""
    result["template"] = None
    result["params"] = None

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

    # Only these imports are allowed
    allowed_modules = {
        "re", "string", "unicodedata", "collections", "functools",
        "itertools", "math", "html", "textwrap", "difflib",
        "hashlib", "json", "urllib",
        "templates",       # our pre-built filter templates
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

    # Execute the function on test inputs to catch runtime errors (broken regex etc.)
    func_name = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef):
            func_name = node.name
            break
    if func_name:
        test_inputs = [
            "This is a test document with some text.",
            "Læs mere her. Klik her for at se mere.",
            "Kort tekst.",
            "A" * 5000,  # long doc
            "",  # empty
        ]
        namespace = {}
        try:
            exec(compile(code, "<validate>", "exec"), namespace)
        except Exception as e:
            return False, f"Code execution failed: {e}"
        func = namespace.get(func_name)
        if func:
            for test in test_inputs:
                try:
                    result = func(test)
                    if not isinstance(result, (str, bool)):
                        return False, f"Function returned {type(result)}, expected str or bool"
                except Exception as e:
                    return False, f"Function crashed on test input: {e}"

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
    """Run the updated filter on all sampled docs. Check for crashes,
    broken regex, and no-ops. Returns (ok, error_message)."""
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


    # Run on all sampled docs, check for crashes and effectiveness
    n_changed = 0
    n_filtered = 0
    for i, doc in enumerate(all_docs):
        try:
            cleaned = lang_mod.clean(doc["text"])
            kept = lang_mod.should_keep(cleaned)
            if cleaned != doc["text"]:
                n_changed += 1
            if not kept:
                n_filtered += 1
        except Exception as e:
            return False, f"Crashed on doc {i} ({doc['doc_id']}): {e}\n{traceback.format_exc()}"

    print(f"  Effect: {n_changed} cleaned, {n_filtered} filtered out of {len(all_docs)} docs", flush=True)

    if n_changed == 0 and n_filtered == 0:
        return False, f"No-op: function had no effect on any of {len(all_docs)} sampled docs"

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

def _ask_qwen_to_fix(parsed, error, all_observations):
    """Ask Qwen to fix its broken code given the error message."""
    print(f"\nAsking Qwen to fix the error...")
    fix_prompt = (
        f"Your Python function had an error:\n\n"
        f"```python\n{parsed['code']}\n```\n\n"
        f"Error: {error}\n\n"
        f"Fix the function. Keep it SIMPLE (max 1-2 regex). "
        f"Make sure all regex patterns are valid and the function actually modifies or filters text.\n"
        f"Function name MUST be unique and descriptive. NEVER use clean_example.\n\n"
        f"Output ONLY:\n\n"
        f"## Problem\nOne sentence.\n\n"
        f"## Type\n{parsed['type']}\n\n"
        f"## Function\n```python\ndef fixed_function(text):\n    return text\n```\n\n"
        f"## Expected Impact\nX% of docs"
    )
    response = query_qwen(fix_prompt)
    print("=" * 70)
    print(response)
    print("=" * 70)
    new_parsed = parse_response(response)
    if new_parsed["code"]:
        return new_parsed
    return parsed  # give up, return original


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
    # Phase 1: Sample all batches in one pass, then observe
    # ---------------------------------------------------------------

    print("Sampling all batches in one pass...")
    batches = sample_documents_multi(
        args.lang, n_batches=n_peeks, n_per_batch=docs_per_peek,
        from_filtered=args.from_filtered, base_seed=base_seed,
    )
    if not batches:
        print("No documents sampled.")
        exit(1)

    all_observations = []
    all_docs = []

    for peek_idx, docs in enumerate(batches):
        if not docs:
            print(f"Peek {peek_idx+1}: empty batch, skipping")
            continue

        all_docs.extend(docs)

        prompt = OBSERVE_PROMPT.format(
            n=len(docs),
            lang=lang_name,
            documents=format_documents(docs),
            strategy_instruction=strategy,
            focus_instruction=focus_instruction,
        )

        print(f"Peek {peek_idx+1}/{n_peeks} ({len(docs)} docs)... ", end="", flush=True)
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

    from templates import TEMPLATES, get_template_menu

    print(f"\nSynthesizing fix from {n_peeks} observations...")
    synth_prompt = SYNTHESIZE_PROMPT.format(
        n_peeks=n_peeks,
        lang=lang_name,
        all_observations="\n\n".join(all_observations),
        template_menu=get_template_menu(),
    )

    t0 = time.time()
    response = query_qwen(synth_prompt)
    elapsed = time.time() - t0
    print(f"Synthesis received in {elapsed:.1f}s\n")

    print("=" * 70)
    print(response)
    print("=" * 70)

    # Parse response
    parsed = parse_response(response)

    if not parsed.get("template") and not parsed.get("code"):
        print("\nRetry: could not parse response, asking again...")
        retry_prompt = (
            "Your response was not in the correct format. Pick a template:\n\n"
            + get_template_menu() + "\n\n"
            "Output ONLY:\n\n"
            "## Problem\nOne sentence.\n\n"
            "## Template\nTEMPLATE_NAME\n\n"
            "## Params\n```json\n{}\n```\n\n"
            "## Expected Impact\nX% of docs"
        )
        response = query_qwen(retry_prompt)
        print("=" * 70)
        print(response)
        print("=" * 70)
        parsed = parse_response(response)

    print(f"\nParsed: template={parsed.get('template')}, problem={parsed['problem'][:120]}")

    if args.dry_run:
        print("\n[dry-run] Would apply this fix but --dry-run was set.")
        exit(0)

    # Generate code from template
    if parsed.get("template") and parsed.get("params") is not None:
        template_name = parsed["template"]
        if template_name not in TEMPLATES:
            print(f"Unknown template: {template_name}")
            exit(1)

        tmpl = TEMPLATES[template_name]
        params_json = json.dumps(parsed["params"])
        # Generate a clean function that calls the template
        func_name = f"{tmpl['type']}_{template_name.lower()}_{args.iteration or 0}"
        if tmpl["type"] == "cleaner":
            code = (
                f"def {func_name}(text):\n"
                f"    from templates import {tmpl['fn'].__name__}\n"
                f"    return {tmpl['fn'].__name__}(text, **{params_json})\n"
            )
            parsed["type"] = "cleaner"
        else:
            code = (
                f"def {func_name}(text):\n"
                f"    from templates import {tmpl['fn'].__name__}\n"
                f"    return {tmpl['fn'].__name__}(text, **{params_json})\n"
            )
            parsed["type"] = "filter"
        parsed["code"] = code
        print(f"Generated from template {template_name}: {func_name}")
    elif not parsed.get("code"):
        print("ERROR: No template or code in response.")
        exit(1)

    # Apply + verify
    MAX_FIX_ATTEMPTS = 3
    for attempt in range(MAX_FIX_ATTEMPTS):
        if parsed.get("code"):
            ok, reason = validate_code(parsed["code"])
            if not ok:
                print(f"Code validation FAILED: {reason}")
                if attempt < MAX_FIX_ATTEMPTS - 1:
                    parsed = _ask_qwen_to_fix(parsed, reason, all_observations)
                    continue
                print("Fix rejected after retries.")
                exit(1)

        ok = apply_fix_to_filter(args.lang, parsed, iteration=args.iteration)
        if not ok:
            print("Fix was NOT applied.")
            exit(1)

        print(f"\nVerifying fix on {len(all_docs)} docs...")
        ok, err = verify_fix(args.lang, all_docs)
        if ok:
            print("Verification passed.")
            break
        else:
            print(f"Verification FAILED: {err}")
            print("Rolling back fix...")
            rollback_fix(args.lang, parsed)
            if attempt < MAX_FIX_ATTEMPTS - 1:
                parsed = _ask_qwen_to_fix(parsed, err, all_observations)
            else:
                print("Fix rejected after retries.")
                exit(1)
