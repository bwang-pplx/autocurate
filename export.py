"""
Export the cleaned dataset to HuggingFace as a language subset.

Downloads the FULL dataset (not capped), applies the filter pipeline,
writes cleaned parquet files, and pushes to HuggingFace.

Each language is a config/subset, matching fineweb-2's structure:
  HuggingFaceFW/fineweb-2        → data/dan_Latn/train/*.parquet
  bwang-pplx/fineweb-2-autocurate → data/dan_Latn/train/*.parquet

Usage:
    python export.py --lang dan_Latn                          # export locally
    python export.py --lang dan_Latn --push                   # export + push to HF
    python export.py --lang dan_Latn --push --repo bwang-pplx/fineweb-2-autocurate

NOTE: Run `python prepare_data.py --lang dan_Latn` WITHOUT --max-docs first
      to download the full dataset for export.
"""

import os
import json
import argparse
import importlib
import time

import pyarrow as pa
import pyarrow.parquet as pq

from prepare_data import get_lang_dir, get_eval_dir, list_raw_parquet_files

HF_REPO = "bwang-pplx/fineweb-2-autocurate"
EXPORT_DIR = "export"

# Preserve all original fineweb-2 columns (discovered at runtime)


def export_language(lang, push=False, repo=None):
    """Apply filter pipeline and export cleaned data as parquet."""
    repo = repo or HF_REPO
    lang_code = lang.split("_")[0]

    # Import language-specific filter
    lang_mod = importlib.import_module(f"filter_{lang_code}")

    parquet_files = list_raw_parquet_files(lang)
    if not parquet_files:
        print("No raw data found.")
        return

    # Exclude eval docs
    eval_ids = set()
    eval_ids_path = os.path.join(get_eval_dir(lang), "eval_doc_ids.json")
    if os.path.exists(eval_ids_path):
        with open(eval_ids_path) as f:
            eval_ids = set(json.load(f))

    # Output directory: export/data/{lang}/train/
    out_dir = os.path.join(EXPORT_DIR, "data", lang, "train")
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    kept = 0
    cleaned_count = 0
    shard_idx = 0
    shard_rows = []
    ROWS_PER_SHARD = 500_000
    t0 = time.time()

    for fi, filepath in enumerate(parquet_files):
        pf = pq.ParquetFile(filepath)
        schema = pf.schema_arrow

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            ids = rg.column("id").to_pylist()

            # Read all columns from the original data
            all_columns = rg.schema.names
            col_data = {col: rg.column(col).to_pylist() for col in all_columns}

            for i in range(len(texts)):
                doc_id = ids[i]
                if doc_id in eval_ids:
                    continue
                total += 1

                text = texts[i]
                cleaned = lang_mod.clean(text)
                if cleaned != text:
                    cleaned_count += 1

                if lang_mod.should_keep(cleaned):
                    kept += 1
                    row = {col: col_data[col][i] for col in all_columns}
                    row["text"] = cleaned  # replace with cleaned text
                    shard_rows.append(row)

                    # Flush shard when full
                    if len(shard_rows) >= ROWS_PER_SHARD:
                        _write_shard(out_dir, shard_idx, shard_rows)
                        shard_idx += 1
                        shard_rows = []

        elapsed = time.time() - t0
        rate = total / max(elapsed, 1)
        print(f"  [{fi+1}/{len(parquet_files)}] {total:,} docs, kept {kept:,} ({100*kept/max(total,1):.1f}%), {rate:,.0f} docs/s", flush=True)

    # Flush remaining
    if shard_rows:
        _write_shard(out_dir, shard_idx, shard_rows)
        shard_idx += 1

    elapsed = time.time() - t0
    print()
    print()
    print(f"Export complete:")
    print(f"  Total docs:   {total:,}")
    print(f"  Kept:         {kept:,} ({100*kept/max(total,1):.1f}%)")
    print(f"  Cleaned:      {cleaned_count:,}")
    print(f"  Shards:       {shard_idx}")
    print(f"  Time:         {elapsed:.0f}s")
    print(f"  Output:       {out_dir}")

    # Push to HuggingFace
    if push:
        _push_to_hf(repo, lang, out_dir, total, kept)


def _write_shard(out_dir, shard_idx, rows):
    """Write a shard of rows as parquet."""
    filename = f"{shard_idx:05d}.parquet"
    filepath = os.path.join(out_dir, filename)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, filepath, compression="zstd")
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"\n  Wrote {filename} ({len(rows):,} rows, {size_mb:.0f} MB)", flush=True)


def _push_to_hf(repo, lang, out_dir, total_docs, kept_docs):
    """Push exported parquet files to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"  Note: {e}")

    # Upload all parquet files
    parquet_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".parquet"))
    print(f"\nPushing {len(parquet_files)} shards to {repo} (subset: {lang})...")

    for f in parquet_files:
        local_path = os.path.join(out_dir, f)
        remote_path = f"data/{lang}/train/{f}"
        print(f"  Uploading {f}...", flush=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo,
            repo_type="dataset",
        )

    # Update README with this language's stats
    readme_path = os.path.join(EXPORT_DIR, "README.md")
    _update_readme(readme_path, repo, lang, total_docs, kept_docs)

    print(f"\nPushed to https://huggingface.co/datasets/{repo}")
    print(f"Subset: {lang}")


def _update_readme(readme_path, repo, lang, total_docs, kept_docs):
    """Update or create the dataset README."""
    from huggingface_hub import HfApi

    header = f"""---
dataset_info:
  - config_name: {lang}
    features:
      - name: text
        dtype: string
      - name: id
        dtype: string
      - name: url
        dtype: string
      - name: dump
        dtype: string
      - name: date
        dtype: string
      - name: language
        dtype: string
      - name: language_score
        dtype: float64
license: odc-by
task_categories:
  - text-generation
language:
  - {lang.split('_')[0]}
tags:
  - fineweb
  - autocurate
  - autoresearch
---

# fineweb-2-autocurate

Autonomously curated subsets of [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).

An AI agent (Qwen3.5-35B-A3B) iteratively analyzed random samples of web documents,
identified quality problems, and proposed heuristic fixes (regex, string operations).
Each fix was validated by training a small language model and measuring BPB improvement
on a Wikipedia eval set. Only fixes that improved BPB were kept.

## Subsets

| Language | Original docs | Kept docs | Kept % |
|---|---|---|---|
"""

    lang_line = f"| {lang} | {total_docs:,} | {kept_docs:,} | {100*kept_docs/max(total_docs,1):.1f}% |\n"

    if os.path.exists(readme_path):
        with open(readme_path) as f:
            content = f.read()
        if lang in content:
            # Update existing line
            import re
            content = re.sub(rf'\| {lang} \|.*\n', lang_line, content)
        else:
            content = content.rstrip() + "\n" + lang_line
    else:
        os.makedirs(os.path.dirname(readme_path), exist_ok=True)
        content = header + lang_line

    with open(readme_path, "w") as f:
        f.write(content)

    # Upload README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export cleaned dataset to HuggingFace")
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace")
    parser.add_argument("--repo", type=str, default=HF_REPO, help="HF repo name")
    args = parser.parse_args()

    export_language(args.lang, push=args.push, repo=args.repo)
