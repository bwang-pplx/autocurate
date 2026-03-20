"""
Data preparation for data-quality autoresearch experiments.
Downloads fineweb-2 for a target language and builds a fixed eval set.

Usage:
    python prepare_data.py                          # full prep for Danish
    python prepare_data.py --lang zho_Hans          # different language
    python prepare_data.py --max-docs 100000        # small test run
    python prepare_data.py --phase download         # only download
    python prepare_data.py --phase eval             # only build eval set

Data is stored in ~/.cache/autoresearch-data/<lang>/.
"""

import os
import sys
import time
import json
import argparse
import random

import pyarrow.parquet as pq
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-data")
HF_DATASET = "HuggingFaceFW/fineweb-2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_lang_dir(lang):
    return os.path.join(CACHE_DIR, lang)

def get_raw_dir(lang):
    return os.path.join(get_lang_dir(lang), "raw")

def get_eval_dir(lang):
    return os.path.join(get_lang_dir(lang), "eval")

def list_raw_parquet_files(lang):
    """Return sorted list of raw parquet files for a language."""
    raw_dir = get_raw_dir(lang)
    files = []
    for root, dirs, filenames in os.walk(raw_dir):
        for f in sorted(filenames):
            if f.endswith(".parquet") and not f.endswith(".tmp"):
                files.append(os.path.join(root, f))
    return files

# ---------------------------------------------------------------------------
# Phase 1: Download fineweb-2 for target language
# ---------------------------------------------------------------------------

def download_language(lang, max_docs=None):
    """Download fineweb-2 parquet files for a language using HF hub."""
    from huggingface_hub import HfApi, hf_hub_download

    raw_dir = get_raw_dir(lang)
    os.makedirs(raw_dir, exist_ok=True)

    api = HfApi()
    # Parquet files are under data/{lang}/train/ (and optionally test/)
    parquet_files = []
    for split in ["train", "test"]:
        try:
            files = list(api.list_repo_tree(
                HF_DATASET,
                path_in_repo=f"data/{lang}/{split}",
                repo_type="dataset",
            ))
            parquet_files.extend([f for f in files if hasattr(f, 'path') and f.path.endswith('.parquet')])
        except Exception:
            pass  # split may not exist
    print(f"Found {len(parquet_files)} parquet files for {lang}")

    total_docs = 0
    downloaded = []
    for i, f in enumerate(parquet_files):
        local_path = os.path.join(raw_dir, os.path.basename(f.path))
        if os.path.exists(local_path):
            print(f"  [{i+1}/{len(parquet_files)}] Already exists: {os.path.basename(f.path)}")
            downloaded.append(local_path)
            if max_docs:
                pf = pq.ParquetFile(local_path)
                total_docs += pf.metadata.num_rows
                if total_docs >= max_docs:
                    print(f"  Reached {total_docs} docs (max_docs={max_docs}), stopping")
                    break
            continue

        t0 = time.time()
        size_gb = f.size / 1e9 if hasattr(f, 'size') and f.size else 0
        print(f"  [{i+1}/{len(parquet_files)}] Downloading {os.path.basename(f.path)} ({size_gb:.1f} GB)...", flush=True)
        dl_path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=f.path,
            repo_type="dataset",
            local_dir=raw_dir,
            local_dir_use_symlinks=False,
        )
        elapsed = time.time() - t0
        speed = size_gb / max(elapsed, 1) * 1024  # MB/s
        print(f"  [{i+1}/{len(parquet_files)}] Done in {elapsed:.0f}s ({speed:.0f} MB/s)", flush=True)
        # hf_hub_download returns the actual path; move to flat structure
        if dl_path != local_path and os.path.exists(dl_path):
            os.rename(dl_path, local_path)
        downloaded.append(local_path)

        if max_docs:
            pf = pq.ParquetFile(local_path)
            total_docs += pf.metadata.num_rows
            if total_docs >= max_docs:
                print(f"  Reached {total_docs} docs (max_docs={max_docs}), stopping")
                break

    total_size = sum(os.path.getsize(p) for p in downloaded if os.path.exists(p))
    print(f"Download complete: {len(downloaded)} files, {total_size/1e9:.1f} GB total")
    return downloaded

# ---------------------------------------------------------------------------
# Phase 2: Build eval set
# ---------------------------------------------------------------------------

def build_eval_set(lang, n_eval_docs=10000):
    """Build a fixed eval set. Sampled from tail of raw data, never changes."""
    eval_dir = get_eval_dir(lang)
    eval_path = os.path.join(eval_dir, "eval.parquet")
    if os.path.exists(eval_path):
        print(f"Eval set already exists at {eval_path}")
        return

    os.makedirs(eval_dir, exist_ok=True)

    parquet_files = list_raw_parquet_files(lang)
    if not parquet_files:
        print("No raw data found. Run download first.")
        return

    # Take from the LAST file to minimize overlap with training
    last_file = parquet_files[-1]
    print(f"Building eval set from {os.path.basename(last_file)}...")
    pf = pq.ParquetFile(last_file)

    eval_texts = []
    eval_ids = []
    for rg_idx in range(pf.num_row_groups - 1, -1, -1):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        ids = rg.column("id").to_pylist()
        for text, doc_id in zip(texts, ids):
            if 500 <= len(text) <= 50000:
                eval_texts.append(text)
                eval_ids.append(doc_id)
                if len(eval_texts) >= n_eval_docs:
                    break
        if len(eval_texts) >= n_eval_docs:
            break

    import pandas as pd
    df = pd.DataFrame({"doc_id": eval_ids, "text": eval_texts})
    df.to_parquet(eval_path, index=False)
    print(f"Eval set: {len(eval_texts):,} docs saved to {eval_path}")

    eval_ids_path = os.path.join(eval_dir, "eval_doc_ids.json")
    with open(eval_ids_path, "w") as f:
        json.dump(eval_ids, f)
    print(f"Eval doc IDs saved to {eval_ids_path}")

# ---------------------------------------------------------------------------
# Runtime: Dataloader that applies clean+filter on the fly
# ---------------------------------------------------------------------------

def make_filtered_dataloader(tokenizer, B, T, lang):
    """
    Dataloader that reads raw parquet, applies filter.py's clean+filter,
    tokenizes on the fly, and packs into batches.
    """
    import torch

    # Load selected doc IDs from filter.py output
    ids_path = os.path.join(get_lang_dir(lang), "selected_doc_ids.json")
    with open(ids_path) as f:
        selected_ids = set(json.load(f))
    print(f"Filtered dataloader: {len(selected_ids):,} selected docs")

    # Import cleaning function from language-specific filter
    import importlib
    lang_code = lang.split("_")[0]
    lang_mod = importlib.import_module(f"filter_{lang_code}")
    clean = lang_mod.clean

    bos_token = tokenizer.get_bos_token_id()
    row_capacity = T + 1
    parquet_files = list_raw_parquet_files(lang)

    # Exclude eval docs
    eval_ids = set()
    eval_ids_path = os.path.join(get_eval_dir(lang), "eval_doc_ids.json")
    if os.path.exists(eval_ids_path):
        with open(eval_ids_path) as f:
            eval_ids = set(json.load(f))

    # Pre-allocate buffers
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    doc_buffer = []
    TOKENIZE_BATCH = 128

    def _doc_iterator():
        """Infinite iterator over cleaned, tokenized docs from selected set."""
        epoch = 1
        docs_yielded = 0
        t_start = time.time()
        while True:
            for fi, filepath in enumerate(parquet_files):
                pf = pq.ParquetFile(filepath)
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column("text").to_pylist()
                    ids = rg.column("id").to_pylist()

                    # Filter to selected IDs and clean
                    batch_texts = []
                    for doc_id, text in zip(ids, texts):
                        if doc_id in selected_ids and doc_id not in eval_ids:
                            cleaned = clean(text)
                            if cleaned.strip():
                                batch_texts.append(cleaned)

                        # Batch tokenize when we have enough
                        if len(batch_texts) >= TOKENIZE_BATCH:
                            token_lists = tokenizer.encode(batch_texts, prepend=bos_token)
                            for tokens in token_lists:
                                docs_yielded += 1
                                yield tokens, epoch
                            batch_texts = []

                    # Flush remaining
                    if batch_texts:
                        token_lists = tokenizer.encode(batch_texts, prepend=bos_token)
                        for tokens in token_lists:
                            docs_yielded += 1
                            yield tokens, epoch
                        batch_texts = []

                # Log once per file
                if fi == 0 or (fi + 1) % 5 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [dataloader] epoch {epoch}, file {fi+1}/{len(parquet_files)}, {docs_yielded:,} docs, {elapsed:.0f}s")

            epoch += 1

    doc_iter = _doc_iterator()

    def refill_buffer(target_size=1000):
        while len(doc_buffer) < target_size:
            tokens, epoch = next(doc_iter)
            doc_buffer.append((tokens, epoch))

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                refill_buffer()
                remaining = row_capacity - pos

                # Best-fit packing
                best_idx = -1
                best_len = 0
                for i, (doc, _) in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc, epoch = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i][0]))
                    doc, epoch = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


def make_eval_dataloader(tokenizer, B, T, lang):
    """Dataloader for the fixed eval set."""
    import torch

    eval_path = os.path.join(get_eval_dir(lang), "eval.parquet")
    df = pq.read_table(eval_path).to_pandas()
    texts = df["text"].tolist()

    bos_id = tokenizer.get_bos_token_id()
    all_tokens = tokenizer.encode(texts, prepend=bos_id)

    flat = []
    for toks in all_tokens:
        flat.extend(toks)
    flat = torch.tensor(flat, dtype=torch.long)

    row_capacity = T + 1
    n_rows = len(flat) // row_capacity
    flat = flat[:n_rows * row_capacity].view(n_rows, row_capacity)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    idx = 0
    while True:
        for row_idx in range(B):
            row_buffer[row_idx] = flat[idx % n_rows]
            idx += 1
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for data-quality autoresearch")
    parser.add_argument("--lang", type=str, default="dan_Latn")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "download", "eval"])
    args = parser.parse_args()

    lang = args.lang
    lang_dir = get_lang_dir(lang)
    os.makedirs(lang_dir, exist_ok=True)
    print(f"Language: {lang}")
    print(f"Cache dir: {lang_dir}")
    print()

    if args.phase in ("all", "download"):
        print("=" * 60)
        print("Phase 1: Download")
        print("=" * 60)
        download_language(lang, max_docs=args.max_docs)
        print()

    if args.phase in ("all", "eval"):
        print("=" * 60)
        print("Phase 2: Build eval set")
        print("=" * 60)
        build_eval_set(lang)
        print()

    print("Done!")
