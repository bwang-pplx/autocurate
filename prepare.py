"""
Data preparation for data-quality autoresearch experiments.
Downloads fineweb-2 for a target language and builds a fixed eval set.

Usage:
    python prepare.py                          # full prep for Danish
    python prepare.py --lang zho_Hans          # different language
    python prepare.py --max-docs 100000        # small test run
    python prepare.py --phase download         # only download
    python prepare.py --phase eval             # only build eval set

Data is stored in ~/.cache/autoresearch-data/<lang>/.
"""

import os
import sys
import time
import json
import argparse
import random

import pickle

import pyarrow.parquet as pq
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-data")
TOKENIZER_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")
HF_DATASET = "HuggingFaceFW/fineweb-2"

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Tokenizer (moved from prepare.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)

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

WIKI_LANG_MAP = {
    "dan_Latn": "da", "deu_Latn": "de", "fra_Latn": "fr", "spa_Latn": "es",
    "ita_Latn": "it", "por_Latn": "pt", "nld_Latn": "nl", "swe_Latn": "sv",
    "pol_Latn": "pl", "tur_Latn": "tr", "vie_Latn": "vi", "ind_Latn": "id",
    "ron_Latn": "ro", "ces_Latn": "cs", "hun_Latn": "hu", "fin_Latn": "fi",
    "nob_Latn": "no", "kor_Hang": "ko", "zho_Hans": "zh",
    "rus_Cyrl": "ru", "ukr_Cyrl": "uk", "bul_Cyrl": "bg",
    "arb_Arab": "ar", "fas_Arab": "fa", "jpn_Jpan": "ja",
    "tha_Thai": "th", "ell_Grek": "el", "heb_Hebr": "he",
    "hin_Deva": "hi", "ben_Beng": "bn",
}


def build_eval_set(lang, n_eval_docs=10000):
    """Build a fixed eval set from Wikipedia for the target language."""
    eval_dir = get_eval_dir(lang)
    eval_path = os.path.join(eval_dir, "eval.parquet")
    if os.path.exists(eval_path):
        print(f"Eval set already exists at {eval_path}")
        return

    os.makedirs(eval_dir, exist_ok=True)

    wiki_lang = WIKI_LANG_MAP.get(lang)
    if wiki_lang:
        print(f"Downloading Wikipedia eval set ({wiki_lang})...")
        eval_texts, eval_ids = _download_wiki_eval(wiki_lang, n_eval_docs)
    else:
        print(f"No Wikipedia mapping for {lang}, falling back to raw data sample")
        eval_texts, eval_ids = _sample_raw_eval(lang, n_eval_docs)

    if not eval_texts:
        print("ERROR: Could not build eval set.")
        return

    import pandas as pd
    df = pd.DataFrame({"doc_id": eval_ids, "text": eval_texts})
    df.to_parquet(eval_path, index=False)
    print(f"Eval set: {len(eval_texts):,} docs saved to {eval_path}")

    eval_ids_path = os.path.join(eval_dir, "eval_doc_ids.json")
    with open(eval_ids_path, "w") as f:
        json.dump(eval_ids, f)
    print(f"Eval doc IDs saved to {eval_ids_path}")


def _download_wiki_eval(wiki_lang, n_docs):
    """Download Wikipedia articles via HuggingFace wikimedia/wikipedia dataset."""
    from huggingface_hub import HfApi, hf_hub_download
    import random

    # wikimedia/wikipedia stores files at 20231101.{lang}/train-*.parquet
    config = f"20231101.{wiki_lang}"
    eval_dir_tmp = os.path.join(CACHE_DIR, "_wiki_tmp")
    os.makedirs(eval_dir_tmp, exist_ok=True)

    # List train parquet files
    try:
        api = HfApi()
        files = list(api.list_repo_tree(
            "wikimedia/wikipedia",
            path_in_repo=config,
            repo_type="dataset",
        ))
        train_files = [f for f in files if hasattr(f, 'path')
                      and 'train' in f.path and f.path.endswith('.parquet')]
    except Exception as e:
        print(f"  Wikipedia listing failed: {e}")
        return [], []

    if not train_files:
        print(f"  No Wikipedia train files found for {config}")
        return [], []

    # Download first shard (enough for 10K articles)
    print(f"  Downloading {train_files[0].path}...")
    dl_path = hf_hub_download(
        repo_id="wikimedia/wikipedia",
        filename=train_files[0].path,
        repo_type="dataset",
        local_dir=eval_dir_tmp,
    )

    print(f"  Reading Wikipedia articles...")
    pf = pq.ParquetFile(dl_path)
    all_texts = []
    all_ids = []
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        ids = rg.column("id").to_pylist() if "id" in rg.schema.names else [f"wiki_{rg_idx}_{i}" for i in range(len(texts))]
        for i, text in enumerate(texts):
            if 500 <= len(text) <= 50000:
                all_texts.append(text)
                all_ids.append(f"wiki_{ids[i]}")

    # Random subsample
    if len(all_texts) > n_docs:
        random.seed(42)
        indices = random.sample(range(len(all_texts)), n_docs)
        all_texts = [all_texts[i] for i in indices]
        all_ids = [all_ids[i] for i in indices]

    print(f"  Got {len(all_texts):,} Wikipedia articles (from {pf.metadata.num_rows:,} total)")
    return all_texts, all_ids


def _sample_raw_eval(lang, n_docs):
    """Fallback: sample from tail of raw fineweb-2 data."""
    parquet_files = list_raw_parquet_files(lang)
    if not parquet_files:
        return [], []

    last_file = parquet_files[-1]
    print(f"  Sampling from {os.path.basename(last_file)}...")
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
                if len(eval_texts) >= n_docs:
                    break
        if len(eval_texts) >= n_docs:
            break

    return eval_texts, eval_ids

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
