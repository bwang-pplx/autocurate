"""
Train a BPE tokenizer on Danish fineweb-2 data.
Same approach as the original prepare.py but reads from our Danish data.

Usage: python setup_tokenizer.py [--lang dan_Latn]

Saves to ~/.cache/autoresearch/tokenizer/ (where prepare.py expects it).
"""

import os
import sys
import time
import pickle
import argparse

import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

from prepare import list_raw_parquet_files

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

VOCAB_SIZE = 8192
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]


def text_iterator(lang, max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from raw fineweb-2 data."""
    parquet_files = list_raw_parquet_files(lang)
    nchars = 0
    for filepath in parquet_files:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return
        print(f"  Tokenizer training: {nchars/1e9:.1f}B chars read...", flush=True)


def setup(lang):
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer already exists at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_raw_parquet_files(lang)
    if not parquet_files:
        print("No raw data found. Run prepare.py --phase download first.")
        sys.exit(1)

    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE}) on {lang}...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(lang), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="danish_bpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")
    print(f"Vocab size: {enc.n_vocab}")

    # Build token_bytes lookup for BPB evaluation
    print("Building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hej verden! Danish: æøå. Numbers: 123."
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Sanity check passed: '{test}' -> {len(encoded)} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="dan_Latn")
    args = parser.parse_args()
    setup(args.lang)
