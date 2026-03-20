"""
Set up a multilingual tokenizer for data-quality experiments.
Uses tiktoken's cl100k_base (GPT-4 tokenizer) which covers all languages well.

Usage: python setup_tokenizer.py

Saves to ~/.cache/autoresearch/tokenizer/ (where prepare.py expects it).
"""

import os
import pickle

import tiktoken
import torch

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def setup():
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer already exists at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # Use cl100k_base (GPT-4 tokenizer) — multilingual, 100K vocab
    print("Loading cl100k_base tokenizer...")
    base = tiktoken.get_encoding("cl100k_base")

    # Re-wrap with our special tokens
    mergeable_ranks = base._mergeable_ranks
    pat_str = base._pat_str
    tokens_offset = max(mergeable_ranks.values()) + 1
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

    enc = tiktoken.Encoding(
        name="cl100k_data",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Saved tokenizer to {tokenizer_pkl} (vocab_size={enc.n_vocab})")

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
    test = "Hej verden! Danish: æøå. Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Sanity check passed: '{test}' -> {len(encoded)} tokens")


if __name__ == "__main__":
    setup()
