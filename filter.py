"""
Apply the language-specific cleaning + filtering pipeline.

Usage:
    python filter.py --lang dan_Latn              # apply pipeline, output selected docs
    python filter.py --lang dan_Latn --preview 10  # show 10 cleaned docs
"""

import os
import json
import argparse
import importlib

import pyarrow.parquet as pq

from prepare_data import get_lang_dir, get_eval_dir, list_raw_parquet_files


def load_lang_filter(lang):
    """Import the language-specific filter module (e.g. filter_dan for dan_Latn)."""
    lang_code = lang.split("_")[0]  # dan_Latn -> dan
    module_name = f"filter_{lang_code}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(f"No filter module found: {module_name}.py")
        print(f"Create it or run peek.py to generate the first rule.")
        raise


# 5 min training ≈ 500M tokens ≈ 2M docs. Keep 5x margin for variety.
MAX_KEPT_DOCS = 5_000_000

def apply_pipeline(lang, preview=0, max_kept=MAX_KEPT_DOCS):
    """Apply clean + filter to documents. Stops early once we have enough."""
    lang_mod = load_lang_filter(lang)
    lang_dir = get_lang_dir(lang)
    parquet_files = list_raw_parquet_files(lang)

    if not parquet_files:
        print("No raw data found. Run prepare_data.py --phase download first.")
        return

    # Load eval doc IDs to exclude
    eval_ids = set()
    eval_ids_path = os.path.join(get_eval_dir(lang), "eval_doc_ids.json")
    if os.path.exists(eval_ids_path):
        with open(eval_ids_path) as f:
            eval_ids = set(json.load(f))

    import time
    total = 0
    kept = 0
    cleaned_count = 0
    selected_ids = []
    preview_docs = []
    t0 = time.time()

    for fi, filepath in enumerate(parquet_files):
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            ids = rg.column("id").to_pylist()

            for doc_id, text in zip(ids, texts):
                if doc_id in eval_ids:
                    continue
                total += 1

                cleaned = lang_mod.clean(text)
                if cleaned != text:
                    cleaned_count += 1

                if lang_mod.should_keep(cleaned):
                    kept += 1
                    selected_ids.append(doc_id)

                    if preview > 0 and len(preview_docs) < preview:
                        preview_docs.append({
                            "doc_id": doc_id,
                            "original_len": len(text),
                            "cleaned_len": len(cleaned),
                            "text_preview": cleaned[:500],
                        })

            if total % 500_000 == 0 and total > 0:
                elapsed = time.time() - t0
                rate = total / elapsed
                print(f"\r  [{fi+1}/{len(parquet_files)} files] {total:,} docs, kept {kept:,} ({100*kept/max(total,1):.1f}%), {rate:,.0f} docs/s", end="", flush=True)

            if max_kept and kept >= max_kept:
                break
        if max_kept and kept >= max_kept:
            break

    # Save selected doc IDs
    output_path = os.path.join(lang_dir, "selected_doc_ids.json")
    with open(output_path, "w") as f:
        json.dump(selected_ids, f)

    elapsed = time.time() - t0
    print()
    print()
    print(f"Total docs:    {total:,}")
    print(f"Time:          {elapsed:.0f}s ({total/max(elapsed,1):,.0f} docs/s)")
    print(f"Kept:          {kept:,} ({100*kept/max(total,1):.1f}%)")
    print(f"Dropped:       {total - kept:,} ({100*(total-kept)/max(total,1):.1f}%)")
    print(f"Cleaned:       {cleaned_count:,} ({100*cleaned_count/max(total,1):.1f}%)")
    print(f"Saved to:      {output_path}")

    if preview_docs:
        print()
        print("=" * 60)
        print("Preview of kept documents:")
        print("=" * 60)
        for doc in preview_docs:
            print(f"\n--- {doc['doc_id']} (orig: {doc['original_len']}, cleaned: {doc['cleaned_len']}) ---")
            print(doc["text_preview"])

    return selected_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply cleaning + filtering pipeline")
    parser.add_argument("--lang", type=str, default="dan_Latn")
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    apply_pipeline(args.lang, preview=args.preview)
