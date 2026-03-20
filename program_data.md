# autoresearch — data quality

Autonomous research on **data quality** for fineweb-2 language subsets.
Qwen analyzes sampled documents, proposes regex/heuristic fixes, and
the fixes are automatically applied to a per-language filter pipeline.

## Setup

1. **Agree on a run tag**: e.g. `mar20-dan`.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`.
3. **Read the in-scope files**:
   - `filter_dan.py` — the auto-growing pipeline for Danish. DO NOT edit manually.
   - `peek.py` — samples docs, queries Qwen, appends fix to filter_dan.py.
   - `filter.py` — dispatcher that imports the right filter_{lang}.py.
   - `prepare_data.py` — fixed data download and dataloaders. Do not modify.
   - `train_data.py` — frozen model + training. Do not modify.
4. **Verify data exists**: `~/.cache/autoresearch-data/dan_Latn/` must have:
   - `raw/` — fineweb-2 parquet files
   - `eval/` — held-out evaluation set
   If not: `uv run prepare_data.py --lang dan_Latn`.
5. **Start vLLM server** for Qwen:
   ```
   vllm serve Qwen/Qwen3.5-35B-A3B --tensor-parallel-size 1
   ```
6. **Initialize results.tsv** and confirm setup.

## The experiment loop

LOOP FOREVER:

1. **Peek**: `uv run peek.py --lang dan_Latn --seed <random> --iteration <N>`
   - Samples 30 docs, sends to Qwen
   - Qwen identifies a quality problem and proposes a fix
   - The fix is automatically appended to `filter_dan.py`
   - After the first pass, use `--from-filtered` to find remaining problems

2. **git commit**: Commit the updated `filter_dan.py`.

3. **Filter**: `uv run filter.py --lang dan_Latn`
   - Applies the full pipeline (all cleaners + all filters)
   - Note: total docs, kept %, cleaned %

4. **Train**: `uv run train_data.py --lang dan_Latn > run.log 2>&1`

5. **Read results**: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

6. **Log to results.tsv** and decide: keep or revert.
   - If BPB improved → keep (advance branch)
   - If BPB worse → `git revert HEAD` (removes the last rule from filter_dan.py)

## Results TSV

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	44.0	keep	baseline (no filter)
b2c3d4e	1.220000	44.0	keep	strip cookie banners
c3d4e5f	1.225000	44.0	discard	drop docs with >50% English
```

## NEVER STOP

Once the loop has begun, do NOT pause. Run until manually stopped.
Each iteration: peek → commit → filter → train → eval → keep/revert.
