# autocurate

Autonomous data quality curation for [fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2). An AI agent iteratively discovers quality problems in web-crawled text and builds a growing pipeline of heuristic fixes — validated by training a small language model and measuring BPB improvement.


## How it works

```
Sample docs → Qwen reads N batches → Synthesize fix → Verify → Filter → Train → BPB improved? → Keep/Revert
```

Each iteration:

1. **Peek** — Sample random documents, send to Qwen (via vLLM) across multiple rounds. Qwen identifies quality problems (boilerplate, spam, truncation, etc.)
2. **Synthesize** — Qwen picks from a menu of pre-built filter templates and provides parameters (string lists, thresholds). No regex or code generation needed.
3. **Verify** — Execute the fix on sampled docs. Reject if it crashes, has no effect, or uses banned imports.
4. **Filter** — Apply the growing pipeline of rules to the corpus (capped at 5M docs for speed).
5. **Train** — Train a frozen small GPT for 5 minutes on the filtered data.
6. **Evaluate** — Measure BPB on a held-out Wikipedia eval set.
7. **Decide** — BPB improved → keep the rule. BPB worse → `git revert`.

The filter file (`filter_{lang}.py`) grows over iterations, accumulating only rules that improve data quality.

## Quick start

```bash
# On a GPU node with 8 GPUs
git clone https://github.com/bwang-pplx/autocurate.git
cd autocurate

# Launch for Danish (downloads data, starts vLLM + training loop)
bash new_lang.sh dan_Latn 100

# Monitor
tail -f logs/loop_*.out
cat results_dan.tsv
```

## Launch on SLURM

```bash
# Single language
bash slurm/launch.sh dan_Latn 100

# Multiple languages in parallel (each takes 8 GPUs)
bash new_lang.sh swe_Latn 100
bash new_lang.sh nob_Latn 100
bash new_lang.sh fin_Latn 100
```

## GPU layout

```
GPU 0-3:  vLLM server (Qwen3.5-35B-A3B, TP=4)
GPU 7:    Training (frozen GPT, 5 min per run)
```

## Filter templates

Instead of generating code, Qwen picks from pre-built, tested templates:

| Template | Type | What it does |
|---|---|---|
| `REMOVE_LINES_CONTAINING` | cleaner | Remove lines containing any of the given strings |
| `DROP_IF_CONTAINS` | filter | Drop doc if it contains N+ of the given strings |
| `REMOVE_PREFIX_LINES` | cleaner | Strip leading lines (navbars, breadcrumbs) |
| `REMOVE_SUFFIX_LINES` | cleaner | Strip trailing lines (footers, disclaimers) |
| `DROP_SHORT_DOCS` | filter | Drop docs under N chars |
| `DROP_IF_KEYWORD_DENSITY` | filter | Drop if keyword density exceeds threshold |
| `REPLACE_STRINGS` | cleaner | Simple string replacement |
| `REMOVE_DUPLICATE_LINES` | cleaner | Remove exact duplicate lines |
| `DROP_BY_LANGUAGE_MARKERS` | filter | Drop docs with too many foreign words |

## Safety

Three validation layers before any rule is applied:

1. **AST check** — Only stdlib + templates imports allowed. No ML, no network calls.
2. **Runtime test** — Execute on all sampled docs. Crash → rollback.
3. **No-op check** — Reject if the function changes zero docs.

If validation fails, Qwen is asked to fix the error (up to 3 attempts).

After training, BPB regression → `git revert` removes the rule.

## Export

When iterations plateau, export the cleaned dataset to HuggingFace:

```bash
# Download full dataset first (loop only downloads 10M docs)
uv run prepare.py --lang dan_Latn

# Export and push
uv run export.py --lang dan_Latn --push
```

Output preserves fineweb-2's structure: `data/{lang}/train/*.parquet`

## Files

| File | Role |
|---|---|
| `peek.py` | Sample docs → Qwen analysis → synthesize fix → verify → append |
| `filter_{lang}.py` | Auto-growing pipeline of cleaners + filters (per language) |
| `filter.py` | Dispatcher: imports the right `filter_{lang}.py` |
| `templates.py` | Pre-built, tested filter templates |
| `prepare.py` | Download fineweb-2 + Wikipedia eval set + tokenizer + dataloader |
| `train.py` | Frozen GPT model + training loop |
| `setup_tokenizer.py` | Train 8K BPE tokenizer on target language |
| `export.py` | Apply all rules to full dataset, push to HuggingFace |
| `new_lang.sh` | Bootstrap a new language in one command |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a visual diagram.

## Acknowledgments

Built on top of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The model architecture, optimizer, and training loop are from the original project — we froze them and redirected the autonomous research loop toward data quality instead of model architecture.
