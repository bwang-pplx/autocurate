#!/bin/bash
#SBATCH --job-name=autoresearch-data
#SBATCH --output=logs/loop_%j.out
#SBATCH --error=logs/loop_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

LANG="${1:-dan_Latn}"
MAX_ITERATIONS="${2:-100}"
VLLM_JOB_ID="${3:-}"

mkdir -p logs

# ---------------------------------------------------------------------------
# Find the vLLM server
# ---------------------------------------------------------------------------

if [ -n "$VLLM_JOB_ID" ]; then
    # Read hostname from the vLLM job
    HOSTFILE="/tmp/vllm_host_${VLLM_JOB_ID}"
    echo "Waiting for vLLM server (job $VLLM_JOB_ID)..."
    for i in $(seq 1 120); do
        if [ -f "$HOSTFILE" ]; then
            VLLM_HOST=$(cat "$HOSTFILE")
            break
        fi
        sleep 5
    done
    if [ -z "$VLLM_HOST" ]; then
        echo "ERROR: Could not find vLLM host after 10 min"
        exit 1
    fi
else
    VLLM_HOST="localhost"
fi

VLLM_URL="http://${VLLM_HOST}:8000/v1/chat/completions"
echo "vLLM server at: $VLLM_URL"

# Wait for vLLM to be ready
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s "${VLLM_URL%/chat/completions}/models" > /dev/null 2>&1; then
        echo "vLLM is ready."
        break
    fi
    sleep 5
done

# Patch peek.py to use the right URL
export VLLM_URL

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

echo "=== Autoresearch Data Quality Loop ==="
echo "Language: $LANG"
echo "Max iterations: $MAX_ITERATIONS"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Create results TSV
RESULTS="results.tsv"
if [ ! -f "$RESULTS" ]; then
    echo -e "commit\tval_bpb\tmemory_gb\tstatus\tdescription" > "$RESULTS"
fi

# ---------------------------------------------------------------------------
# Baseline: run training with no filter (all docs)
# ---------------------------------------------------------------------------

echo "=== Baseline (no filter) ==="

# Create a pass-through selected_doc_ids (all docs)
python3 -c "
import json, os, pyarrow.parquet as pq
from prepare_data import list_raw_parquet_files, get_lang_dir, get_eval_dir
lang = '$LANG'
eval_ids = set()
eval_path = os.path.join(get_eval_dir(lang), 'eval_doc_ids.json')
if os.path.exists(eval_path):
    eval_ids = set(json.load(open(eval_path)))
all_ids = []
for fp in list_raw_parquet_files(lang):
    pf = pq.ParquetFile(fp)
    for rg in range(pf.num_row_groups):
        ids = pf.read_row_group(rg).column('id').to_pylist()
        all_ids.extend([i for i in ids if i not in eval_ids])
    print(f'  {len(all_ids):,} docs...')
json.dump(all_ids, open(os.path.join(get_lang_dir(lang), 'selected_doc_ids.json'), 'w'))
print(f'Baseline: {len(all_ids):,} docs selected')
"

uv run train_data.py --lang "$LANG" > run.log 2>&1
BASELINE_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
BASELINE_MEM=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
BASELINE_MEM_GB=$(echo "$BASELINE_MEM / 1024" | bc -l | xargs printf "%.1f")

COMMIT=$(git rev-parse --short HEAD)
echo -e "${COMMIT}\t${BASELINE_BPB}\t${BASELINE_MEM_GB}\tkeep\tbaseline (no filter)" >> "$RESULTS"
echo "Baseline: val_bpb=$BASELINE_BPB mem=${BASELINE_MEM_GB}GB"

BEST_BPB="$BASELINE_BPB"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for ITER in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "=========================================="
    echo "=== Iteration $ITER / $MAX_ITERATIONS ==="
    echo "=========================================="

    SEED=$((42 + ITER * 1000))

    # Step 1: Peek (observe + synthesize + apply)
    echo "[iter $ITER] Peeking..."
    PEEK_ARGS="--lang $LANG --iteration $ITER --seed $SEED"
    if [ "$ITER" -gt 5 ]; then
        PEEK_ARGS="$PEEK_ARGS --from-filtered"
    fi

    # Temporarily patch VLLM_URL in peek.py via env
    VLLM_URL_ORIG=$(python3 -c "
import re
with open('peek.py') as f:
    m = re.search(r'VLLM_URL = \"(.+?)\"', f.read())
    print(m.group(1) if m else '')
")
    if [ "$VLLM_HOST" != "localhost" ]; then
        sed -i "s|$VLLM_URL_ORIG|$VLLM_URL|g" peek.py
    fi

    if ! uv run peek.py $PEEK_ARGS 2>&1 | tee "logs/peek_iter${ITER}.log"; then
        echo "[iter $ITER] Peek failed, skipping iteration"
        # Revert peek.py URL change
        if [ "$VLLM_HOST" != "localhost" ]; then
            sed -i "s|$VLLM_URL|$VLLM_URL_ORIG|g" peek.py
        fi
        continue
    fi

    # Revert peek.py URL change
    if [ "$VLLM_HOST" != "localhost" ]; then
        sed -i "s|$VLLM_URL|$VLLM_URL_ORIG|g" peek.py
    fi

    # Step 2: Commit
    LANG_CODE=$(echo "$LANG" | cut -d'_' -f1)
    git add "filter_${LANG_CODE}.py"
    DESCRIPTION=$(grep "^Problem:" "logs/peek_iter${ITER}.log" | head -1 | cut -c10-80 || echo "iteration $ITER")
    git commit -m "iter $ITER: $DESCRIPTION" 2>/dev/null || true
    COMMIT=$(git rev-parse --short HEAD)

    # Step 3: Filter
    echo "[iter $ITER] Filtering..."
    uv run filter.py --lang "$LANG" 2>&1 | tail -5

    # Step 4: Train
    echo "[iter $ITER] Training..."
    uv run train_data.py --lang "$LANG" > run.log 2>&1

    # Step 5: Read results
    VAL_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
    PEAK_MEM=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
    MEM_GB=$(echo "$PEAK_MEM / 1024" | bc -l | xargs printf "%.1f")

    if [ -z "$VAL_BPB" ]; then
        echo "[iter $ITER] Training crashed!"
        echo -e "${COMMIT}\t0.000000\t0.0\tcrash\t${DESCRIPTION}" >> "$RESULTS"
        git revert --no-edit HEAD 2>/dev/null || git reset --hard HEAD~1
        continue
    fi

    echo "[iter $ITER] val_bpb=$VAL_BPB (best=$BEST_BPB) mem=${MEM_GB}GB"

    # Step 6: Keep or revert
    IMPROVED=$(python3 -c "print('yes' if float('$VAL_BPB') < float('$BEST_BPB') else 'no')")
    if [ "$IMPROVED" = "yes" ]; then
        echo "[iter $ITER] KEEP — BPB improved from $BEST_BPB to $VAL_BPB"
        echo -e "${COMMIT}\t${VAL_BPB}\t${MEM_GB}\tkeep\t${DESCRIPTION}" >> "$RESULTS"
        BEST_BPB="$VAL_BPB"
    else
        echo "[iter $ITER] DISCARD — BPB did not improve ($VAL_BPB >= $BEST_BPB)"
        echo -e "${COMMIT}\t${VAL_BPB}\t${MEM_GB}\tdiscard\t${DESCRIPTION}" >> "$RESULTS"
        git revert --no-edit HEAD 2>/dev/null || git reset --hard HEAD~1
    fi
done

echo ""
echo "=== Done ==="
echo "Best val_bpb: $BEST_BPB"
echo "Results: $RESULTS"
cat "$RESULTS"
