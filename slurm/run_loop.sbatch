#!/bin/bash
#SBATCH --job-name=autoresearch-data
#SBATCH --output=logs/loop_%j.out
#SBATCH --error=logs/loop_%j.out
#SBATCH --time=24:00:00
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

# 8 GPUs on one node:
#   GPU 0-3: vLLM server (Qwen3.5-35B-A3B, TP=4)
#   GPU 7:   training (train.py)

LANG="${1:-dan_Latn}"
MAX_ITERATIONS="${2:-100}"
LANG_CODE=$(echo "$LANG" | cut -d'_' -f1)

mkdir -p logs

# File lock for git operations (multiple languages may run in parallel)
GIT_LOCK="/tmp/autocurate_git.lock"
git_locked() {
    # Run a git command under flock to prevent concurrent access
    flock -w 60 "$GIT_LOCK" "$@"
}

echo "$(date) | Installing dependencies..."
uv sync 2>&1
echo "$(date) | Dependencies ready."
echo ""
echo "$(date) | Autoresearch Data Quality Loop"
echo "Language: $LANG"
echo "Max iterations: $MAX_ITERATIONS"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo ""

# ---------------------------------------------------------------------------
# Start vLLM server on GPU 0
# ---------------------------------------------------------------------------

# vLLM needs its own venv (torch version conflicts with training)
VLLM_VENV=".venv-vllm"
if [ ! -f "$VLLM_VENV/bin/vllm" ]; then
    echo "$(date) | Creating vLLM venv..."
    uv venv "$VLLM_VENV" --python 3.10 2>&1
    echo "$(date) | Installing vLLM (this takes a few minutes)..."
    VIRTUAL_ENV="$VLLM_VENV" uv pip install vllm 2>&1 | tail -3
fi
# Patch vLLM to skip SageMaker bootstrap (crashes on non-SageMaker hosts)
SAGEMAKER_ROUTER="$VLLM_VENV/lib/python3.10/site-packages/vllm/entrypoints/sagemaker/api_router.py"
if grep -q "sagemaker_standards.bootstrap" "$SAGEMAKER_ROUTER" 2>/dev/null; then
    echo "$(date) | Patching vLLM to skip SageMaker bootstrap..."
    sed -i 's/return sagemaker_standards.bootstrap(app)/return app/' "$SAGEMAKER_ROUTER"
fi

echo "$(date) | Starting vLLM server on GPU 0-3 (TP=4)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-frontend-multiprocessing \
    > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "$(date) | Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "$(date) | vLLM is ready (took ${i}x5s)."
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "$(date) | ERROR: vLLM process died. Check logs/vllm_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "$(date) | ERROR: vLLM not ready after 10 min"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Export for peek.py
export VLLM_URL="http://localhost:8000/v1/chat/completions"
export JUDGE_MODEL="Qwen/Qwen3.5-35B-A3B"

# Training uses GPU 7
export CUDA_VISIBLE_DEVICES=7

# Cleanup vLLM on exit
trap "kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# ---------------------------------------------------------------------------
# Create results TSV
# ---------------------------------------------------------------------------

RESULTS="results_${LANG_CODE}.tsv"
if [ ! -f "$RESULTS" ]; then
    printf "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n" > "$RESULTS"
fi

# ---------------------------------------------------------------------------
# Baseline: train on all docs (no filter)
# ---------------------------------------------------------------------------

echo ""
# Ensure tokenizer exists
echo "$(date) | Setting up tokenizer..."
PYTHONUNBUFFERED=1 uv run setup_tokenizer.py --lang "$LANG" 2>&1

# Skip baseline if already done
if grep -q "baseline" "$RESULTS" 2>/dev/null; then
    echo "$(date) | Baseline already in results.tsv, skipping"
    BEST_BPB=$(grep "keep" "$RESULTS" | tail -1 | awk '{print $2}')
    echo "$(date) | Best BPB so far: $BEST_BPB"
else
echo "$(date) | === Baseline (no filter) ==="

# Baseline: train with empty filter (clean+filter on the fly, no rules yet)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=7 uv run train.py --lang "$LANG" 2>&1 | tee /tmp/train_${LANG_CODE}.log
BASELINE_BPB=$(grep "^val_bpb:" /tmp/train_${LANG_CODE}.log | awk '{print $2}')
BASELINE_MEM=$(grep "^peak_vram_mb:" /tmp/train_${LANG_CODE}.log | awk '{print $2}')

if [ -z "$BASELINE_BPB" ]; then
    echo "$(date) | ERROR: Baseline training failed!"
    tail -20 /tmp/train_${LANG_CODE}.log
    exit 1
fi

BASELINE_MEM_GB=$(python3 -c "print(f'{float(\"$BASELINE_MEM\")/1024:.1f}')")
COMMIT=$(git_locked git rev-parse --short HEAD)
printf "%s\t%s\t%s\tkeep\tbaseline (no filter)\n" "$COMMIT" "$BASELINE_BPB" "$BASELINE_MEM_GB" >> "$RESULTS"
echo "$(date) | Baseline: val_bpb=$BASELINE_BPB mem=${BASELINE_MEM_GB}GB"

BEST_BPB="$BASELINE_BPB"
fi  # end baseline skip check

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for ITER in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "$(date) | ========== Iteration $ITER / $MAX_ITERATIONS =========="

    SEED=$((42 + ITER * 1000))

    # Step 1: Peek
    PEEK_ARGS="--lang $LANG --iteration $ITER --seed $SEED"

    # Reset filter file to last committed state before peeking
    git_locked git checkout -- "filter_${LANG_CODE}.py" 2>/dev/null || true

    echo "$(date) | [iter $ITER] Peeking..."
    CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 uv run peek.py $PEEK_ARGS 2>&1
    PEEK_EXIT=$?

    if [ $PEEK_EXIT -ne 0 ]; then
        echo "$(date) | [iter $ITER] Peek failed (exit $PEEK_EXIT), skipping"
        # Ensure clean state after failure
        git_locked git checkout -- "filter_${LANG_CODE}.py" 2>/dev/null || true
        continue
    fi

    # Step 2: Commit
    if ! git diff --quiet "filter_${LANG_CODE}.py" 2>/dev/null; then
        DESCRIPTION="iteration $ITER"
        [ -z "$DESCRIPTION" ] && DESCRIPTION="iteration $ITER"
        git_locked git add "filter_${LANG_CODE}.py"
        git_locked git commit -m "iter $ITER [$LANG_CODE]: $DESCRIPTION" --no-verify 2>/dev/null || true
    else
        echo "$(date) | [iter $ITER] No changes to filter, skipping"
        continue
    fi
    COMMIT=$(git_locked git rev-parse --short HEAD)

    # Step 3: Train (filter happens on the fly inside the dataloader)
    echo "$(date) | [iter $ITER] Training..."
    CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 uv run train.py --lang "$LANG" 2>&1 | tee /tmp/train_${LANG_CODE}.log

    # Step 5: Read results
    VAL_BPB=$(grep "^val_bpb:" /tmp/train_${LANG_CODE}.log | awk '{print $2}')
    PEAK_MEM=$(grep "^peak_vram_mb:" /tmp/train_${LANG_CODE}.log | awk '{print $2}')

    if [ -z "$VAL_BPB" ]; then
        echo "$(date) | [iter $ITER] Training crashed!"
        tail -10 /tmp/train_${LANG_CODE}.log
        printf "%s\t0.000000\t0.0\tcrash\t%s\n" "$COMMIT" "$DESCRIPTION" >> "$RESULTS"
        git_locked git checkout HEAD~1 -- "filter_${LANG_CODE}.py" 2>/dev/null
        git_locked git add "filter_${LANG_CODE}.py"
        git_locked git commit -m "revert iter $ITER [$LANG_CODE]" --no-verify 2>/dev/null || true
        continue
    fi

    MEM_GB=$(python3 -c "print(f'{float(\"$PEAK_MEM\")/1024:.1f}')")
    echo "$(date) | [iter $ITER] val_bpb=$VAL_BPB (best=$BEST_BPB) mem=${MEM_GB}GB"

    # Step 6: Keep or revert
    IMPROVED=$(python3 -c "print('yes' if float('$VAL_BPB') < float('$BEST_BPB') else 'no')")
    if [ "$IMPROVED" = "yes" ]; then
        STATUS="keep"
        echo "$(date) | [iter $ITER] KEEP — BPB improved $BEST_BPB → $VAL_BPB"
        printf "%s\t%s\t%s\tkeep\t%s\n" "$COMMIT" "$VAL_BPB" "$MEM_GB" "$DESCRIPTION" >> "$RESULTS"
        BEST_BPB="$VAL_BPB"
    else
        STATUS="discard"
        echo "$(date) | [iter $ITER] DISCARD — BPB $VAL_BPB >= $BEST_BPB"
        printf "%s\t%s\t%s\tdiscard\t%s\n" "$COMMIT" "$VAL_BPB" "$MEM_GB" "$DESCRIPTION" >> "$RESULTS"
        git_locked git checkout HEAD~1 -- "filter_${LANG_CODE}.py" 2>/dev/null
        git_locked git add "filter_${LANG_CODE}.py"
        git_locked git commit -m "revert iter $ITER [$LANG_CODE]" --no-verify 2>/dev/null || true
    fi

    # Log full details to history
    FUNC_CODE=$(python3 -c "
import json, re
log = open('logs/peek_${LANG_CODE}_iter${ITER}.log').read()
# Extract the function code from the log
m = re.search(r'## Function\s*\n\`\`\`python\s*\n(.*?)\`\`\`', log, re.DOTALL)
code = m.group(1).strip() if m else ''
print(json.dumps({
    'iteration': $ITER,
    'commit': '$COMMIT',
    'status': '$STATUS',
    'val_bpb': float('$VAL_BPB'),
    'best_bpb': float('$BEST_BPB'),
    'description': '$DESCRIPTION',
    'code': code,
    'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
}))
" 2>/dev/null)
    [ -n "$FUNC_CODE" ] && echo "$FUNC_CODE" >> peek_history_${LANG_CODE}.jsonl
done

echo ""
echo "$(date) | === Done ==="
echo "Best val_bpb: $BEST_BPB"
cat "$RESULTS"
