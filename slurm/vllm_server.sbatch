#!/bin/bash
#SBATCH --job-name=vllm-qwen
#SBATCH --output=logs/vllm_%j.out
#SBATCH --error=logs/vllm_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

mkdir -p logs

echo "Starting vLLM server on $(hostname):8000..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Write hostname so the loop job knows where to connect
echo "$(hostname)" > /tmp/vllm_host_${SLURM_JOB_ID}

vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --host 0.0.0.0 \
    --port 8000
