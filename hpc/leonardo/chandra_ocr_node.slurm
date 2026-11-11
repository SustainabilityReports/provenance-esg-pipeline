#!/bin/bash
#SBATCH --job-name=parser1-ocr
# NOTE: no #SBATCH --account here: pass --account and -J on the sbatch
# command line, so the same template works across allocations.
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --output=chandra_%j.out
#SBATCH --error=chandra_%j.err

# ============================================================================
# Chandra OCR 2 batch job on Leonardo Booster (single node, 4x A100)
# ============================================================================
#
# vLLM runs inside the official vllm/vllm-openai:v0.17.0 Singularity container
# (required for chandra-ocr-2's qwen3_5 architecture, not in PyPI wheels).
# The workhorse runs on the host venv and talks to vLLM via HTTP.
#
# Usage:
#   sbatch --account=<ACCOUNT> chandra_ocr_node.slurm                 # all PDFs in $WORK/pdf
#   sbatch --account=<ACCOUNT> chandra_ocr_node.slurm /path/to/pdfs   # one folder
#
# Expects in $CHANDRA_WORK_DIR (default $WORK): containers/vllm-openai-v0.17.0.sif,
# chandra-env/ (host venv with this repository installed), hf_cache/ (model
# snapshot of datalab-to/chandra-ocr-2) and a checkout of this repository in
# provenance-esg-pipeline/.
#
# ============================================================================

echo "========================================"
echo "  Chandra OCR 2 on Leonardo Booster"
echo "  Single Node Configuration (1 node, 4 GPUs)"
echo "========================================"
echo ""
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_NODELIST"
echo "Num Nodes:    $SLURM_NNODES"
echo "Tasks/Node:   $SLURM_NTASKS_PER_NODE"
echo "Start time:   $(date)"
echo ""

# ============================================================================
# Configuration
# ============================================================================

PDF_INPUT="${1:-$WORK/pdf}"

MODEL_PATH="datalab-to/chandra-ocr-2"
VLLM_MODEL_NAME="chandra"

BATCH_SIZE=4
WORKERS_PER_GPU=12
MAX_OUTPUT_TOKENS=16000

# CHANDRA_WORK_DIR is exported by the launcher (sbatch --export) so the
# script works regardless of which project $WORK points at by default.
WORK_DIR="${CHANDRA_WORK_DIR:-$WORK}"
SCRATCH_DIR="$CINECA_SCRATCH"
CACHE_DIR="$CINECA_SCRATCH/cache"
VLLM_SIF="$WORK_DIR/containers/vllm-openai-v0.17.0.sif"

echo "PDF Input:    $PDF_INPUT"
echo "Work Dir:     $WORK_DIR"
echo "Scratch Dir:  $SCRATCH_DIR"
echo "vLLM image:   $VLLM_SIF"
echo ""

if [ ! -f "$VLLM_SIF" ]; then
    echo "ERROR: Singularity image not found at $VLLM_SIF"
    echo "       Pull with: singularity pull $VLLM_SIF docker://vllm/vllm-openai:v0.17.0"
    exit 1
fi

# ============================================================================
# Environment Setup (host side - for workhorse)
# ============================================================================

echo "Setting up host environment..."

module purge
module load python/3.11
module load cuda/12.1

if [ -d "$WORK_DIR/chandra-env" ]; then
    source "$WORK_DIR/chandra-env/bin/activate"
    echo "[OK] Activated virtual environment"
else
    echo "ERROR: Virtual environment not found at $WORK_DIR/chandra-env"
    exit 1
fi

export HF_HOME="$WORK_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$CACHE_DIR/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ============================================================================
# Get Node List
# ============================================================================

NODES=($(scontrol show hostnames $SLURM_NODELIST))
NUM_NODES=${#NODES[@]}
NUM_GPUS_PER_NODE=4
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo ""
echo "Nodes allocated: ${NODES[@]}"
echo "Total GPUs: $TOTAL_GPUS"
echo ""

# ============================================================================
# Create vLLM launcher script (runs on each task, launches Singularity)
# ============================================================================

VLLM_LAUNCHER="$WORK_DIR/vllm_launcher_${SLURM_JOB_ID}.sh"
cat > "$VLLM_LAUNCHER" << 'LAUNCHER_EOF'
#!/bin/bash
# vLLM launcher - runs on each task (one per GPU), launches Singularity container

MODEL_PATH="$VLLM_MODEL_PATH"
VLLM_MODEL_NAME="$VLLM_MODEL_NAME"
WORK_DIR="$VLLM_WORK_DIR"
VLLM_SIF="$VLLM_SIF_PATH"

GPU_ID=$SLURM_LOCALID
PORT=$((8000 + GPU_ID))
GLOBAL_ID=$SLURM_PROCID

echo "[Task $GLOBAL_ID] Starting vLLM on $(hostname), GPU $GPU_ID, port $PORT"

# vllm-openai image expects HF cache at /root/.cache/huggingface
# Bind our hf_cache there, and HF_HUB_OFFLINE=1 makes it use the cached snapshot
exec singularity exec --nv \
    --env CUDA_VISIBLE_DEVICES=$GPU_ID \
    --env HF_HOME=/hf_cache \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    -B "$WORK_DIR/hf_cache:/hf_cache" \
    "$VLLM_SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$VLLM_MODEL_NAME" \
        --host 0.0.0.0 \
        --port $PORT \
        --dtype bfloat16 \
        --max-model-len 18000 \
        --max-num-seqs 64 \
        --max-num-batched-tokens 8192 \
        --gpu-memory-utilization 0.85 \
        --enable-prefix-caching \
        --no-enforce-eager \
        --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'
LAUNCHER_EOF
chmod +x "$VLLM_LAUNCHER"

# ============================================================================
# Start vLLM Servers on All Nodes using srun
# ============================================================================

echo "Starting vLLM servers on all $TOTAL_GPUS GPUs (via Singularity)..."

export VLLM_MODEL_PATH="$MODEL_PATH"
export VLLM_MODEL_NAME="$VLLM_MODEL_NAME"
export VLLM_WORK_DIR="$WORK_DIR"
export VLLM_SIF_PATH="$VLLM_SIF"

srun --ntasks=$TOTAL_GPUS --ntasks-per-node=4 --cpus-per-task=8 \
    bash "$VLLM_LAUNCHER" &

VLLM_PID=$!

echo ""
echo "  Waiting for vLLM servers to initialize (this takes several minutes)..."
echo "  Models load sequentially on shared filesystem, please be patient..."
sleep 600

# ============================================================================
# Verify Servers with Graceful Degradation
# ============================================================================

echo ""
echo "Verifying vLLM servers (with graceful degradation)..."

declare -a READY_GPUS=()
declare -a READY_ENDPOINTS=()

for node_idx in $(seq 0 $((NUM_NODES - 1))); do
    node=${NODES[$node_idx]}
    echo "  Checking $node..."

    for gpu_id in $(seq 0 $((NUM_GPUS_PER_NODE - 1))); do
        port=$((8000 + gpu_id))
        global_gpu_id=$((node_idx * NUM_GPUS_PER_NODE + gpu_id))

        ready=false
        for attempt in $(seq 1 90); do
            if curl -s --connect-timeout 5 "http://${node}:${port}/v1/models" > /dev/null 2>&1; then
                echo "    [OK] GPU $global_gpu_id ($node:$port) ready (attempt $attempt)"
                READY_GPUS+=($global_gpu_id)
                READY_ENDPOINTS+=("$node:$port")
                ready=true
                break
            fi
            sleep 10
        done

        if [ "$ready" = false ]; then
            echo "    [WARN] GPU $global_gpu_id ($node:$port) failed to start after 15 minutes - skipping"
        fi
    done
done

NUM_READY=${#READY_GPUS[@]}
if [ $NUM_READY -eq 0 ]; then
    echo ""
    echo "ERROR: No vLLM servers available. Exiting."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo ""
echo "[OK] $NUM_READY/$TOTAL_GPUS vLLM servers ready - continuing with available GPUs"

# ============================================================================
# Build Endpoint Configuration (only ready GPUs)
# ============================================================================

CONFIG_FILE="$SCRATCH_DIR/chandra_config_${SLURM_JOB_ID}.json"

ENDPOINTS_JSON="{"
first=true
for i in $(seq 0 $((NUM_READY - 1))); do
    gpu_id=${READY_GPUS[$i]}
    endpoint=${READY_ENDPOINTS[$i]}

    if [ "$first" = true ]; then
        first=false
    else
        ENDPOINTS_JSON+=","
    fi
    ENDPOINTS_JSON+="\"leonardo-gpu-$gpu_id\": \"$endpoint\""
done
ENDPOINTS_JSON+="}"

cat > "$CONFIG_FILE" << EOF
{
    "pdf_path": "$PDF_INPUT",
    "endpoints": $ENDPOINTS_JSON,
    "file_filter": {
        "enabled": false,
        "prefix": [],
        "suffix": ""
    },
    "processing": {
        "workers_per_gpu": $WORKERS_PER_GPU,
        "fresh_restart_interval": 0
    }
}
EOF

echo "Created config with $NUM_READY endpoints: $CONFIG_FILE"
echo ""
echo "Endpoints:"
cat "$CONFIG_FILE" | python -c "import sys,json; d=json.load(sys.stdin); [print(f'  {k}: {v}') for k,v in d['endpoints'].items()]"

# ============================================================================
# Run Processing
# ============================================================================

echo ""
echo "========================================"
echo "  Starting Chandra OCR processing"
echo "  ($NUM_READY GPUs, ~$((NUM_READY * 80)) pages/hour estimated)"
echo "========================================"
echo ""

export PDF_INPUT="$PDF_INPUT"
export CHANDRA_CONFIG="$CONFIG_FILE"
export VLLM_API_KEY="EMPTY"
export IMAGE_DPI=200
export MIN_IMAGE_DIM=2048
# Loosen the worker's "no PDF completed" stall timeout for very long reports
# (300-900 page documents need >1800 s to finish their first document).
# Honors a value passed via `sbatch --export` (single-giant folders use 5400); else 3600.
export CHANDRA_STALL_TIMEOUT_SECONDS="${CHANDRA_STALL_TIMEOUT_SECONDS:-3600}"

export PYTHONPATH="$WORK_DIR/provenance-esg-pipeline:$PYTHONPATH"

echo "Testing Python connectivity to endpoints..."
python -c "
import requests
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
for name, endpoint in config['endpoints'].items():
    url = f'http://{endpoint}/v1/models'
    try:
        resp = requests.get(url, timeout=5)
        print(f'  {name} ({endpoint}): {resp.status_code}')
    except Exception as e:
        print(f'  {name} ({endpoint}): ERROR - {e}')
"

cd "$WORK_DIR/provenance-esg-pipeline"

# ============================================================================
# Log-heartbeat Watchdog (Layer 1: defense-in-depth, independent of workhorse)
# ============================================================================
# Liveness signal = MTIME of the job's stderr log ($WATCH_LOG). The workhorse
# emits a "Status: ... N done (X/min)" heartbeat line every ~60s for its WHOLE
# lifetime: OCR, per-document merge, AND batch cleanup. So as long as the
# process is alive the log keeps growing; if the log goes silent for
# $WATCHDOG_STALL_SECONDS the engine is genuinely hung (the Apr-2026 vLLM-crash
# case) and we scancel.
#
# Why mtime and NOT _full.json / *.json counts (two prior false-kill incidents):
#   - _full.json count: lands at *document* granularity. Very long documents
#     (300-900 pages) don't complete a single document inside the grace window,
#     so a healthy job would show 0 _full.json and get killed.
#   - any *.json count: rises during OCR but PLATEAUS/DROPS during the final
#     merge phase because cleanup_batch_files() deletes _batches/ after each
#     merge, so a job quietly finalizing writes "no new json" and would be
#     killed mid-merge. Log mtime is immune to both: it tracks the process,
#     not its output files.
# Blast radius cap: ~WATCHDOG_STALL_SECONDS of wall clock ≈ 24 core-hr on 1 node.

OUTPUT_DIR="${PDF_INPUT}-chandra"
# The worker reads OCR_DIR for its output location.
export OCR_DIR="$OUTPUT_DIR"
# Heartbeat log = this job's OWN stderr. `#SBATCH --error=chandra_%j.err` is
# relative to the SUBMISSION directory, so derive the path from
# SLURM_SUBMIT_DIR rather than assuming where the job was submitted from: a
# watchdog that watches a non-existent file would kill healthy jobs.
WATCH_LOG="${SLURM_SUBMIT_DIR:-$PWD}/chandra_${SLURM_JOB_ID}.err"
WATCHDOG_POLL_SECONDS=60
WATCHDOG_STALL_SECONDS=900        # 15 min of log silence = genuine hang (heartbeat is ~60s, so 15× margin)
WATCHDOG_FIRST_OUTPUT_GRACE=1800  # 30 min for the log to start (vLLM warmup + first workhorse heartbeat)

mkdir -p "$OUTPUT_DIR"

watchdog_loop() {
    local last_mtime=0
    local last_progress_time start_time
    start_time=$(date +%s)
    last_progress_time=$start_time
    while true; do
        sleep "$WATCHDOG_POLL_SECONDS"
        local mtime now stall
        # stat the heartbeat log; 0 if it doesn't exist yet.
        mtime=$(stat -c %Y "$WATCH_LOG" 2>/dev/null || echo 0)
        now=$(date +%s)
        if [ "$mtime" -gt "$last_mtime" ]; then
            last_mtime=$mtime
            last_progress_time=$now
            continue
        fi
        stall=$(( now - last_progress_time ))
        # Grace period: before the log ever updates, use the looser threshold.
        if [ "$last_mtime" -eq 0 ]; then
            if [ $(( now - start_time )) -ge "$WATCHDOG_FIRST_OUTPUT_GRACE" ]; then
                # "I cannot find my own log" is a MONITORING failure, not
                # evidence the job is hung — and killing a healthy job because
                # the watcher is misconfigured is strictly worse than not
                # watching it.
                # Distinguish the two cases by whether the file exists at all.
                if [ ! -e "$WATCH_LOG" ]; then
                    echo "[WATCHDOG] DISABLED: heartbeat log $WATCH_LOG does not exist after $(( now - start_time ))s. This is a watchdog misconfiguration, NOT a stall — refusing to scancel $SLURM_JOB_ID. The job continues UNWATCHED; fix WATCH_LOG." >&2
                    return 0
                fi
                echo "[WATCHDOG] STALL: log $WATCH_LOG exists but never updated in $(( now - start_time ))s (grace ${WATCHDOG_FIRST_OUTPUT_GRACE}s) — scancelling $SLURM_JOB_ID" >&2
                scancel "$SLURM_JOB_ID"
                return 1
            fi
            continue
        fi
        # Steady state: log has updated at least once; apply the tight threshold.
        if [ "$stall" -ge "$WATCHDOG_STALL_SECONDS" ]; then
            echo "[WATCHDOG] STALL: no log heartbeat in ${stall}s (threshold ${WATCHDOG_STALL_SECONDS}s) — scancelling $SLURM_JOB_ID" >&2
            scancel "$SLURM_JOB_ID"
            return 1
        fi
    done
}

python scripts/run_pipeline.py "$CONFIG_FILE" &
WORKHORSE_PID=$!

watchdog_loop &
WATCHDOG_PID=$!

wait $WORKHORSE_PID
PROCESS_EXIT_CODE=$?

# Stop the watchdog now that the workhorse has exited (normally or via scancel).
kill $WATCHDOG_PID 2>/dev/null
wait $WATCHDOG_PID 2>/dev/null

# ============================================================================
# Timeout Retry Loop (up to 2 retries for timed-out batches)
# ============================================================================

TIMEOUT_LOG="$OUTPUT_DIR/_timeout_log.json"

for retry in 1 2; do
    if [ -f "$TIMEOUT_LOG" ]; then
        TIMEOUT_COUNT=$(python -c "
import json
try:
    with open('$TIMEOUT_LOG') as f:
        t = json.load(f)
    print(len(t))
except:
    print(0)
")
        if [ "$TIMEOUT_COUNT" -gt 0 ]; then
            echo ""
            echo "========================================"
            echo "  Retry $retry: Re-processing $TIMEOUT_COUNT timed-out batches"
            echo "========================================"
            echo ""

            python -c "
import json
try:
    with open('$TIMEOUT_LOG', 'w') as f:
        json.dump({}, f)
    print('Cleared timeout log for retry')
except Exception as e:
    print(f'Error clearing timeout log: {e}')
"

            python scripts/run_pipeline.py "$CONFIG_FILE"
            PROCESS_EXIT_CODE=$?
        else
            echo "No timed-out batches to retry"
            break
        fi
    else
        echo "No timeout log found, skipping retry"
        break
    fi
done

# ============================================================================
# Cleanup
# ============================================================================

echo ""
echo "Cleaning up..."

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

rm -f "$VLLM_LAUNCHER"
rm -f "$CONFIG_FILE"

echo ""
echo "========================================"
echo "  Job Complete"
echo "========================================"
echo "End time:     $(date)"
echo "Exit code:    $PROCESS_EXIT_CODE"
echo ""

if [ -d "$OUTPUT_DIR" ]; then
    echo "Results:      $OUTPUT_DIR"
    echo "Total files:  $(find "$OUTPUT_DIR" -name "*_full.json" | wc -l)"
fi

exit $PROCESS_EXIT_CODE
