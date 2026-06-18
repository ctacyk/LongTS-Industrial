#!/bin/bash
#
# HTSR-VL Benchmark — InternVL3 一键评测脚本
#
# 用法: bash run_server_benchmark.sh
#
# 自动串行完成 3 个模型的: vLLM 启动 → 推理 → 关闭 → 评测
# 全程无需手动干预，跑完直接看 results/

set -e

# ============================================================
# 配置区（按需修改）
# ============================================================
MODELS=(
    "OpenGVLab/InternVL3-2B"
    "OpenGVLab/InternVL3-9B"
    "OpenGVLab/InternVL3-14B"
)

PORT=8000
TP_SIZE=1                   # tensor-parallel，单卡=1，双卡=2
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=48
GPU_UTIL=0.95

BENCHMARK_DIR="MultiAgentTS/Data/benchmark_tb/LongTS_Industrial"
DATA_DIR="MultiAgentTS/Data/benchmark_tb"
RESULTS_DIR="results/benchmark_eval_v2"

# HuggingFace 模型缓存放到大磁盘，避免系统盘爆满
export HF_HOME="/root/autodl-tmp/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

# 推理用（本地 vLLM）
export VLM_BASE_URL="http://localhost:${PORT}/v1"
export VLM_API_KEY="EMPTY"

# 评测用（DashScope API, Judge + Embedding）
# 如需覆盖默认 key，取消下面注释并填入你的 key
# export JUDGE_API_KEY="sk-your-dashscope-key"

# ============================================================
# 主流程
# ============================================================
TOTAL=${#MODELS[@]}
echo "========================================================"
echo "  HTSR-VL Benchmark — InternVL3 Auto-Runner"
echo "========================================================"
echo "  Models:    ${MODELS[*]}"
echo "  TP size:   ${TP_SIZE}"
echo "  Port:      ${PORT}"
echo "  GPU util:  ${GPU_UTIL}"
echo "========================================================"
echo ""

START_ALL=$(date +%s)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    IDX=$((i + 1))
    SAFE_NAME="${MODEL//\//--}"

    echo ""
    echo "========================================================"
    echo "  [$IDX/$TOTAL] $MODEL"
    echo "========================================================"

    # --- 1. 启动 vLLM ---
    echo "[$(date +%H:%M:%S)] Starting vLLM server for $MODEL ..."
    vllm serve "$MODEL" \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --gpu-memory-utilization $GPU_UTIL \
        --trust-remote-code \
        --port $PORT \
        --disable-log-requests \
        > "vllm_${SAFE_NAME}.log" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"

    # --- 2. 等待 vLLM 就绪 ---
    echo "[$(date +%H:%M:%S)] Waiting for vLLM to be ready ..."
    MAX_WAIT=600
    WAITED=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
        WAITED=$((WAITED + 5))
        if [ $WAITED -ge $MAX_WAIT ]; then
            echo "  ERROR: vLLM failed to start within ${MAX_WAIT}s. Check vllm_${SAFE_NAME}.log"
            kill $VLLM_PID 2>/dev/null || true
            continue 2
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "  ERROR: vLLM process died. Check vllm_${SAFE_NAME}.log"
            continue 2
        fi
    done
    echo "[$(date +%H:%M:%S)] vLLM ready! (waited ${WAITED}s)"

    # --- 3. 运行推理 ---
    echo "[$(date +%H:%M:%S)] Running inference ..."
    START_INF=$(date +%s)
    python run_benchmark_eval.py \
        --models "$MODEL" \
        --rpm 99999 \
        --sequential \
        --benchmark-dir "$BENCHMARK_DIR" \
        --data-dir "$DATA_DIR" \
        --results-dir "$RESULTS_DIR"
    END_INF=$(date +%s)
    INF_MIN=$(( (END_INF - START_INF) / 60 ))
    echo "[$(date +%H:%M:%S)] Inference + eval done in ${INF_MIN} min"

    # --- 4. 关闭 vLLM ---
    echo "[$(date +%H:%M:%S)] Stopping vLLM ..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 5
    echo "  vLLM stopped."
done

END_ALL=$(date +%s)
TOTAL_MIN=$(( (END_ALL - START_ALL) / 60 ))

echo ""
echo "========================================================"
echo "  ALL DONE! Total time: ${TOTAL_MIN} min"
echo "========================================================"
echo ""
echo "Results:"
for MODEL in "${MODELS[@]}"; do
    SAFE_NAME="${MODEL//\//--}"
    EVAL_FILE="${RESULTS_DIR}/${MODEL//\//--}/eval_results.json"
    if [ -f "$EVAL_FILE" ]; then
        echo "  $MODEL -> OK"
    else
        EVAL_FILE2="${RESULTS_DIR}/${MODEL//\//--}/eval_results.json"
        if [ -f "$EVAL_FILE2" ]; then
            echo "  $MODEL -> OK"
        else
            echo "  $MODEL -> MISSING (check logs)"
        fi
    fi
done
