#!/usr/bin/env bash
# T23 ms-swift 推理：启动 swift deploy -> 等待就绪 -> 跑统一 prompt suite -> 停止服务。
# 从仓库根目录运行：bash scripts/run_t23_all.sh
set -euo pipefail

ENV_NAME="gpp-swift-4.4.3"
PORT="${MS_SWIFT_PORT:-8000}"
MODEL_DIR="$PWD/assets/modelscope/qwen2.5-0.5b-instruct"
LOG_DIR="$PWD/docs/evidence/T23"
LOG="$LOG_DIR/t23-deploy.log"
BASE_URL="http://localhost:${PORT}"

echo "== T23 ms-swift infer/deploy =="
echo "== env: ${ENV_NAME}"
echo "== model: ${MODEL_DIR}"
mkdir -p "$LOG_DIR"

# 清理可能残留的 swift deploy 进程
pkill -f "swift deploy.*qwen2.5-0.5b-instruct" >/dev/null 2>&1 || true

cleanup() {
  echo "== stopping ms-swift deploy =="
  pkill -f "swift deploy.*qwen2.5-0.5b-instruct" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== starting swift deploy =="
CUDA_VISIBLE_DEVICES=0 \
nohup conda run --no-capture-output -n "${ENV_NAME}" swift deploy \
  --model "${MODEL_DIR}" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served_model_name qwen2.5-0.5b-instruct \
  --vllm_gpu_memory_utilization 0.85 \
  --vllm_max_model_len 2048 \
  --vllm_max_num_seqs 4 \
  > "${LOG}" 2>&1 &

SERVER_PID=$!
echo "== swift deploy pid=${SERVER_PID} log=${LOG} =="

echo "== waiting for ${BASE_URL}/v1/models =="
for i in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "== ms-swift ready after ${i}s =="
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: swift deploy process exited" >&2
    tail -50 "${LOG}" >&2 || true
    exit 1
  fi
  sleep 1
  if [[ "${i}" == "180" ]]; then
    echo "ERROR: ms-swift did not become ready in 180s" >&2
    tail -50 "${LOG}" >&2 || true
    exit 1
  fi
done

echo "== running client benchmark =="
conda run --no-capture-output -n gpp-core python -I \
  src/t23_ms_swift_infer/client_benchmark.py \
  --base-url "${BASE_URL}" \
  --max-tokens 16

echo "== T23 done =="
