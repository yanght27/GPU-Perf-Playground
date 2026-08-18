#!/usr/bin/env bash
# T20 vLLM Serving：启动 vLLM 容器 -> 等待就绪 -> 跑统一 prompt suite -> 清理容器。
# 从仓库根目录运行：bash scripts/run_t20_all.sh
set -euo pipefail

IMAGE="vllm/vllm-openai:v0.27.1"
CONTAINER="gpp-vllm-t20"
PORT="${VLLM_PORT:-8000}"
MODEL_DIR="$PWD/assets/modelscope/qwen2.5-0.5b-instruct"
BASE_URL="http://localhost:${PORT}"

echo "== T20 vLLM serving =="
echo "== image: ${IMAGE}"
echo "== model: ${MODEL_DIR}"

# 清理可能残留的旧容器
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

cleanup() {
  echo "== stopping vLLM container =="
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== starting vLLM container =="
docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  -v "${MODEL_DIR}:/models/qwen2.5-0.5b-instruct:ro" \
  -p "${PORT}:8000" \
  --ipc=host \
  "${IMAGE}" \
  --model /models/qwen2.5-0.5b-instruct \
  --served-model-name qwen2.5-0.5b-instruct \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --attention-backend FLASH_ATTN

echo "== waiting for ${BASE_URL}/v1/models =="
for i in $(seq 1 120); do
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "== vLLM ready after ${i}s =="
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: vLLM container exited" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -50 || true
    exit 1
  fi
  sleep 1
  if [[ "${i}" == "120" ]]; then
    echo "ERROR: vLLM did not become ready in 120s" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -50 || true
    exit 1
  fi
done

echo "== running client benchmark =="
conda run --no-capture-output -n gpp-core python -I \
  src/t20_vllm_serving/client_benchmark.py \
  --base-url "${BASE_URL}" \
  --max-tokens 16

echo "== T20 done =="
