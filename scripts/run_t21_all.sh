#!/usr/bin/env bash
# T21 SGLang Serving：启动 SGLang 容器 -> 等待就绪 -> 跑统一 prompt suite -> 清理容器。
# 从仓库根目录运行：bash scripts/run_t21_all.sh
set -euo pipefail

IMAGE="lmsysorg/sglang:v0.5.17"
CONTAINER="gpp-sglang-t21"
PORT="${SGLANG_PORT:-30000}"
MODEL_DIR="$PWD/assets/modelscope/qwen2.5-0.5b-instruct"
BASE_URL="http://localhost:${PORT}"

echo "== T21 SGLang serving =="
echo "== image: ${IMAGE}"
echo "== model: ${MODEL_DIR}"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

cleanup() {
  echo "== stopping SGLang container =="
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== starting SGLang container =="
docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -v "${MODEL_DIR}:/models/qwen2.5-0.5b-instruct:ro" \
  -p "${PORT}:30000" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path /models/qwen2.5-0.5b-instruct \
  --host 0.0.0.0 \
  --port 30000

echo "== waiting for ${BASE_URL}/v1/models =="
for i in $(seq 1 120); do
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "== SGLang ready after ${i}s =="
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: SGLang container exited" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -50 || true
    exit 1
  fi
  sleep 1
  if [[ "${i}" == "120" ]]; then
    echo "ERROR: SGLang did not become ready in 120s" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -50 || true
    exit 1
  fi
done

echo "== running client benchmark =="
conda run --no-capture-output -n gpp-core python -I \
  src/t21_sglang_serving/client_benchmark.py \
  --base-url "${BASE_URL}" \
  --max-tokens 16

echo "== T21 done =="
