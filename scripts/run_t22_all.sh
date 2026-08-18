#!/usr/bin/env bash
# T22 TensorRT-LLM Serving：convert checkpoint -> build engine -> start server -> 跑统一 prompt suite -> 清理。
# 从仓库根目录运行：bash scripts/run_t22_all.sh
set -euo pipefail

IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
CONTAINER="gpp-trtllm-t22"
PORT="${TRTLLM_PORT:-8000}"
MODEL_DIR="$PWD/assets/modelscope/qwen2.5-0.5b-instruct"
WORK="$PWD/caches/trt-llm/t22"
CKPT_DIR="$WORK/checkpoint"
ENGINE_DIR="$WORK/engines"
BASE_URL="http://localhost:${PORT}"

echo "== T22 TensorRT-LLM serving =="
echo "== image: ${IMAGE}"
echo "== model: ${MODEL_DIR}"
echo "== work: ${WORK}"

mkdir -p "$WORK"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

cleanup() {
  echo "== stopping TRT-LLM container =="
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== [1/4] convert checkpoint =="
docker run --rm \
  --gpus all \
  -v "${MODEL_DIR}:/models/qwen2.5-0.5b-instruct:ro" \
  -v "${WORK}:/workspace/t22" \
  --entrypoint python3 \
  "${IMAGE}" \
  /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir /models/qwen2.5-0.5b-instruct \
  --output_dir /workspace/t22/checkpoint \
  --dtype bfloat16

echo "== [2/4] build engine =="
docker run --rm \
  --gpus all \
  -v "${WORK}:/workspace/t22" \
  --entrypoint trtllm-build \
  "${IMAGE}" \
  --checkpoint_dir /workspace/t22/checkpoint \
  --output_dir /workspace/t22/engines \
  --gpt_attention_plugin bfloat16 \
  --gemm_plugin bfloat16 \
  --max_batch_size 4 \
  --max_input_len 512 \
  --max_seq_len 1024

echo "== [3/4] start TRT-LLM server =="
docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  -v "${MODEL_DIR}:/models/qwen2.5-0.5b-instruct:ro" \
  -v "${WORK}:/workspace/t22" \
  -p "${PORT}:8000" \
  --entrypoint trtllm-serve \
  "${IMAGE}" \
  /workspace/t22/engines \
  --tokenizer /models/qwen2.5-0.5b-instruct \
  --host 0.0.0.0 \
  --port 8000

echo "== waiting for ${BASE_URL}/v1/models =="
for i in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/v1/models" >/dev/null 2>&1; then
    echo "== TRT-LLM ready after ${i}s =="
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: TRT-LLM container exited" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -50 || true
    exit 1
  fi
  sleep 1
  if [[ "${i}" == "180" ]]; then
    echo "ERROR: TRT-LLM did not become ready in 180s" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -50 || true
    exit 1
  fi
done

echo "== [4/4] run client benchmark =="
conda run --no-capture-output -n gpp-core python -I \
  src/t22_tensorrt_llm/client_benchmark.py \
  --base-url "${BASE_URL}" \
  --max-tokens 16

echo "== T22 done =="
