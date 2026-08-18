#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="gpp-core"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PYTORCH_MIRROR="https://mirror.nju.edu.cn/pytorch/whl/cu130/"
MODE="install"

usage() {
  cat <<'EOF'
Usage: bash environments/gpp-core.sh [--verify-only]

Run this command from the GPU-Perf-Playground repository root.
The default mode creates/updates gpp-core, downloads the pinned model and
dataset into assets/modelscope, and runs the basic package/GPU/asset checks.
--verify-only skips all network and installation work.
EOF
}

if (( $# > 1 )); then
  usage >&2
  exit 2
fi
case "${1:-}" in
  "") ;;
  --verify-only) MODE="verify" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

command -v realpath >/dev/null || { echo "ERROR: realpath is required for path confinement." >&2; exit 1; }
SCRIPT_FILE="$(realpath -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${SCRIPT_FILE}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
if [[ "$(pwd -P)" != "${PROJECT_ROOT}" || ! -f config/day0-lock.json ]]; then
  echo "ERROR: run this script from the GPU-Perf-Playground repository root." >&2
  exit 2
fi

assert_repo_path() {
  local resolved
  resolved="$(realpath -m -- "$1")"
  case "${resolved}" in
    "${PROJECT_ROOT}"|"${PROJECT_ROOT}"/*) ;;
    *) echo "ERROR: $1 resolves outside the repository: ${resolved}" >&2; exit 1 ;;
  esac
}
assert_repo_path assets

command -v python3 >/dev/null || {
  echo "ERROR: python3 is required to read config/day0-lock.json." >&2
  exit 1
}
lock_values="$(python3 -I - "${ENV_NAME}" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(Path("config/day0-lock.json").read_text())
environment = next(item for item in lock["environments"] if item["name"] == sys.argv[1])
packages = environment["packages"]
model = next(item for item in lock["assets"] if item["kind"] == "model")
dataset = next(item for item in lock["assets"] if item["kind"] == "dataset")
print("\t".join((
    environment["python"], packages["torch"], packages["triton"],
    packages["transformers"], packages["modelscope-hub"], packages["numpy"],
    model["modelscope_id"], model["revision"], model["relative_path"],
    model["revision_record"], dataset["modelscope_id"], dataset["revision"],
    dataset["relative_path"], dataset["revision_record"],
)))
PY
)"
IFS=$'\t' read -r PYTHON_VERSION TORCH_VERSION TRITON_VERSION \
  TRANSFORMERS_VERSION MODELSCOPE_HUB_VERSION NUMPY_VERSION \
  MODEL_ID MODEL_REVISION MODEL_RELATIVE_PATH MODEL_REVISION_RECORD \
  DATASET_ID DATASET_REVISION DATASET_RELATIVE_PATH DATASET_REVISION_RECORD \
  <<<"${lock_values}"
assert_repo_path "assets/${MODEL_RELATIVE_PATH}"
assert_repo_path "assets/${MODEL_REVISION_RECORD}"
assert_repo_path "assets/${DATASET_RELATIVE_PATH}"
assert_repo_path "assets/${DATASET_REVISION_RECORD}"

if [[ "${MODE}" == "install" ]]; then
  assert_repo_path caches
  export PIP_CACHE_DIR="${PROJECT_ROOT}/caches/pip"
  export TRITON_CACHE_DIR="${PROJECT_ROOT}/caches/triton/${ENV_NAME}"
  export CUDA_CACHE_PATH="${PROJECT_ROOT}/caches/cuda/${ENV_NAME}"
  assert_repo_path "${PIP_CACHE_DIR}"
  assert_repo_path "${TRITON_CACHE_DIR}"
  assert_repo_path "${CUDA_CACHE_PATH}"
  mkdir -p "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"
else
  VERIFY_CACHE_ROOT="${TMPDIR:-/tmp}/gpp-day0-verify-${UID}/${ENV_NAME}"
  export PIP_CACHE_DIR="${VERIFY_CACHE_ROOT}/pip"
  export TRITON_CACHE_DIR="${VERIFY_CACHE_ROOT}/triton"
  export CUDA_CACHE_PATH="${VERIFY_CACHE_ROOT}/cuda"
  mkdir -p "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"
fi

command -v conda >/dev/null || {
  echo "ERROR: conda is not on PATH." >&2
  exit 1
}

run_env() {
  conda run --no-capture-output -n "${ENV_NAME}" "$@"
}

conda_env_exists() {
  conda env list --json | python3 -I -c '
import json, sys
from pathlib import Path
name = sys.argv[1]
raise SystemExit(0 if any(Path(path).name == name for path in json.load(sys.stdin)["envs"]) else 1)
' "${ENV_NAME}"
}

ensure_conda_env() {
  if [[ "$(conda run -n "${ENV_NAME}" python -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)" == "${PYTHON_VERSION}" ]] \
    && conda run -n "${ENV_NAME}" python -I -m pip --version >/dev/null 2>&1; then
    echo "Using existing Python ${PYTHON_VERSION} conda environment: ${ENV_NAME}"
    return
  fi

  if conda_env_exists; then
    observed_python="$(conda run -n "${ENV_NAME}" python -c \
      'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ "${observed_python}" != "${PYTHON_VERSION}" ]]; then
      echo "ERROR: ${ENV_NAME} exists with Python ${observed_python:-unusable}; expected ${PYTHON_VERSION}." >&2
      echo "Preserve anything needed, remove that environment manually, then rerun; the script will not delete it." >&2
      exit 1
    fi
    echo "Repairing pip in existing ${ENV_NAME}."
    if ! conda install --yes --name "${ENV_NAME}" \
      --override-channels --channel defaults --solver libmamba \
      "python=${PYTHON_VERSION}" pip; then
      echo "ERROR: pip in ${ENV_NAME} could not be repaired; preserve anything needed, then remove that environment manually and rerun." >&2
      exit 1
    fi
  else
    conda create --yes --name "${ENV_NAME}" \
      --override-channels --channel defaults --solver libmamba \
      --no-default-packages "python=${PYTHON_VERSION}" pip
  fi
}

if [[ "${MODE}" == "install" ]]; then
  ensure_conda_env

  run_env python -I -m pip install \
    --index-url "${PYTORCH_MIRROR}" \
    --extra-index-url "${PYPI_MIRROR}" \
    --timeout 600 --retries 5 \
    "torch==${TORCH_VERSION}" "triton==${TRITON_VERSION}"

  run_env python -I -m pip install \
    --index-url "${PYPI_MIRROR}" \
    --timeout 600 --retries 5 \
    "transformers==${TRANSFORMERS_VERSION}" \
    "modelscope-hub==${MODELSCOPE_HUB_VERSION}" \
    "numpy==${NUMPY_VERSION}"

  mkdir -p assets/modelscope

  MODEL_DIR="assets/${MODEL_RELATIVE_PATH}"
  run_env ms download \
    --repo-type model \
    --revision "${MODEL_REVISION}" \
    --local_dir "${MODEL_DIR}" \
    "${MODEL_ID}"

  DATASET_DIR="assets/${DATASET_RELATIVE_PATH}"
  run_env ms download \
    --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --local_dir "${DATASET_DIR}" \
    "${DATASET_ID}"

  printf '%s\n' "${MODEL_REVISION}" \
    >"assets/${MODEL_REVISION_RECORD}"
  printf '%s\n' "${DATASET_REVISION}" \
    >"assets/${DATASET_REVISION_RECORD}"
fi

if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
  echo "ERROR: conda environment ${ENV_NAME} does not exist; run without --verify-only first." >&2
  exit 1
fi

run_env python -I -m pip check
run_env python -I - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import modelscope_hub
import numpy as np
import torch
import transformers
import triton

lock = json.loads(Path("config/day0-lock.json").read_text())
environment = next(item for item in lock["environments"] if item["name"] == "gpp-core")
expected_versions = environment["packages"]
hardware = lock["hardware_requirements"]
expected_python = tuple(int(part) for part in environment["python"].split("."))
if sys.version_info[:2] != expected_python:
    raise SystemExit(f"python: expected {environment['python']}, observed {sys.version.split()[0]}")
for distribution, expected in expected_versions.items():
    observed = metadata.version(distribution)
    if observed != expected:
        raise SystemExit(f"{distribution}: expected {expected}, observed {observed}")

for asset in lock["assets"]:
    root = Path("assets") / asset["relative_path"]
    expected_files = {item["path"]: item for item in asset["files"]}
    symlinks = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()
    )
    if symlinks:
        raise SystemExit(f"{asset['name']} contains symlinks: {symlinks}")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(expected_files) - actual_files)
    extra = sorted(actual_files - set(expected_files))
    if missing or (asset.get("fail_on_extra_files") and extra):
        raise SystemExit(f"{asset['name']} file-set mismatch: missing={missing}, extra={extra}")
    for relative_path, expected in expected_files.items():
        path = root / relative_path
        if path.stat().st_size != expected["size"]:
            raise SystemExit(f"size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise SystemExit(f"sha256 mismatch: {path}")
    revision_record = Path("assets") / asset["revision_record"]
    if revision_record.read_text().strip() != asset["revision"]:
        raise SystemExit(f"revision record mismatch: {revision_record}")

model_asset = next(item for item in lock["assets"] if item["kind"] == "model")
tokenizer = json.loads(
    Path("assets").joinpath(
        model_asset["relative_path"], model_asset["tokenizer_file"]
    ).read_text()
)
chat_template = tokenizer.get("chat_template", "")
for marker in model_asset["chat_template_required_markers"]:
    if marker not in chat_template:
        raise SystemExit(f"chat template marker missing: {marker}")

print("python_environment: gpp-core")
print("torch:", torch.__version__, "wheel_cuda:", torch.version.cuda)
print("transformers:", transformers.__version__, "triton:", triton.__version__)
print("numpy:", np.__version__)
print("model_and_dataset_integrity: PASS")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in gpp-core")
properties = torch.cuda.get_device_properties(0)
capability = f"{properties.major}.{properties.minor}"
memory_mib = properties.total_memory // (1024 * 1024)
if properties.name != hardware["gpu_name"]:
    raise SystemExit(f"GPU: expected {hardware['gpu_name']}, observed {properties.name}")
if capability != hardware["compute_capability"]:
    raise SystemExit(f"compute capability: expected {hardware['compute_capability']}, observed {capability}")
if memory_mib < hardware["minimum_memory_mib"]:
    raise SystemExit(f"GPU memory: expected >= {hardware['minimum_memory_mib']} MiB, observed {memory_mib} MiB")
x = torch.ones((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
if y[0, 0].item() != 1024.0:
    raise SystemExit("gpp-core CUDA matmul returned the wrong value")
print("gpu:", properties.name, "sm:", capability, "memory_mib:", memory_mib)
print("matmul_value:", y[0, 0].item())
print("gpp_core: PASS")
PY
