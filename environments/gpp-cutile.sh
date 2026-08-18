#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="gpp-cutile"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
MODE="install"

usage() {
  cat <<'EOF'
Usage: bash environments/gpp-cutile.sh [--verify-only]

Run from the repository root. The default mode creates/updates the Python 3.12
cuTile/CuPy lane and runs its basic package and CUDA check.
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
command -v python3 >/dev/null || { echo "ERROR: python3 is required to read config/day0-lock.json." >&2; exit 1; }
lock_values="$(python3 -I - "${ENV_NAME}" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(Path("config/day0-lock.json").read_text())
environment = next(item for item in lock["environments"] if item["name"] == sys.argv[1])
packages = environment["packages"]
tileir = environment["resolved_tileir_packages"]
print("\t".join((
    environment["python"], packages["cuda-tile"], packages["cupy-cuda13x"],
    packages["numpy"], tileir["cuda-toolkit"], tileir["nvidia-cuda-crt"],
    tileir["nvidia-cuda-nvcc"], tileir["nvidia-cuda-runtime"],
    tileir["nvidia-cuda-tileiras"], tileir["nvidia-nvjitlink"],
    tileir["nvidia-nvvm"],
)))
PY
)"
IFS=$'\t' read -r PYTHON_VERSION CUTILE_VERSION CUPY_VERSION NUMPY_VERSION \
  CUDA_TOOLKIT_VERSION CUDA_CRT_VERSION CUDA_NVCC_VERSION CUDA_RUNTIME_VERSION \
  CUDA_TILEIRAS_VERSION NVJITLINK_VERSION NVVM_VERSION <<<"${lock_values}"

if [[ "${MODE}" == "install" ]]; then
  assert_repo_path caches
  export PIP_CACHE_DIR="${PROJECT_ROOT}/caches/pip"
  export CUPY_CACHE_DIR="${PROJECT_ROOT}/caches/cupy/${ENV_NAME}"
  export CUDA_CACHE_PATH="${PROJECT_ROOT}/caches/cuda/${ENV_NAME}"
  assert_repo_path "${PIP_CACHE_DIR}"
  assert_repo_path "${CUPY_CACHE_DIR}"
  assert_repo_path "${CUDA_CACHE_PATH}"
  mkdir -p "${PIP_CACHE_DIR}" "${CUPY_CACHE_DIR}" "${CUDA_CACHE_PATH}"
else
  VERIFY_CACHE_ROOT="${TMPDIR:-/tmp}/gpp-day0-verify-${UID}/${ENV_NAME}"
  export PIP_CACHE_DIR="${VERIFY_CACHE_ROOT}/pip"
  export CUPY_CACHE_DIR="${VERIFY_CACHE_ROOT}/cupy"
  export CUDA_CACHE_PATH="${VERIFY_CACHE_ROOT}/cuda"
  mkdir -p "${CUPY_CACHE_DIR}" "${CUDA_CACHE_PATH}"
fi
command -v conda >/dev/null || { echo "ERROR: conda is not on PATH." >&2; exit 1; }

run_env() { conda run --no-capture-output -n "${ENV_NAME}" "$@"; }

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
    --index-url "${PYPI_MIRROR}" \
    --timeout 600 --retries 5 \
    "cuda-tile[tileiras]==${CUTILE_VERSION}" \
    "cupy-cuda13x==${CUPY_VERSION}" \
    "numpy==${NUMPY_VERSION}" \
    "cuda-toolkit==${CUDA_TOOLKIT_VERSION}" \
    "nvidia-cuda-crt==${CUDA_CRT_VERSION}" \
    "nvidia-cuda-nvcc==${CUDA_NVCC_VERSION}" \
    "nvidia-cuda-runtime==${CUDA_RUNTIME_VERSION}" \
    "nvidia-cuda-tileiras==${CUDA_TILEIRAS_VERSION}" \
    "nvidia-nvjitlink==${NVJITLINK_VERSION}" \
    "nvidia-nvvm==${NVVM_VERSION}"
fi

if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
  echo "ERROR: conda environment ${ENV_NAME} does not exist; run without --verify-only first." >&2
  exit 1
fi

run_env python -I -m pip check
run_env python -I - <<'PY'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
import cuda.tile
import cupy as cp
import numpy as np

lock = json.loads(Path("config/day0-lock.json").read_text())
environment = next(item for item in lock["environments"] if item["name"] == "gpp-cutile")
expected = environment["packages"] | environment["resolved_tileir_packages"]
hardware = lock["hardware_requirements"]
expected_python = tuple(int(part) for part in environment["python"].split("."))
if sys.version_info[:2] != expected_python:
    raise SystemExit(f"python: expected {environment['python']}, observed {sys.version.split()[0]}")
for distribution, version in expected.items():
    observed = metadata.version(distribution)
    if observed != version:
        raise SystemExit(f"{distribution}: expected {version}, observed {observed}")

x = cp.ones((1024, 1024), dtype=cp.float32)
y = x @ x
cp.cuda.Stream.null.synchronize()
if float(y[0, 0]) != 1024.0:
    raise SystemExit("gpp-cutile CUDA matmul returned the wrong value")
properties = cp.cuda.runtime.getDeviceProperties(0)
name = properties["name"]
if isinstance(name, bytes):
    name = name.decode()
capability = f"{properties['major']}.{properties['minor']}"
memory_mib = properties["totalGlobalMem"] // (1024 * 1024)
if name != hardware["gpu_name"]:
    raise SystemExit(f"GPU: expected {hardware['gpu_name']}, observed {name}")
if capability != hardware["compute_capability"]:
    raise SystemExit(f"compute capability: expected {hardware['compute_capability']}, observed {capability}")
if memory_mib < hardware["minimum_memory_mib"]:
    raise SystemExit(f"GPU memory: expected >= {hardware['minimum_memory_mib']} MiB, observed {memory_mib} MiB")
print("python_environment: gpp-cutile")
print("cuda-tile:", metadata.version("cuda-tile"), "cupy:", cp.__version__)
print("numpy:", np.__version__, "cuda_runtime:", cp.cuda.runtime.runtimeGetVersion())
print("gpu:", name, "sm:", capability, "memory_mib:", memory_mib)
print("matmul_value:", float(y[0, 0]))
print("gpp_cutile: PASS")
PY
