#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="gpp-cute"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PYTORCH_MIRROR="https://mirror.nju.edu.cn/pytorch/whl/cu130/"
NJU_PYPI_MIRROR="https://mirror.nju.edu.cn/pypi/web/simple"
MODE="install"

usage() {
  cat <<'EOF'
Usage: bash environments/gpp-cute.sh [--verify-only]

Run from the repository root. The default mode creates/updates gpp-cute,
checks out the pinned CUTLASS source under third_party/cutlass, installs its
matching CuTe DSL wheel, and runs basic import/CUDA checks.
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
assert_repo_path third_party/cutlass
command -v python3 >/dev/null || { echo "ERROR: python3 is required to read config/day0-lock.json." >&2; exit 1; }
lock_values="$(python3 -I - "${ENV_NAME}" <<'PY'
import json
import sys
from pathlib import Path

lock = json.loads(Path("config/day0-lock.json").read_text())
environment = next(item for item in lock["environments"] if item["name"] == sys.argv[1])
packages = environment["packages"]
resolved = environment["resolved_packages"]
cutlass = environment["cutlass"]
print("\t".join((
    environment["python"], packages["torch"], packages["cuda-python"],
    packages["nvidia-cutlass-dsl"], packages["numpy"],
    resolved["nvidia-cuda-nvdisasm"],
    resolved["protobuf"], cutlass["repository"], cutlass["commit"],
)))
PY
)"
IFS=$'\t' read -r PYTHON_VERSION TORCH_VERSION CUDA_PYTHON_VERSION \
  CUTLASS_DSL_VERSION NUMPY_VERSION NVDISASM_VERSION PROTOBUF_VERSION \
  CUTLASS_REPOSITORY CUTLASS_COMMIT <<<"${lock_values}"

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
command -v conda >/dev/null || { echo "ERROR: conda is not on PATH." >&2; exit 1; }
command -v git >/dev/null || { echo "ERROR: git is not on PATH." >&2; exit 1; }

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

validate_cutlass_tree() {
  local origin dirty
  origin="$(git -C third_party/cutlass remote get-url origin)"
  if [[ "${origin}" != "${CUTLASS_REPOSITORY}" && "${origin}" != "git@github.com:NVIDIA/cutlass.git" ]]; then
    echo "ERROR: CUTLASS origin is not the locked NVIDIA repository; the observed URL is intentionally not printed because it may contain credentials." >&2
    exit 1
  fi
  dirty="$(git -C third_party/cutlass status --porcelain --untracked-files=all)"
  if [[ -n "${dirty}" ]]; then
    echo "ERROR: third_party/cutlass has local changes; preserve or remove them manually before rerunning:" >&2
    printf '%s\n' "${dirty}" >&2
    exit 1
  fi
}

if [[ "${MODE}" == "install" ]]; then
  ensure_conda_env

  run_env python -I -m pip install \
    --index-url "${PYTORCH_MIRROR}" \
    --extra-index-url "${PYPI_MIRROR}" \
    --timeout 600 --retries 5 \
    "torch==${TORCH_VERSION}"

  mkdir -p third_party
  if [[ ! -d third_party/cutlass/.git ]]; then
    git clone "${CUTLASS_REPOSITORY}" third_party/cutlass
  else
    validate_cutlass_tree
  fi
  git -C third_party/cutlass fetch origin "${CUTLASS_COMMIT}"
  git -C third_party/cutlass checkout --detach "${CUTLASS_COMMIT}"
  validate_cutlass_tree
  if [[ "$(git -C third_party/cutlass rev-parse HEAD)" != "${CUTLASS_COMMIT}" ]]; then
    echo "ERROR: CUTLASS checkout did not resolve to ${CUTLASS_COMMIT}." >&2
    exit 1
  fi
  if ! grep -Fxq "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" \
    third_party/cutlass/python/CuTeDSL/requirements-cu13.txt; then
    echo "ERROR: the locked CUTLASS commit does not request CuTe DSL ${CUTLASS_DSL_VERSION}." >&2
    exit 1
  fi

  run_env python -I -m pip install \
    --index-url "${NJU_PYPI_MIRROR}" \
    --timeout 600 --retries 5 \
    "cuda-python==${CUDA_PYTHON_VERSION}" \
    "numpy==${NUMPY_VERSION}" \
    "nvidia-cuda-nvdisasm==${NVDISASM_VERSION}" \
    "protobuf==${PROTOBUF_VERSION}"

  (
    cd third_party/cutlass
    PIP_INDEX_URL="${NJU_PYPI_MIRROR}" \
      PIP_TIMEOUT=600 \
      PIP_RETRIES=5 \
      conda run --no-capture-output -n "${ENV_NAME}" \
      bash python/CuTeDSL/setup.sh --cu13
  )
  validate_cutlass_tree
fi

if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
  echo "ERROR: conda environment ${ENV_NAME} does not exist; run without --verify-only first." >&2
  exit 1
fi
if [[ ! -d third_party/cutlass/.git ]]; then
  echo "ERROR: third_party/cutlass is missing; run without --verify-only first." >&2
  exit 1
fi
validate_cutlass_tree
observed_commit="$(git -C third_party/cutlass rev-parse HEAD)"
if [[ "${observed_commit}" != "${CUTLASS_COMMIT}" ]]; then
  echo "ERROR: CUTLASS commit mismatch: ${observed_commit}" >&2
  exit 1
fi

run_env python -I -m pip check
run_env python -I - <<'PY'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
import cutlass
import cutlass.cute
import torch

lock = json.loads(Path("config/day0-lock.json").read_text())
environment = next(item for item in lock["environments"] if item["name"] == "gpp-cute")
expected = environment["packages"] | environment["resolved_packages"]
hardware = lock["hardware_requirements"]
expected_python = tuple(int(part) for part in environment["python"].split("."))
if sys.version_info[:2] != expected_python:
    raise SystemExit(f"python: expected {environment['python']}, observed {sys.version.split()[0]}")
for distribution, version in expected.items():
    observed = metadata.version(distribution)
    if observed != version:
        raise SystemExit(f"{distribution}: expected {version}, observed {observed}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in gpp-cute")
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
    raise SystemExit("gpp-cute CUDA matmul returned the wrong value")
print("python_environment: gpp-cute")
print("torch:", torch.__version__, "wheel_cuda:", torch.version.cuda)
print("cutlass-dsl:", metadata.version("nvidia-cutlass-dsl"))
print("gpu:", properties.name, "sm:", capability, "memory_mib:", memory_mib)
print("matmul_value:", y[0, 0].item())
print("gpp_cute: PASS")
PY
