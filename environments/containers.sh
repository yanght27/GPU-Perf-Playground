#!/usr/bin/env bash
set -euo pipefail

MODE="install"

usage() {
  cat <<'EOF'
Usage: bash environments/containers.sh [--verify-only]

Run from the repository root after the learner has installed/started Docker
and NVIDIA Container Toolkit. The default mode pulls the pinned images;
--verify-only uses existing local images. Both modes check image architecture
and basic GPU passthrough. This script never runs sudo or changes the daemon.
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

command -v realpath >/dev/null || { echo "ERROR: realpath is required to identify the repository root." >&2; exit 1; }
SCRIPT_FILE="$(realpath -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${SCRIPT_FILE}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
if [[ "$(pwd -P)" != "${PROJECT_ROOT}" || ! -f config/day0-lock.json ]]; then
  echo "ERROR: run this script from the GPU-Perf-Playground repository root." >&2
  exit 2
fi
command -v python3 >/dev/null || {
  echo "ERROR: python3 is required to read config/day0-lock.json." >&2
  exit 1
}
lock_values="$(python3 -I - <<'PY'
import json
from pathlib import Path

lock = json.loads(Path("config/day0-lock.json").read_text())
host = lock["container_host"]
toolkit_cli = host["nvidia_container_toolkit"].split("-", 1)[0]
print("\t".join((host["docker_engine"], toolkit_cli)))
PY
)"
IFS=$'\t' read -r EXPECTED_DOCKER_VERSION EXPECTED_NVIDIA_CTK_VERSION <<<"${lock_values}"
images_output="$(python3 -I - <<'PY'
import json
from pathlib import Path

lock = json.loads(Path("config/day0-lock.json").read_text())
for container in lock["containers"]:
    print(container["image"])
PY
)"
mapfile -t images <<<"${images_output}"

command -v docker >/dev/null || {
  echo "ERROR: Docker is not installed. Follow docs/01-day-0.md, then rerun." >&2
  exit 1
}
command -v nvidia-ctk >/dev/null || {
  echo "ERROR: NVIDIA Container Toolkit is not installed. Follow docs/01-day-0.md, then rerun." >&2
  exit 1
}

if [[ -n "${DOCKER_HOST:-}" && "${DOCKER_HOST}" != "unix:///var/run/docker.sock" ]]; then
  echo "ERROR: DOCKER_HOST points away from the adopted local WSL Docker socket." >&2
  exit 1
fi
active_context="$(docker context show)"
context_endpoint="$(docker context inspect "${active_context}" --format '{{.Endpoints.docker.Host}}')"
if [[ "${active_context}" != "default" || "${context_endpoint}" != "unix:///var/run/docker.sock" ]]; then
  echo "ERROR: expected Docker context default at unix:///var/run/docker.sock; observed context ${active_context}." >&2
  exit 1
fi
docker info >/dev/null || {
  echo "ERROR: the current user cannot reach the Docker daemon." >&2
  echo "Start Docker and refresh docker-group membership as documented in docs/01-day-0.md." >&2
  exit 1
}

docker_client="$(docker version --format '{{.Client.Version}}')"
docker_server="$(docker version --format '{{.Server.Version}}')"
nvidia_ctk_version="$(nvidia-ctk --version | sed -n '1s/.* version //p')"
docker_os="$(docker info --format '{{.OperatingSystem}}')"
docker_kernel="$(docker info --format '{{.KernelVersion}}')"
if [[ "${docker_client}" != "${EXPECTED_DOCKER_VERSION}" || "${docker_server}" != "${EXPECTED_DOCKER_VERSION}" ]]; then
  echo "ERROR: expected Docker client/server ${EXPECTED_DOCKER_VERSION}; observed ${docker_client}/${docker_server}." >&2
  exit 1
fi
if [[ "${nvidia_ctk_version}" != "${EXPECTED_NVIDIA_CTK_VERSION}" ]]; then
  echo "ERROR: expected NVIDIA Container Toolkit ${EXPECTED_NVIDIA_CTK_VERSION}; observed ${nvidia_ctk_version}." >&2
  exit 1
fi
if [[ "${docker_os}" != Ubuntu\ 22.04* || "${docker_kernel}" != *microsoft-standard-WSL2* ]]; then
  echo "ERROR: Docker daemon is not the adopted Ubuntu 22.04 WSL2 native engine." >&2
  echo "Observed OS/kernel: ${docker_os} / ${docker_kernel}" >&2
  exit 1
fi

echo "docker_context: ${active_context} (${context_endpoint})"
echo "docker_client: ${docker_client}"
echo "docker_server: ${docker_server}"
echo "docker_os: ${docker_os}"
echo "docker_kernel: ${docker_kernel}"
echo "nvidia_container_toolkit: ${nvidia_ctk_version}"

runtime_map="$(docker info --format '{{json .Runtimes}}')"
if [[ "${runtime_map}" != *nvidia* ]]; then
  echo "ERROR: Docker does not list the nvidia runtime: ${runtime_map}" >&2
  exit 1
fi

if [[ "${MODE}" == "install" ]]; then
  docker_root="$(docker info --format '{{.DockerRootDir}}')"
  available_kib="$(df -Pk "${docker_root}" | awk 'NR == 2 {print $4}')"
  if (( available_kib < 50 * 1024 * 1024 )); then
    echo "WARNING: Docker root ${docker_root} has less than 50 GiB free; the three framework images may exhaust it." >&2
  fi
  for image in "${images[@]}"; do
    docker pull "${image}"
  done
fi

for image in "${images[@]}"; do
  architecture="$(docker image inspect --format '{{.Architecture}}' "${image}")"
  if [[ "${architecture}" != "amd64" ]]; then
    echo "ERROR: ${image} architecture is ${architecture}, expected amd64." >&2
    exit 1
  fi
  echo "image: ${image}"
  echo "architecture: ${architecture}"
  docker image inspect --format 'repo_digests: {{json .RepoDigests}}' "${image}"
done

gpu_query=(--query-gpu=driver_version,name,compute_cap,memory.total --format=csv,noheader,nounits)
for image in "${images[@]}"; do
  echo "GPU passthrough: ${image}"
  gpu_output="$(docker run --rm --gpus all --entrypoint nvidia-smi "${image}" "${gpu_query[@]}")"
  printf '%s\n' "${gpu_output}"
  GPP_GPU_OUTPUT="${gpu_output}" python3 -I - <<'PY'
import json
import os
from pathlib import Path

lock = json.loads(Path("config/day0-lock.json").read_text())
expected = lock["hardware_requirements"]
rows = []
for line in os.environ["GPP_GPU_OUTPUT"].splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) == 4:
        rows.append(fields)
if not rows:
    raise SystemExit("container nvidia-smi returned no parseable GPU row")
matches = [
    row for row in rows
    if row[1] == expected["gpu_name"]
    and row[2] == expected["compute_capability"]
    and int(row[3]) >= expected["minimum_memory_mib"]
]
if not matches:
    raise SystemExit(
        "container GPU identity mismatch: expected "
        f"{expected['gpu_name']} sm_{expected['compute_capability'].replace('.', '')} "
        f"with >= {expected['minimum_memory_mib']} MiB"
    )
PY
done

echo "container_gpu_passthrough: PASS"
echo "Serving/model Quick Starts remain deferred to their owning Tickets."
