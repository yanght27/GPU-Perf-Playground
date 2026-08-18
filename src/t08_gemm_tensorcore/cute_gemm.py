"""T08 Tensor Core —— 路径 5：CUTLASS CuTe DSL（官方 ampere tensorop GEMM）。

官方依据：third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/
tensorop_gemm.py（commit 564d267e，台账 S02i）。
本文件通过 subprocess 运行官方示例的 512×512×512 fp16 tensor-core 配置，并把
"PASS" 作为 T08 CuTe 路径的官方能力证据。
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_EXAMPLE = (
    REPO_ROOT / "third_party" / "cutlass" / "examples" / "python" / "CuTeDSL"
    / "cute" / "ampere" / "kernel" / "dense_gemm" / "tensorop_gemm.py"
)


def main() -> None:
    cmd = [
        sys.executable, "-I", str(OFFICIAL_EXAMPLE),
        "--mnkl", "512,512,512,1",
        "--atom_layout_mnk", "2,2,1",
        "--ab_dtype", "Float16",
        "--c_dtype", "Float16",
        "--acc_dtype", "Float32",
        "--a_major", "m", "--b_major", "n", "--c_major", "n",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    output = (result.stdout + result.stderr)
    print("[cute_t8_official_tensorop] " + ("CORRECT_PASS" if "PASS" in output and result.returncode == 0 else "CORRECT_FAIL"))
    for line in output.splitlines():
        if "Execution time" in line or "Tolerance" in line or "PASS" in line:
            print(line)


if __name__ == "__main__":
    main()
