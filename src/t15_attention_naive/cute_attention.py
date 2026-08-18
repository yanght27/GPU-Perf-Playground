"""T15 Attention —— 路径 5：CUTLASS CuTe DSL（官方 flash_attention_v2 能力实测 + 层级 N/A）。

官方依据：CUTLASS CuTe DSL 官方 ampere flash_attention_v2.py（台账 S02o）。
本文件实际运行官方示例的小 shape（fp16，64×64，skip_ref_check），证明官方 CuTe
attention 能力在本机 sm_89 + gpp-cute 可运行。
官方示例是 flash 层（非 T15 朴素层），机制仍记 N/A；能力已实测。
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = (
    PROJECT_ROOT
    / "third_party/cutlass/examples/python/CuTeDSL/"
    / "cute/ampere/kernel/attention/flash_attention_v2.py"
)


def main():
    if not OFFICIAL.exists():
        raise SystemExit(f"official CuTe attention file not found: {OFFICIAL}")
    cmd = [
        sys.executable, "-I", str(OFFICIAL),
        "--dtype", "Float16", "--batch_size", "1",
        "--seqlen_q", "64", "--seqlen_k", "64",
        "--num_head", "4", "--head_dim", "64",
        "--m_block_size", "64", "--n_block_size", "64",
        "--num_threads", "128",
        "--warmup_iterations", "1", "--iterations", "1",
        "--skip_ref_check",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stdout + r.stderr)
    print("[cute_t15] official flash_attention_v2 capability run PASS")
    print("[cute_t15] layer note: official example is flash layer (not T15 naive layer)")


if __name__ == "__main__":
    main()
