"""T16 KV Cache —— 路径 5：CUTLASS CuTe DSL（官方 KV-Cache 能力检查 + 最近官方能力实测）。

官方依据：CUTLASS CuTe DSL 官方 Blackwell MLA decode 示例（台账 S02p）与
官方 ampere flash_attention_v2.py（台账 S02o）。
v1.7 五路径规则：T16 学习变量是 KV Cache。官方 KV-Cache（page table +
variable-length KV sequences）示例只存在于 Blackwell（sm_100）MLA decode 路径，
本机 sm_89 无法运行，因此机制记 N/A；最接近的官方 attention 能力是
flash_attention_v2，本文件在同一路径槽内完成两件事：
1) 静态检查 Blackwell KV-Cache 官方文件并记录 N/A 原因；
2) 实际运行官方 flash_attention_v2 小 shape（fp16，64×64，skip_ref_check），
   证明最近官方 CuTe attention 能力在本机 sm_89 + gpp-cute 可运行。
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUTLASS_PY = PROJECT_ROOT / "third_party/cutlass/examples/python/CuTeDSL"
KV_OFFICIAL = (
    CUTLASS_PY / "cute/blackwell/kernel/attention/mla/mla_decode_fp16.py"
)
FLASH_OFFICIAL = (
    CUTLASS_PY / "cute/ampere/kernel/attention/flash_attention_v2.py"
)


def capability_check():
    if not KV_OFFICIAL.exists():
        raise SystemExit(f"official CuTe KV-cache file not found: {KV_OFFICIAL}")
    text = KV_OFFICIAL.read_text()
    has_kv = "KV cache" in text or "KV Cache" in text
    is_blackwell = "blackwell" in str(KV_OFFICIAL).lower()
    print(f"[cute_t16] official={KV_OFFICIAL} has_kv_cache={has_kv} blackwell_path={is_blackwell}")
    print("[cute_t16] N/A: official KV-cache example is Blackwell MLA decode (sm_100), "
          "not runnable on this sm_89 machine. Capability pointer: S02p.")


def capability_run():
    if not FLASH_OFFICIAL.exists():
        raise SystemExit(f"official CuTe attention file not found: {FLASH_OFFICIAL}")
    cmd = [
        sys.executable, "-I", str(FLASH_OFFICIAL),
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
    print("[cute_t16] official flash_attention_v2 capability run PASS")
    print("[cute_t16] layer note: official KV-cache example is Blackwell-only; "
          "nearest flash capability passes")


if __name__ == "__main__":
    capability_check()
    capability_run()
