"""T24 推理统一对比：汇总 T20–T23 四框架的 TTFT/ITL/TPOT/吞吐。

官方依据：
- vLLM PagedAttention / Continuous Batching（S05）；
- SGLang RadixAttention（S06）；
- TensorRT-LLM Overview（S07d）；
- ms-swift 推理部署（S08e）。

用法：
    conda run --no-capture-output -n gpp-core python -I \
        src/t24_inference_compare/compare_frameworks.py

    # 只检查脚本，不要求证据文件存在：
    conda run --no-capture-output -n gpp-core python -I \
        src/t24_inference_compare/compare_frameworks.py --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORKS = [
    ("T20_vLLM", ROOT / "docs/evidence/T20/t20-run-all.txt"),
    ("T21_SGLang", ROOT / "docs/evidence/T21/t21-run-all.txt"),
    ("T22_TRTLLM", ROOT / "docs/evidence/T22/t22-run-all.txt"),
    ("T23_msSwift", ROOT / "docs/evidence/T23/t23-run-all.txt"),
]


def parse_evidence(path: Path):
    """从 client_benchmark 输出中解析 SUMMARY 行。"""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    pattern = re.compile(
        r"\[t\d+\]\s+(\w+):\s+ttft=([\d.]+|nan)\s+itl=([\d.]+|nan)\s+"
        r"tpot=([\d.]+|nan)\s+throughput=([\d.]+|nan)\s+new_tokens=(\d+)"
    )
    for m in pattern.finditer(text):
        rows.append({
            "prompt": m.group(1),
            "ttft_s": m.group(2),
            "itl_s": m.group(3),
            "tpot_s": m.group(4),
            "throughput": m.group(5),
            "new_tokens": m.group(6),
        })
    return rows if rows else None


def print_comparison():
    print("[t24] inference framework comparison")
    print(f"{'framework':<12} {'prompt':<10} {'ttft_s':<8} {'itl_s':<8} {'tpot_s':<8} {'tok/s':<8} {'new_tokens':<10}")
    for name, path in FRAMEWORKS:
        rows = parse_evidence(path)
        if rows is None:
            print(f"{name:<12} {'NO_DATA':<10}  (run scripts/run_t2x_all.sh on real GPU)")
            continue
        for r in rows:
            print(f"{name:<12} {r['prompt']:<10} {r['ttft_s']:<8} {r['itl_s']:<8} "
                  f"{r['tpot_s']:<8} {r['throughput']:<8} {r['new_tokens']:<10}")
    print("[t24] NOTE: 结论必须结合原始证据日志，禁止只看本表下结论。")


def main():
    parser = argparse.ArgumentParser(description="T24 inference framework comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="do not read evidence files; only validate script")
    args = parser.parse_args()
    if args.dry_run:
        print("[t24] dry-run: no evidence file reading")
        for name, path in FRAMEWORKS:
            print(f"[t24]   would read {name}: {path}")
        return
    print_comparison()


if __name__ == "__main__":
    main()
