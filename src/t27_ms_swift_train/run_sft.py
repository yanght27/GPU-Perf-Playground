"""T27 ms-swift SFT/LoRA 最小训练入口。

官方依据：
- ms-swift `swift sft` 命令（S08a/S08c）。

这个文件是 `swift sft` CLI 的薄封装，方便统一参数和复现；真正训练逻辑由 ms-swift 执行。

用法：
    conda run --no-capture-output -n gpp-swift-4.4.3 python -I \
        src/t27_ms_swift_train/run_sft.py \
        --model "$PWD/assets/modelscope/qwen2.5-0.5b-instruct" \
        --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#8' \
        --output-dir caches/t27_output
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="T27 ms-swift SFT/LoRA wrapper")
    parser.add_argument("--model", type=str, required=True, help="local model path")
    parser.add_argument("--dataset", type=str, default="AI-ModelScope/alpaca-gpt4-data-zh#8")
    parser.add_argument("--tuner-type", type=str, default="lora")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--target-modules", type=str, default="all-linear")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="caches/t27_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cmd = [
        "swift", "sft",
        "--model", args.model,
        "--dataset", args.dataset,
        "--tuner_type", args.tuner_type,
        "--torch_dtype", args.torch_dtype,
        "--max_steps", str(args.max_steps),
        "--per_device_train_batch_size", str(args.batch_size),
        "--per_device_eval_batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--target_modules", args.target_modules,
        "--gradient_accumulation_steps", "1",
        "--max_length", str(args.max_length),
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
    ]
    print("[t27] running:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
