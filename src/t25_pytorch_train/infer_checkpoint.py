"""T25 训练后推理：加载训练保存的 checkpoint，用同一模型做 chat 推理。

官方依据：
- Transformers `AutoModelForCausalLM` / `AutoTokenizer`（S16）。

用法：
    conda run --no-capture-output -n gpp-core python -I \
        src/t25_pytorch_train/infer_checkpoint.py \
        --checkpoint-dir caches/t25_output \
        --prompt "你好，请用一句话介绍你自己。" \
        --max-new-tokens 32
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description="T25 checkpoint inference")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="directory saved by train_baseline.py")
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint dir not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[t25_infer] loading checkpoint from {ckpt} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    if device == "cuda":
        model.to(device)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
    print(f"[t25_infer] prompt={args.prompt!r}")
    print(f"[t25_infer] response={response!r}")


if __name__ == "__main__":
    main()
