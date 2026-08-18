"""T25 训练后评测：在 held-out 子集上计算 loss/perplexity，并对比生成结果。

说明：当前 alpaca-gpt4-data-zh 快照只有 train.csv，没有独立 test split。
因此我们从 train.csv 的后面部分取 held-out 样本（默认跳过前 1000 条训练用数据）。

官方依据：
- Transformers AutoModelForCausalLM / AutoTokenizer（S16）。

用法：
    conda run --no-capture-output -n gpp-core python -I \
        src/t25_pytorch_train/eval_checkpoint.py \
        --checkpoint-dir caches/t25_output \
        --eval-samples 20 \
        --skip-rows 1000 \
        --max-new-tokens 32
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "modelscope" / "qwen2.5-0.5b-instruct"
DATA_CSV = Path(__file__).resolve().parents[2] / "assets" / "modelscope" / "alpaca-gpt4-data-zh" / "train.csv"


def load_held_out(csv_path, skip_rows, eval_samples):
    """返回 [(user_content, reference_output)]，从 skip_rows 之后取 eval_samples 条。"""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < skip_rows:
                continue
            if len(rows) >= eval_samples:
                break
            instruction = row["instruction"]
            inp = row["input"].strip()
            user_content = instruction if not inp else f"{instruction}\n{inp}"
            rows.append((user_content, row["output"]))
    return rows


def build_chat_text(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="T25 checkpoint evaluation")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--eval-samples", type=int, default=20)
    parser.add_argument("--skip-rows", type=int, default=1000,
                        help="skip first N rows (assumed training data)")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint dir not found: {ckpt}")
    if not DATA_CSV.exists():
        raise SystemExit(f"dataset not found: {DATA_CSV}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[t25_eval] loading checkpoint from {ckpt} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    if device == "cuda":
        model.to(device)

    held_out = load_held_out(DATA_CSV, args.skip_rows, args.eval_samples)
    print(f"[t25_eval] held_out_samples={len(held_out)} skip_rows={args.skip_rows}")

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for idx, (user_content, ref) in enumerate(held_out):
            text = build_chat_text(tokenizer, user_content)
            enc = tokenizer(text, truncation=True, max_length=args.max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += float(loss.detach().cpu()) * input_ids.numel()
            total_tokens += input_ids.numel()

            if idx < 3:
                gen_text = build_chat_text(tokenizer, user_content)
                gen_inputs = tokenizer(gen_text, return_tensors="pt").to(device)
                out = model.generate(
                    **gen_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    out[0][gen_inputs["input_ids"].shape[-1]:], skip_special_tokens=True
                )
                print(f"[t25_eval] sample={idx}")
                print(f"[t25_eval]   ref      = {ref[:60]!r}")
                print(f"[t25_eval]   generate = {response[:60]!r}")

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    print(f"[t25_eval] avg_loss={avg_loss:.4f} perplexity={ppl:.4f} tokens={total_tokens}")
    print("[t25_eval] DONE")


if __name__ == "__main__":
    main()
