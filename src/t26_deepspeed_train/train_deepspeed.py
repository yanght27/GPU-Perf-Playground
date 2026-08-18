"""T26 DeepSpeed 训练：在 T25 PyTorch 循环上接入 DeepSpeed，单卡 ZeRO-1/2 实测。

官方依据：
- DeepSpeed 官方文档（S04）：deepspeed.initialize、ZeRO 配置。

用法（单卡）：
    conda run --no-capture-output -n gpp-deepspeed-0.19.5 python -I \
        src/t26_deepspeed_train/train_deepspeed.py \
        --num-samples 8 --max-steps 2 --batch-size 2 --max-length 128 \
        --zero-stage 2 --output-dir caches/t26_output

    # 也可以用官方 launcher
    conda run --no-capture-output -n gpp-deepspeed-0.19.5 deepspeed \
        --num_gpus 1 src/t26_deepspeed_train/train_deepspeed.py \
        --num-samples 8 --max-steps 2 --batch-size 2 --max-length 128 \
        --zero-stage 2 --output-dir caches/t26_output
"""

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "modelscope" / "qwen2.5-0.5b-instruct"
DATA_CSV = Path(__file__).resolve().parents[2] / "assets" / "modelscope" / "alpaca-gpt4-data-zh" / "train.csv"


class AlpacaSFTDataset(Dataset):
    def __init__(self, csv_path, tokenizer, num_samples, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= num_samples:
                    break
                instruction = row["instruction"]
                inp = row["input"].strip()
                output = row["output"]
                user_content = instruction if not inp else f"{instruction}\n{inp}"
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx], truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return enc["input_ids"][0], enc["attention_mask"][0]


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b[0] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b[1] for b in batch], batch_first=True, padding_value=0
    )
    return input_ids, attention_mask


def build_ds_config(batch_size, lr, zero_stage):
    return {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr},
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": 1,
            },
        },
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": True},
    }


def main():
    parser = argparse.ArgumentParser(description="T26 DeepSpeed minimal training baseline")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--zero-stage", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=str, default="caches/t26_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-checkpoint", action="store_true")
    # DeepSpeed launcher may pass --local_rank
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit(f"model dir not found: {MODEL_DIR}")
    if not DATA_CSV.exists():
        raise SystemExit(f"dataset not found: {DATA_CSV}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[t26] loading tokenizer from {MODEL_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    print(f"[t26] loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.train()

    ds_config = build_ds_config(args.batch_size, args.lr, args.zero_stage)
    print(f"[t26] ds_config={ds_config}", flush=True)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
    )

    dataset = AlpacaSFTDataset(DATA_CSV, tokenizer, args.num_samples, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"[t26] dataset samples={len(dataset)} zero_stage={args.zero_stage} "
          f"max_steps={args.max_steps} device={model_engine.device}")

    total_tokens = 0
    total_samples = 0
    total_time = 0.0
    for step, (input_ids, attention_mask) in enumerate(loader):
        if step >= args.max_steps:
            break
        input_ids = input_ids.to(model_engine.device)
        attention_mask = attention_mask.to(model_engine.device)

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            end.record()
            torch.cuda.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
        else:
            t0 = time.perf_counter()
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()
            seconds = time.perf_counter() - t0

        batch_tokens = int(input_ids.numel())
        total_tokens += batch_tokens
        total_samples += input_ids.shape[0]
        total_time += seconds
        print(f"[t26] step={step} loss={float(loss.detach().cpu()):.4f} "
              f"step_time={seconds:.4f}s tokens={batch_tokens}")

    avg_step_time = total_time / max(args.max_steps, 1)
    samples_per_s = total_samples / total_time if total_time > 0 else float("nan")
    tokens_per_s = total_tokens / total_time if total_time > 0 else float("nan")
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[t26] avg_step_time={avg_step_time:.4f}s samples_per_s={samples_per_s:.4f} "
          f"tokens_per_s={tokens_per_s:.2f} memory_mb={mem_mb:.1f}")

    if args.skip_checkpoint:
        print("[t26] skip checkpoint save (--skip-checkpoint)")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_engine.module.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        torch.save(optimizer.state_dict(), str(out_dir / "optimizer.pt"))
        print(f"[t26] checkpoint saved to {out_dir}")
    print("[t26] ALL PASS")


if __name__ == "__main__":
    main()
