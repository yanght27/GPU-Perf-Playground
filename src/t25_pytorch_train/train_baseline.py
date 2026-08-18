"""T25 PyTorch 训练基线：最小训练循环 + checkpoint + 单卡指标。

官方依据：
- PyTorch 官方训练教程（S15）；
- Transformers `AutoModelForCausalLM` / `AutoTokenizer`（S16）。

用法：
    conda run --no-capture-output -n gpp-core python -I \
        src/t25_pytorch_train/train_baseline.py \
        --num-samples 8 --max-steps 2 --batch-size 2 --max-length 128 \
        --output-dir caches/t25_output

    # 最小 CPU 验证：
    conda run --no-capture-output -n gpp-core python -I \
        src/t25_pytorch_train/train_baseline.py \
        --num-samples 2 --max-steps 1 --batch-size 1 --max-length 32 \
        --output-dir /tmp/t25_cpu_check
"""

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

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
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
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


def load_model_tokenizer(device):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[t25] loading tokenizer from {MODEL_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    print(f"[t25] loading model on {device} dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.train()
    if device == "cuda":
        model.to(device)
    return tokenizer, model


def time_step(model, input_ids, attention_mask, optimizer, device):
    """执行一次 forward/backward/optimizer.step，返回 (loss, seconds)。"""
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        seconds = time.perf_counter() - t0
    return float(loss.detach().cpu()), seconds


def save_checkpoint(model, tokenizer, optimizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    torch.save(optimizer.state_dict(), str(output_dir / "optimizer.pt"))
    print(f"[t25] checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="T25 PyTorch minimal training baseline")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="caches/t25_output")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="skip checkpoint save (for quick sandbox verification)")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit(f"model dir not found: {MODEL_DIR}")
    if not DATA_CSV.exists():
        raise SystemExit(f"dataset not found: {DATA_CSV}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"[t25] seed={args.seed}", flush=True)

    tokenizer, model = load_model_tokenizer(device)
    dataset = AlpacaSFTDataset(DATA_CSV, tokenizer, args.num_samples, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print(f"[t25] dataset samples={len(dataset)} batch_size={args.batch_size} "
          f"max_steps={args.max_steps} max_length={args.max_length} device={device}")

    total_tokens = 0
    total_samples = 0
    total_time = 0.0
    for step, (input_ids, attention_mask) in enumerate(loader):
        if step >= args.max_steps:
            break
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss, seconds = time_step(model, input_ids, attention_mask, optimizer, device)
        batch_tokens = int(input_ids.numel())
        total_tokens += batch_tokens
        total_samples += input_ids.shape[0]
        total_time += seconds
        print(f"[t25] step={step} loss={loss:.4f} step_time={seconds:.4f}s "
              f"tokens={batch_tokens} samples={input_ids.shape[0]}")

    avg_step_time = total_time / max(args.max_steps, 1)
    samples_per_s = total_samples / total_time if total_time > 0 else float("nan")
    tokens_per_s = total_tokens / total_time if total_time > 0 else float("nan")
    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[t25] avg_step_time={avg_step_time:.4f}s samples_per_s={samples_per_s:.4f} "
          f"tokens_per_s={tokens_per_s:.2f} memory_mb={mem_mb:.1f}")

    if args.skip_checkpoint:
        print(f"[t25] skip checkpoint save (--skip-checkpoint)")
    else:
        save_checkpoint(model, tokenizer, optimizer, Path(args.output_dir))
    print("[t25] ALL PASS")


if __name__ == "__main__":
    main()
