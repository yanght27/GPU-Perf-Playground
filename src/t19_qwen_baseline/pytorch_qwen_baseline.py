"""T19 Qwen/Transformers 基线：固定快照上的生成正确性、确定性与基线指标。

官方依据：
- Qwen2.5-0.5B-Instruct 官方模型卡 Quick Start（assets/modelscope/.../README.md，台账 S20）；
- Transformers `AutoModelForCausalLM` / `AutoTokenizer` / `apply_chat_template`（台账 S16）；
- ModelScope 固定 revision 快照（config/day0-lock.json，台账 S14/S20）。

本脚本只做 T19 最小闭环：
1) 从 assets/modelscope 固定快照加载 tokenizer + model；
2) 验证 chat template 标记与 tokenizer round-trip；
3) 用固定 prompt suite 跑两次 greedy 生成，验证确定性；
4) 记录 dtype、设备、显存/RSS、prefill/generate/decode 耗时与 tokens/s。

用法：
    conda run --no-capture-output -n gpp-core python -I \
        src/t19_qwen_baseline/pytorch_qwen_baseline.py
    # 可选：只跑单个 prompt、控制生成长度
    conda run --no-capture-output -n gpp-core python -I \
        src/t19_qwen_baseline/pytorch_qwen_baseline.py --prompt math_zh --max-new-tokens 8
"""

import argparse
import resource
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t19_qwen_baseline.prompt_suite import PROMPT_SUITE, DEFAULT_MAX_NEW_TOKENS

MODEL_DIR = Path(__file__).resolve().parents[2] / "assets" / "modelscope" / "qwen2.5-0.5b-instruct"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _memory_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    # Linux RSS，单位为 KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_model_and_tokenizer():
    device = _device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[t19] loading tokenizer from {MODEL_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    print(f"[t19] loading model on {device} dtype={dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if device == "cuda":
        model.to(device)
    return tokenizer, model, device


def verify_tokenizer(tokenizer):
    print("[t19] tokenizer round-trip + chat template check", flush=True)
    text = "Hello Qwen2.5"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded.strip() == text.strip(), f"round-trip failed: {decoded!r} != {text!r}"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for marker in ("<|im_start|>", "<|im_end|>", "<|im_start|>assistant"):
        assert marker in chat, f"chat template missing marker {marker}"
    print("[t19] tokenizer round-trip PASS")
    print("[t19] chat template markers PASS")
    print(chat[:200].replace("\n", "\\n"))


def build_inputs(tokenizer, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")


def time_prefill(model, inputs, device):
    """只跑一次 prefill forward，返回秒数。"""
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(**inputs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0
    t0 = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    return time.perf_counter() - t0


def time_generate(model, tokenizer, inputs, max_new_tokens, device):
    """跑一次 generate，返回 (秒数, 新 token 数)。"""
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        end.record()
        torch.cuda.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        seconds = time.perf_counter() - t0
    new_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
    return seconds, new_tokens, out


def run_prompt(tokenizer, model, device, prompt, max_new_tokens):
    inputs = build_inputs(tokenizer, prompt["system"], prompt["user"])
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    prefill_s = time_prefill(model, inputs, device)
    gen_s1, new_tokens1, out1 = time_generate(model, tokenizer, inputs, max_new_tokens, device)
    resp1 = tokenizer.decode(out1[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # 第二次生成用于确定性验证（同输入、同 greedy 配置）
    gen_s2, new_tokens2, out2 = time_generate(model, tokenizer, inputs, max_new_tokens, device)
    resp2 = tokenizer.decode(out2[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    deterministic = resp1 == resp2
    decode_s = max(gen_s1 - prefill_s, 0.0)
    tokens_per_s = new_tokens1 / decode_s if decode_s > 0 else float("nan")

    print(f"[t19] prompt={prompt['name']} device={device} dtype={next(model.parameters()).dtype}")
    print(f"[t19]   input_tokens={inputs['input_ids'].shape[-1]} new_tokens={new_tokens1}")
    print(f"[t19]   prefill_s={prefill_s:.4f} generate_s={gen_s1:.4f} decode_s={decode_s:.4f} tokens_per_s={tokens_per_s:.2f}")
    print(f"[t19]   deterministic={deterministic}")
    print(f"[t19]   response={resp1!r}")
    if not deterministic:
        raise SystemExit(f"determinism FAIL for {prompt['name']}: {resp1!r} != {resp2!r}")
    return dict(
        name=prompt["name"],
        input_tokens=int(inputs["input_ids"].shape[-1]),
        new_tokens=new_tokens1,
        prefill_s=round(prefill_s, 4),
        generate_s=round(gen_s1, 4),
        decode_s=round(decode_s, 4),
        tokens_per_s=round(tokens_per_s, 2),
        deterministic=deterministic,
        response=resp1,
    )


def main():
    parser = argparse.ArgumentParser(description="T19 Qwen2.5-0.5B-Instruct baseline")
    parser.add_argument("--prompt", choices=[p["name"] for p in PROMPT_SUITE], default=None,
                        help="只跑指定 prompt；默认跑整个 suite")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="每次生成的最大新 token 数（默认 16）")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        raise SystemExit(f"model dir not found: {MODEL_DIR}")

    tokenizer, model, device = load_model_and_tokenizer()
    verify_tokenizer(tokenizer)

    prompts = [p for p in PROMPT_SUITE if args.prompt is None or p["name"] == args.prompt]
    results = []
    for prompt in prompts:
        results.append(run_prompt(tokenizer, model, device, prompt, args.max_new_tokens))

    print(f"[t19] memory_mb={_memory_mb(device):.1f} device={device}")
    print("[t19] SUMMARY")
    for r in results:
        print(f"[t19]   {r['name']}: in={r['input_tokens']} new={r['new_tokens']} "
              f"prefill={r['prefill_s']}s gen={r['generate_s']}s decode={r['decode_s']}s "
              f"tok/s={r['tokens_per_s']} deterministic={r['deterministic']}")
    print("[t19] ALL PASS")


if __name__ == "__main__":
    main()
