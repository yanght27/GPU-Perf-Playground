"""T22 TensorRT-LLM Serving 客户端：向 OpenAI 兼容服务发请求，记录 TTFT/ITL/TPOT/吞吐。

官方依据：
- TRT-LLM `trtllm-serve` 命令文档（S07b）；
- OpenAI-compatible endpoints：`/v1/models`、`/v1/completions`、`/v1/chat/completions`（S07b）。

用法：
    conda run --no-capture-output -n gpp-core python -I \
        src/t22_tensorrt_llm/client_benchmark.py \
        --base-url http://localhost:8000 \
        --max-tokens 16

    # 不连服务器，只检查脚本和参数是否可运行：
    conda run --no-capture-output -n gpp-core python -I \
        src/t22_tensorrt_llm/client_benchmark.py --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t19_qwen_baseline.prompt_suite import PROMPT_SUITE, DEFAULT_MAX_NEW_TOKENS


def chat_completion_stream(base_url, prompt, max_tokens):
    """向 TRT-LLM 发 streaming ChatCompletion，返回耗时指标和文本。"""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_time = None
    token_times = []
    text_parts = []

    with requests.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now - start
                token_times.append(now)
                text_parts.append(content)

    end = time.perf_counter()
    total_s = end - start
    text = "".join(text_parts)
    new_tokens = len(token_times)

    ttft_s = first_token_time
    if len(token_times) >= 2:
        itls = [token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)]
        avg_itl_s = sum(itls) / len(itls)
    else:
        avg_itl_s = float("nan")
    tpot_s = total_s / new_tokens if new_tokens else float("nan")
    throughput = new_tokens / total_s if total_s > 0 else float("nan")

    return {
        "prompt": prompt["name"],
        "text": text,
        "new_tokens": new_tokens,
        "ttft_s": ttft_s,
        "avg_itl_s": avg_itl_s,
        "tpot_s": tpot_s,
        "total_s": total_s,
        "throughput_tok_s": throughput,
    }


def run(base_url, max_tokens, prompts):
    print(f"[t22] base_url={base_url} max_tokens={max_tokens}")
    models = requests.get(base_url.rstrip("/") + "/v1/models", timeout=10)
    models.raise_for_status()
    print(f"[t22] /v1/models -> {models.json()['data'][0]['id']}")

    results = []
    for prompt in prompts:
        r = chat_completion_stream(base_url, prompt, max_tokens)
        results.append(r)
        print(f"[t22] prompt={r['prompt']} new_tokens={r['new_tokens']} "
              f"ttft={r['ttft_s']:.4f}s itl={r['avg_itl_s']:.4f}s "
              f"tpot={r['tpot_s']:.4f}s total={r['total_s']:.4f}s "
              f"throughput={r['throughput_tok_s']:.2f} tok/s")
        print(f"[t22]   response={r['text']!r}")

    print("[t22] SUMMARY")
    for r in results:
        print(f"[t22]   {r['prompt']}: ttft={r['ttft_s']:.4f} itl={r['avg_itl_s']:.4f} "
              f"tpot={r['tpot_s']:.4f} throughput={r['throughput_tok_s']:.2f} "
              f"new_tokens={r['new_tokens']} response={r['text']!r}")


def main():
    parser = argparse.ArgumentParser(description="T22 TensorRT-LLM OpenAI-compatible client benchmark")
    parser.add_argument("--base-url", default="http://localhost:8000", help="TRT-LLM server base URL")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="max new tokens per request")
    parser.add_argument("--prompt", choices=[p["name"] for p in PROMPT_SUITE], default=None,
                        help="only run one prompt")
    parser.add_argument("--dry-run", action="store_true",
                        help="do not connect to server; only validate script/params")
    args = parser.parse_args()

    prompts = [p for p in PROMPT_SUITE if args.prompt is None or p["name"] == args.prompt]
    if args.dry_run:
        print("[t22] dry-run: no server connection")
        for p in prompts:
            print(f"[t22]   would send prompt={p['name']} max_tokens={args.max_tokens}")
        return

    run(args.base_url, args.max_tokens, prompts)


if __name__ == "__main__":
    main()
