"""T19 固定 prompt suite（后续 T20–T24 统一复用）。

官方依据：Qwen2.5-0.5B-Instruct 官方 README Quick Start 的 messages 结构
（assets/modelscope/qwen2.5-0.5b-instruct/README.md，台账 S20）；
Transformers `apply_chat_template`（台账 S16）。
"""

PROMPT_SUITE = [
    {
        "name": "intro_zh",
        "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "user": "你好，请用一句话介绍你自己。",
    },
    {
        "name": "math_zh",
        "system": "You are a helpful assistant.",
        "user": "计算 12*8 等于多少？只回答数字。",
    },
    {
        "name": "code_en",
        "system": "You are a helpful assistant.",
        "user": "Write a Python function that adds two numbers and returns the result.",
    },
]

DEFAULT_MAX_NEW_TOKENS = 16
