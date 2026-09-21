"""
LLMOps 内部平台 API 客户端
使用 OpenAI 兼容接口，支持同步和流式调用。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_api_key = os.getenv("LLMOPS_API_KEY")
_base_url = os.getenv("LLMOPS_BASE_URL")
_model = os.getenv("LLMOPS_MODEL", "default")

if not _api_key:
    raise ValueError("LLMOPS_API_KEY 未设置，请检查 .env 文件")
if not _base_url:
    raise ValueError("LLMOPS_BASE_URL 未设置，请检查 .env 文件")

_client = OpenAI(api_key=_api_key, base_url=_base_url)


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    stream: bool = False,
) -> str:
    """
    调用 LLM 对话接口。

    Args:
        messages: 消息列表，格式 [{"role": "system"|"user"|"assistant", "content": "..."}]
        model: 模型名，默认使用 .env 中的 LLMOPS_MODEL
        temperature: 生成温度 (0-2)
        max_tokens: 最大输出 token 数
        stream: 是否流式输出

    Returns:
        LLM 的回复文本
    """
    response = _client.chat.completions.create(
        model=model or _model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        collected = []
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                collected.append(delta.content)
                print(delta.content, end="", flush=True)
        print()
        return "".join(collected)
    else:
        return response.choices[0].message.content


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """
    调用 LLM 并支持 Function Calling（工具调用）。

    Args:
        messages: 消息列表
        tools: OpenAI 格式的 tools 定义
        model: 模型名
        temperature: 生成温度
        max_tokens: 最大输出 token 数

    Returns:
        完整的 response 对象，可从中提取 tool_calls
    """
    response = _client.chat.completions.create(
        model=model or _model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message


def embed(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """
    调用 LLM API 获取文本 embedding 向量。

    Args:
        texts: 待嵌入的文本列表
        model: embedding 模型名，默认使用 .env 中的 LLMOPS_EMBED_MODEL 或 LLMOPS_MODEL

    Returns:
        embedding 向量列表，每个向量为 float 列表
    """
    import os
    embed_model = model or os.getenv("LLMOPS_EMBED_MODEL") or _model

    try:
        response = _client.embeddings.create(
            model=embed_model,
            input=texts,
        )
        return [d.embedding for d in response.data]
    except Exception:
        # 如果 API 不支持 embeddings 端点，回退到简单的 TF-IDF 向量
        print("[Warning] Embedding API 不可用，使用字符级回退向量")
        return _fallback_embed(texts)


def _fallback_embed(texts: list[str], dim: int = 128) -> list[list[float]]:
    """简单的字符 bigram 哈希向量作为 embedding 回退方案"""
    import hashlib
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        # 字符 bigram 哈希
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            h = int(hashlib.md5(bigram.encode()).hexdigest()[:8], 16)
            idx = h % dim
            vec[idx] += 1.0
        # 字符 unigram 哈希
        for ch in text:
            h = int(hashlib.md5(ch.encode()).hexdigest()[:8], 16)
            idx = h % dim
            vec[idx] += 0.5
        # L2 归一化
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        vectors.append(vec)
    return vectors


def chat_json(
    messages: list[dict],
    json_schema: dict | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict | list | None:
    """
    调用 LLM 并强制返回结构化 JSON。

    Args:
        messages: 消息列表
        json_schema: 可选的 JSON Schema（用于 response_format）
        model: 模型名
        temperature: 生成温度（结构化输出建议 0.1-0.3）
        max_tokens: 最大输出 token 数

    Returns:
        解析后的 JSON 对象 (dict/list)，失败返回 None
    """
    import json

    kwargs = {
        "model": model or _model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # 如果模型支持 response_format
    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": json_schema,
            }
        }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"[chat_json] 错误: {e}")
        return None


if __name__ == "__main__":
    # 快速测试
    print(f"API Base URL: {_base_url}")
    print(f"Model: {_model}")
    print("---")
    reply = chat([
        {"role": "user", "content": "你好！请用一句话介绍你自己。"}
    ])
    print(f"回复: {reply}")

    # 测试 embedding
    print("\n--- Embedding 测试 ---")
    vecs = embed(["股权穿透", "财务分析", "贵州茅台酒股份有限公司"])
    print(f"  生成了 {len(vecs)} 个向量, 维度={len(vecs[0])}")
