from __future__ import annotations

import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from llm_bench.schemas import ModelConfig
from llm_bench.settings import Settings, settings as default_settings


@dataclass
class CompletionResult:
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def make_client(model: ModelConfig, cfg: Settings | None = None) -> AsyncOpenAI:
    cfg = cfg or default_settings
    return AsyncOpenAI(
        api_key=model.api_key,
        base_url=model.base_url,
        timeout=cfg.request_timeout_s,
        max_retries=cfg.max_retries,
    )


async def chat_complete(
    model: ModelConfig,
    user_prompt: str,
    *,
    system_prompt: str | None = None,
    cfg: Settings | None = None,
) -> CompletionResult:
    client = make_client(model, cfg)
    messages = []
    sys = system_prompt if system_prompt is not None else model.system_prompt
    if sys:
        messages.append({"role": "system", "content": sys})
    messages.append({"role": "user", "content": user_prompt})

    started = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model.model_id,
        messages=messages,
        temperature=model.temperature,
        max_tokens=model.max_tokens,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    text = (resp.choices[0].message.content or "").strip()
    usage = resp.usage
    return CompletionResult(
        text=text,
        latency_ms=latency_ms,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )
