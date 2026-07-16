"""
LLM generation via LiteLLM (provider-agnostic). Default target is Groq, so a weak
CPU never runs a large model. Streaming yields text for a live UI; the final
chunk carries token usage for cost tracking.
"""
from __future__ import annotations
from typing import Iterator


def stream_completion(model: str, system_prompt: str, messages: list[dict],
                      temperature: float, max_tokens: int) -> Iterator[dict]:
    """Yields {'text': ...} chunks, then a final {'usage': {...}} chunk."""
    from litellm import completion
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    resp = completion(
        model=model, messages=full_messages,
        temperature=temperature, max_tokens=max_tokens,
        stream=True, stream_options={"include_usage": True},
    )
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for part in resp:
        choices = getattr(part, "choices", None)
        if choices:
            delta = choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                yield {"text": token}
        u = getattr(part, "usage", None)
        if u:
            usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                     "completion_tokens": getattr(u, "completion_tokens", 0) or 0}
    yield {"usage": usage}
