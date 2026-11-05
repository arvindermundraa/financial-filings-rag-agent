"""Thin LLM interface so the agent can run against Claude or a scripted fake in tests.

A model returns a dict with:
  stop_reason: "tool_use" or "end_turn"
  content: list of blocks, each {"type": "text", "text": ...}
           or {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
This is the shape of the Anthropic Messages API, so the real client is a pass-through.
"""
from __future__ import annotations

import os
from typing import Protocol

from . import config


class ChatModel(Protocol):
    def __call__(self, system: str, messages: list[dict], tools: list[dict]) -> dict: ...


class ClaudeModel:
    def __init__(self, model: str = config.LLM_MODEL, max_tokens: int = 1024):
        import anthropic  # imported here so tests and offline eval don't need a key

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to use the LLM agent.")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                           temperature=0, system=system, tools=tools,
                                           messages=messages)
        return {"stop_reason": resp.stop_reason,
                "content": [b.model_dump(exclude_none=True) for b in resp.content],
                "usage": {"input_tokens": resp.usage.input_tokens,
                          "output_tokens": resp.usage.output_tokens}}


class ScriptedModel:
    """Replays canned responses, in order. Used in tests and for a no-key demo."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def __call__(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise RuntimeError("ScriptedModel ran out of responses")
        return self.responses.pop(0)
