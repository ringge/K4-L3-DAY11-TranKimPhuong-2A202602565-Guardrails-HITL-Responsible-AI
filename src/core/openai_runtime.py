"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-2.5-2.6b (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)

_OPENROUTER_REQUEST_INTERVAL_SECONDS = 60 / 20 + 0.1
_OPENROUTER_RATE_LIMIT_RETRIES = 5


def _is_rate_limit_error(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429 or (
        "ratelimit" in type(error).__name__.casefold()
    )


def _retry_after_seconds(error: Exception) -> float | None:
    """Read both HTTP and OpenRouter's nested upstream retry hints."""
    values: list[float] = []

    def add_delay(raw: object) -> None:
        if raw is None:
            return
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(raw))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                return
        if math.isfinite(seconds):
            values.append(max(0.0, seconds))

    response = getattr(error, "response", None)
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping) and response is not None:
        try:
            body = response.json()
        except (TypeError, ValueError):
            body = None

    metadata = None
    if isinstance(body, Mapping):
        detail = body.get("error", body)
        if isinstance(detail, Mapping):
            metadata = detail.get("metadata")
    if isinstance(metadata, Mapping):
        add_delay(metadata.get("retry_after_seconds"))
        add_delay(metadata.get("retry_after_seconds_raw"))

    for headers in (
        getattr(response, "headers", None),
        getattr(error, "headers", None),
        metadata.get("headers") if isinstance(metadata, Mapping) else None,
    ):
        if headers is not None and hasattr(headers, "get"):
            add_delay(headers.get("retry-after") or headers.get("Retry-After"))

    return max(values) if values else None


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)
    _openrouter_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _last_openrouter_request_at: float | None = field(
        default=None, init=False, repr=False
    )

    def _client(self):
        from openai import OpenAI

        kwargs = dict(self.client_kwargs or {})
        if self.provider == "openrouter":
            # The Blue runner handles 429s so a retry never replays input plugins.
            kwargs.setdefault("max_retries", 0)
        return OpenAI(**kwargs)

    async def chat(self, agent: OpenAIAgent, user_message: str) -> str:
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return blocked

        block_msg = await self._run_input_plugins(user_message)
        if block_msg is not None:
            return block_msg

        client = self._client()
        completion = await self._create_completion(
            client,
            agent,
            user_message,
        )
        text = (completion.choices[0].message.content or "").strip()

        for hook in self.output_hooks:
            text = hook(text)

        text = await self._run_output_plugins(text)
        return text

    async def _create_completion(self, client, agent: OpenAIAgent, user_message: str):
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": agent.instruction},
                {"role": "user", "content": user_message},
            ],
            "temperature": self.temperature,
        }
        if self.provider != "openrouter":
            return client.chat.completions.create(**request)

        for retry_count in range(_OPENROUTER_RATE_LIMIT_RETRIES + 1):
            await self._pace_openrouter_request()
            try:
                return client.chat.completions.create(**request)
            except Exception as error:
                if not _is_rate_limit_error(error):
                    raise
                if retry_count == _OPENROUTER_RATE_LIMIT_RETRIES:
                    raise RuntimeError(
                        "Blue model remained rate limited by OpenRouter/Liquid "
                        "after 5 retries; the assignment results were not written."
                    ) from error

                backoff = _OPENROUTER_REQUEST_INTERVAL_SECONDS * (2**retry_count)
                retry_after = _retry_after_seconds(error)
                delay = max(backoff, (retry_after + 1) if retry_after is not None else 0)
                print(
                    f"Blue model rate limited; retrying in {delay:.0f}s "
                    f"({retry_count + 1}/{_OPENROUTER_RATE_LIMIT_RETRIES}).",
                    flush=True,
                )
                await asyncio.sleep(delay)

        raise AssertionError("OpenRouter retry loop exited unexpectedly")

    async def _pace_openrouter_request(self) -> None:
        async with self._openrouter_lock:
            if self._last_openrouter_request_at is not None:
                elapsed = time.monotonic() - self._last_openrouter_request_at
                wait_seconds = _OPENROUTER_REQUEST_INTERVAL_SECONDS - elapsed
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
            self._last_openrouter_request_at = time.monotonic()

    async def _run_input_plugins(self, user_message: str) -> str | None:
        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext()
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            return _content_to_text(result)
        return None

    async def _run_output_plugins(self, text: str) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            try:
                out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=_Ctx(), llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
        return _content_to_text(llm_response.content) or text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-2.5-2.6b."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
