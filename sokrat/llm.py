"""Production-minded wrapper around the OpenAI client.

Handles the unglamorous things that make an agent trustworthy in production:
  * structured outputs via strict JSON schema (Pydantic) — no manual parsing;
  * embeddings for retrieval;
  * a tool-calling chat entry point for the agent loop;
  * automatic retries with backoff on transient errors;
  * an on-disk cache for structured/embedding calls so batch runs over thousands
    of learners never re-pay for identical calls;
  * token + cost accounting so a run's cost is always visible.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("SOKRAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("SOKRAT_EMBED_MODEL", "text-embedding-3-small")

# Approximate USD per 1M tokens. Override via env for your model / plan.
PRICE_IN = float(os.getenv("SOKRAT_PRICE_IN", "0.15"))
PRICE_OUT = float(os.getenv("SOKRAT_PRICE_OUT", "0.60"))


class LLMRefusal(RuntimeError):
    """Raised when the model refuses to answer (handled as a first-class error)."""


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * PRICE_IN
            + self.completion_tokens / 1_000_000 * PRICE_OUT
        )

    def summary(self) -> str:
        return (
            f"{self.calls} call(s), {self.cache_hits} cache hit(s), "
            f"{self.total_tokens:,} tokens, ~${self.cost_usd:.4f}"
        )


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        *,
        cache_dir: str | os.PathLike = ".sokrat/cache",
        use_cache: bool = True,
        max_retries: int = 4,
    ) -> None:
        # Imported lazily so the package imports (and tests collect) without the
        # SDK installed or an API key present.
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or DEFAULT_MODEL
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.usage = Usage()
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- caching ----------------------------------------------------------
    def _key(self, *parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _retry(self, fn):
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn()
            except LLMRefusal:
                raise
            except Exception as err:  # rate limit / timeout / 5xx
                last_err = err
                if attempt == self.max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} tries: {last_err}")

    # -- structured output ------------------------------------------------
    def parse(self, *, system: str, user: str, schema: Type[T], temperature: float = 0.3) -> T:
        """Return an instance of `schema`, produced with strict structured output."""
        key = self._key("parse", self.model, schema.__name__, system, user, temperature)
        if self.use_cache and (p := self._cache_path(key)).exists():
            self.usage.cache_hits += 1
            return schema.model_validate_json(p.read_text(encoding="utf-8"))

        def _do() -> T:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
                temperature=temperature,
            )
            msg = completion.choices[0].message
            if getattr(msg, "refusal", None):
                raise LLMRefusal(msg.refusal)
            self.usage.calls += 1
            if completion.usage:
                self.usage.prompt_tokens += completion.usage.prompt_tokens
                self.usage.completion_tokens += completion.usage.completion_tokens
            return msg.parsed  # type: ignore[return-value]

        result = self._retry(_do)
        if self.use_cache:
            self._cache_path(key).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (cached per text)."""
        vectors: list[list[float] | None] = [None] * len(texts)
        to_fetch: list[int] = []
        for i, text in enumerate(texts):
            key = self._key("embed", EMBED_MODEL, text)
            if self.use_cache and (p := self._cache_path(key)).exists():
                self.usage.cache_hits += 1
                vectors[i] = json.loads(p.read_text(encoding="utf-8"))
            else:
                to_fetch.append(i)

        if to_fetch:
            def _do():
                return self.client.embeddings.create(
                    model=EMBED_MODEL, input=[texts[i] for i in to_fetch]
                )

            resp = self._retry(_do)
            self.usage.calls += 1
            if resp.usage:
                self.usage.prompt_tokens += resp.usage.prompt_tokens
            for slot, item in zip(to_fetch, resp.data):
                vectors[slot] = item.embedding
                if self.use_cache:
                    key = self._key("embed", EMBED_MODEL, texts[slot])
                    self._cache_path(key).write_text(
                        json.dumps(item.embedding), encoding="utf-8"
                    )
        return [v for v in vectors if v is not None]  # type: ignore[misc]

    # -- tool-calling chat (the agent's engine) ---------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.4,
    ):
        """One turn of a tool-calling conversation. Returns the raw message object.

        Not cached — the agent loop is stateful and each turn is unique.
        """

        def _do():
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
            return self.client.chat.completions.create(**kwargs)

        completion = self._retry(_do)
        self.usage.calls += 1
        if completion.usage:
            self.usage.prompt_tokens += completion.usage.prompt_tokens
            self.usage.completion_tokens += completion.usage.completion_tokens
        return completion.choices[0].message
