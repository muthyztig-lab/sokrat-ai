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

PRICE_IN = float(os.getenv("SOKRAT_PRICE_IN", "0.15"))
PRICE_OUT = float(os.getenv("SOKRAT_PRICE_OUT", "0.60"))


class LLMRefusal(RuntimeError):
    pass


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
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or DEFAULT_MODEL
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.usage = Usage()
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

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
            except Exception as err:
                last_err = err
                if attempt == self.max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} tries: {last_err}")

    def parse(self, *, system: str, user: str, schema: Type[T], temperature: float = 0.3) -> T:
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
            return msg.parsed

        result = self._retry(_do)
        if self.use_cache:
            self._cache_path(key).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def embed(self, texts: list[str]) -> list[list[float]]:
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
        return [v for v in vectors if v is not None]

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.4,
    ):
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
