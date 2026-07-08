from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ingest import ingest_path
from .llm import LLMClient
from .models import Chunk, Citation

INDEX_DIR = Path(".sokrat/index")


class Retriever:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, llm: LLMClient) -> None:
        self.chunks = chunks
        self.matrix = matrix
        self.llm = llm

    @classmethod
    def build(cls, source_path: str, llm: LLMClient) -> "Retriever":
        chunks = ingest_path(source_path)
        if not chunks:
            raise ValueError(f"No readable course material found at: {source_path}")
        vectors = llm.embed([c.text for c in chunks])
        matrix = _normalize(np.array(vectors, dtype=np.float32))
        return cls(chunks, matrix, llm)

    def save(self, index_dir: str | Path = INDEX_DIR) -> None:
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.matrix)
        (d / "chunks.json").write_text(
            json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, llm: LLMClient, index_dir: str | Path = INDEX_DIR) -> "Retriever":
        d = Path(index_dir)
        if not (d / "embeddings.npy").exists():
            raise FileNotFoundError(
                f"No index at {d}. Run `sokrat ingest <path>` first."
            )
        matrix = np.load(d / "embeddings.npy")
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text("utf-8"))]
        return cls(chunks, matrix, llm)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = _normalize(np.array(self.llm.embed([query])[0], dtype=np.float32)[None, :])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def citations(self, query: str, k: int = 4) -> list[Citation]:
        return [
            Citation(source=chunk.source, quote=_trim(chunk.text))
            for chunk, _ in self.search(query, k)
        ]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _trim(text: str, limit: int = 320) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"
