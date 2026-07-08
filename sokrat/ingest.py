"""Load course materials from disk and split them into retrievable chunks.

Supported formats: .txt, .md, .pdf (PDF needs `pypdf`). Everything else is
skipped with a warning so one bad file never breaks an ingest run.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def read_file(path: Path) -> str:
    """Return the plain text of a single supported file (or '' if unsupported)."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as err:  # pragma: no cover - depends on optional dep
            raise RuntimeError("Reading PDF files requires `pip install pypdf`.") from err
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return ""


def chunk_text(text: str, *, target_chars: int = 1100, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries.

    Overlap keeps context from spilling across a hard cut, which improves
    retrieval quality for questions that straddle two paragraphs.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= target_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # Carry a little overlap from the tail of the previous chunk.
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
            # A single very long paragraph still needs hard splitting.
            while len(current) > target_chars * 1.5:
                chunks.append(current[:target_chars])
                current = current[target_chars - overlap :]
    if current:
        chunks.append(current)
    return chunks


def ingest_path(path: str | Path) -> list[Chunk]:
    """Walk a file or directory and return all chunks, tagged with their source."""
    root = Path(path)
    files: list[Path]
    if root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file())
    else:
        files = [root]

    chunks: list[Chunk] = []
    for file in files:
        try:
            raw = read_file(file)
        except RuntimeError as err:
            print(f"  ! skipped {file.name}: {err}")
            continue
        if not raw.strip():
            continue
        for piece in chunk_text(raw):
            chunks.append(Chunk(id=len(chunks), source=file.name, text=piece))
    return chunks
