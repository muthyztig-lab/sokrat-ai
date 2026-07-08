from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as err:
            raise RuntimeError("Reading PDF files requires `pip install pypdf`.") from err
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return ""


def chunk_text(text: str, *, target_chars: int = 1100, overlap: int = 150) -> list[str]:
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
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
            while len(current) > target_chars * 1.5:
                chunks.append(current[:target_chars])
                current = current[target_chars - overlap :]
    if current:
        chunks.append(current)
    return chunks


def ingest_path(path: str | Path) -> list[Chunk]:
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
