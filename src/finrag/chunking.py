"""Split filing pages into overlapping word windows.

Each chunk keeps its source (doc_name, page) so retrieval results can be cited and scored
against FinanceBench's gold evidence pages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    doc_name: str
    page: int
    text: str

    @property
    def company(self) -> str:
        from .query import company_key  # doc names look like "JOHNSON_JOHNSON_2022_10K"
        return company_key(self.doc_name)

    @property
    def citation(self) -> str:
        return f"{self.doc_name} p.{self.page}"


def normalize(text: str) -> str:
    """PDF extraction splits table cells onto separate lines; collapse all whitespace."""
    return _WS.sub(" ", text).strip()


def chunk_page(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    words = normalize(text).split(" ")
    if words == [""]:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in pages:
        for text in chunk_page(p["text"], size, overlap):
            chunks.append(Chunk(len(chunks), p["doc_name"], p["page"], text))
    return chunks
