"""Download FinanceBench (questions + the SEC filings they cite) and extract page text.

FinanceBench (Islam et al., 2023, Patronus AI) is 150 open-source questions over 84 public
10-K / 10-Q / 8-K filings from 32 companies. Each question comes with a gold answer and the
filing page(s) that contain the evidence. The data is CC BY-NC 4.0, so this repo downloads it
instead of re-hosting it.
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from . import config


def load_questions(path: Path = config.QUESTIONS_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def download(workers: int = 8) -> None:
    """Fetch the question file and every filing it references."""
    _fetch(f"{config.FINANCEBENCH_RAW}/data/financebench_open_source.jsonl", config.QUESTIONS_PATH)
    docs = sorted({q["doc_name"] for q in load_questions()})
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda d: _fetch(f"{config.FINANCEBENCH_RAW}/pdfs/{d}.pdf",
                                       config.PDF_DIR / f"{d}.pdf"), docs))
    print(f"downloaded {len(docs)} filings to {config.PDF_DIR}")


def _extract(pdf_path: str) -> list[dict]:
    from pypdf import PdfReader

    doc = Path(pdf_path).stem
    reader = PdfReader(pdf_path)
    # Page numbers are 0-based to match FinanceBench's evidence_page_num.
    return [{"doc_name": doc, "page": i, "text": page.extract_text() or ""}
            for i, page in enumerate(reader.pages)]


def extract_pages(workers: int = 2) -> int:
    """Write one JSON line per filing page to data/pages.jsonl."""
    pdfs = sorted(str(p) for p in config.PDF_DIR.glob("*.pdf"))
    n = 0
    with ProcessPoolExecutor(workers) as pool, open(config.PAGES_PATH, "w") as out:
        for pages in pool.map(_extract, pdfs):
            for p in pages:
                out.write(json.dumps(p) + "\n")
                n += 1
    print(f"extracted {n} pages from {len(pdfs)} filings")
    return n


def load_pages(path: Path = config.PAGES_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
