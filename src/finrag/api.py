"""FastAPI service.

GET  /health          liveness check
POST /search          retrieval only (no LLM, no API key needed)
POST /ask             full agent answer with citations (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .agent import FilingsAgent
from .index import METHODS, SearchIndex
from .query import process

app = FastAPI(title="Financial Filings RAG Agent", version="1.0.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=20)
    method: str = "hybrid"
    doc_name: str | None = None
    company: str | None = None
    process_query: bool = True   # clean + finance synonyms
    auto_route: bool = True      # read company / fiscal period from the query if no filter given


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@lru_cache(maxsize=1)
def get_index() -> SearchIndex:
    return SearchIndex.load(config.INDEX_DIR)


@lru_cache(maxsize=1)
def get_router():
    from .data import load_questions
    from .evaluation import build_router
    return build_router(load_questions()) if config.QUESTIONS_PATH.exists() else None


@lru_cache(maxsize=1)
def get_agent() -> FilingsAgent:
    from .llm import ClaudeModel
    return FilingsAgent(get_index(), ClaudeModel(), router=get_router())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    if req.method not in METHODS:
        raise HTTPException(422, f"method must be one of {METHODS}")
    start = time.perf_counter()
    doc, company = req.doc_name, req.company
    router = get_router() if req.auto_route and not (doc or company) else None
    if router:
        route = router.route(req.query)
        doc, company = route.doc_name, route.company
    query = process(req.query) if req.process_query else req.query
    hits = get_index().search(query, k=req.k, method=req.method, doc_name=doc, company=company)
    return {
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "filter": {"doc_name": doc, "company": company},
        "results": [{"citation": h.chunk.citation, "rank": h.rank, "score": h.score,
                     "text": h.chunk.text} for h in hits],
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        agent = get_agent()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    start = time.perf_counter()
    result = agent.ask(req.question)
    return {"latency_ms": round((time.perf_counter() - start) * 1000, 1), **result.__dict__}
