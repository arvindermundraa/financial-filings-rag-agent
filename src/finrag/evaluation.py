"""Evaluation harness.

Retrieval (offline, no API key):
  For each of the 150 FinanceBench questions, retrieve chunks, collapse them to unique filing
  pages in rank order, and check whether a gold evidence page appears.
    hit@k  - share of questions with a gold page in the top k pages
    MRR@10 - mean of 1 / rank of the first gold page (0 if not in the top 10)
  Two scopes:
    all     - search all 84 filings at once (hard: thousands of similar-looking pages)
    filing  - search only the filing the question is about (what the agent does when the
              question names the company and year)

Answers (needs ANTHROPIC_API_KEY):
  Runs the agent, then a separate LLM call grades each answer against the gold answer as
  correct / incorrect / refused, and checks whether the answer cites a gold page.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .index import METHODS, SearchIndex
from .query import Router, process

KS = (1, 3, 5, 10)


def gold_pages(q: dict) -> set[tuple[str, int]]:
    return {(e["doc_name"], int(e["evidence_page_num"])) for e in q["evidence"]}


def ranked_pages(index: SearchIndex, query: str, method: str, doc_name: str | None,
                 company: str | None = None, depth: int = 10, pool: int = 60) -> list[tuple[str, int]]:
    seen, out = set(), []
    for h in index.search(query, k=pool, method=method, doc_name=doc_name, company=company, pool=pool):
        key = (h.chunk.doc_name, h.chunk.page)
        if key not in seen:
            seen.add(key)
            out.append(key)
            if len(out) == depth:
                break
    return out


DEFAULT_CONFIGS = [("raw", "all"), ("processed", "all"), ("processed", "routed"),
                   ("raw", "filing"), ("processed", "filing")]


def build_router(questions: list[dict]) -> Router:
    """Company display names come from the dataset's company field (a filing registry in practice)."""
    from .query import company_key
    names = {q["company"]: company_key(q["doc_name"]) for q in questions}
    return Router([q["doc_name"] for q in questions], names)


def evaluate_retrieval(index: SearchIndex, questions: list[dict],
                       methods: tuple[str, ...] = METHODS,
                       configs: list[tuple[str, str]] | None = None) -> dict:
    """configs: (query_mode, scope) pairs.
    query_mode: raw = question as written, processed = cleaned + finance synonyms
    scope: all = every filing, routed = filter read from the question by the Router,
           filing = the question's true filing (upper bound for routing)"""
    configs = configs or DEFAULT_CONFIGS
    router = build_router(questions)
    report: dict = {"n_questions": len(questions), "n_chunks": len(index.chunks),
                    "n_pages": len({(c.doc_name, c.page) for c in index.chunks}),
                    "n_filings": len({c.doc_name for c in index.chunks}), "results": []}
    routing = [router.route(q["question"]) for q in questions]
    report["routing"] = {
        "doc_exact": round(sum(r.doc_name == q["doc_name"] for r, q in zip(routing, questions)) / len(questions), 4),
        "company_found": round(sum(r.company is not None for r in routing) / len(questions), 4),
    }
    for mode, scope in configs:
        for method in methods:
            hits = {k: 0 for k in KS}
            rr, latencies = [], []
            by_type: dict[str, list[int]] = defaultdict(list)
            for q in questions:
                gold = gold_pages(q)
                start = time.perf_counter()
                query = process(q["question"]) if mode == "processed" else q["question"]
                route = routing[questions.index(q)] if scope == "routed" else None
                doc = q["doc_name"] if scope == "filing" else (route.doc_name if route else None)
                company = route.company if route and not doc else None
                pages = ranked_pages(index, query, method, doc, company)
                latencies.append((time.perf_counter() - start) * 1000)
                first = next((i for i, p in enumerate(pages, start=1) if p in gold), None)
                for k in KS:
                    hits[k] += int(first is not None and first <= k)
                rr.append(1.0 / first if first else 0.0)
                by_type[q["question_type"]].append(int(first is not None and first <= 5))
            n = len(questions)
            latencies.sort()
            report["results"].append({
                "query": mode, "scope": scope, "method": method,
                **{f"hit@{k}": round(hits[k] / n, 4) for k in KS},
                "mrr@10": round(sum(rr) / n, 4),
                "hit@5_by_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(by_type.items())},
                "latency_ms_p50": round(statistics.median(latencies), 1),
                "latency_ms_p95": round(latencies[int(0.95 * (n - 1))], 1),
            })
    return report


def retrieval_markdown(report: dict) -> str:
    lines = [f"Questions: {report['n_questions']} | filings: {report['n_filings']} | "
             f"pages: {report['n_pages']} | chunks: {report['n_chunks']}", "",
             f"Router: exact filing {report['routing']['doc_exact']:.0%}, "
             f"company found {report['routing']['company_found']:.0%}", "",
             "| query | scope | method | hit@1 | hit@3 | hit@5 | hit@10 | MRR@10 | p50 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        lines.append(f"| {r['query']} | {r['scope']} | {r['method']} | {r['hit@1']:.3f} | {r['hit@3']:.3f} | "
                     f"{r['hit@5']:.3f} | {r['hit@10']:.3f} | {r['mrr@10']:.3f} | {r['latency_ms_p50']} |")
    return "\n".join(lines) + "\n"


# ---------------- answer quality (LLM-as-judge) ----------------
JUDGE_PROMPT = """You grade answers to financial questions against a reference answer.
Question: {question}
Reference answer: {gold}
Candidate answer: {answer}

Reply with exactly one word:
CORRECT if the candidate gives the same figure or conclusion as the reference (small rounding differences are fine),
REFUSED if the candidate says it could not find the answer,
INCORRECT otherwise."""


def judge(client, model: str, question: str, gold: str, answer: str) -> str:
    resp = client.messages.create(model=model, max_tokens=5, temperature=0, messages=[
        {"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, answer=answer)}])
    word = resp.content[0].text.strip().upper()
    return word if word in {"CORRECT", "INCORRECT", "REFUSED"} else "INCORRECT"


def evaluate_answers(agent, questions: list[dict], out_path: Path, judge_model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    rows, counts = [], defaultdict(int)
    for q in questions:
        start = time.perf_counter()
        res = agent.ask(q["question"])
        latency = (time.perf_counter() - start) * 1000
        verdict = judge(client, judge_model, q["question"], q["answer"], res.answer)
        gold = {f"{d} p.{p}" for d, p in gold_pages(q)}
        counts[verdict] += 1
        rows.append({"id": q["financebench_id"], "verdict": verdict,
                     "cites_gold_page": bool(gold & set(res.citations)),
                     "unsupported_citations": res.unsupported_citations,
                     "steps": res.steps, "latency_ms": round(latency),
                     "tokens": res.usage, "answer": res.answer, "gold": q["answer"]})
    n = len(rows)
    summary = {
        "n": n,
        "accuracy": round(counts["CORRECT"] / n, 4),
        "refusal_rate": round(counts["REFUSED"] / n, 4),
        "incorrect_rate": round(counts["INCORRECT"] / n, 4),
        "gold_citation_rate": round(sum(r["cites_gold_page"] for r in rows) / n, 4),
        "unsupported_citation_rate": round(sum(bool(r["unsupported_citations"]) for r in rows) / n, 4),
        "latency_ms_p50": statistics.median(r["latency_ms"] for r in rows),
        "avg_steps": round(statistics.mean(r["steps"] for r in rows), 2),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return summary
