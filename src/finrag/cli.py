"""Command line entry point.

  python -m finrag.cli download        # fetch FinanceBench questions + 84 filings
  python -m finrag.cli extract         # PDF pages -> data/pages.jsonl
  python -m finrag.cli build           # chunk + build BM25 and dense index
  python -m finrag.cli eval-retrieval  # offline retrieval metrics -> results/
  python -m finrag.cli ask "question"  # agent answer (needs ANTHROPIC_API_KEY)
  python -m finrag.cli eval-answers --limit 50
"""
from __future__ import annotations

import argparse
import json
import time

from . import config
from .chunking import chunk_pages
from .data import download, extract_pages, load_pages, load_questions
from .index import SearchIndex

RESULTS = config.ROOT / "results"


def main() -> None:
    p = argparse.ArgumentParser(prog="finrag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("extract")
    sub.add_parser("build")
    sub.add_parser("eval-retrieval")
    a = sub.add_parser("ask")
    a.add_argument("question")
    e = sub.add_parser("eval-answers")
    e.add_argument("--limit", type=int, default=150)
    e.add_argument("--judge-model", default=config.LLM_MODEL)
    args = p.parse_args()

    if args.cmd == "download":
        download()
    elif args.cmd == "extract":
        extract_pages()
    elif args.cmd == "build":
        start = time.perf_counter()
        chunks = chunk_pages(load_pages(), config.CHUNK_WORDS, config.CHUNK_OVERLAP)
        SearchIndex.build(chunks).save()
        print(f"indexed {len(chunks)} chunks in {time.perf_counter() - start:.0f}s")
    elif args.cmd == "eval-retrieval":
        from .evaluation import evaluate_retrieval, retrieval_markdown

        report = evaluate_retrieval(SearchIndex.load(), load_questions())
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "retrieval_eval.json").write_text(json.dumps(report, indent=2))
        md = retrieval_markdown(report)
        (RESULTS / "retrieval_eval.md").write_text(md)
        print(md)
    elif args.cmd == "ask":
        from .agent import FilingsAgent
        from .llm import ClaudeModel

        from .evaluation import build_router
        res = FilingsAgent(SearchIndex.load(), ClaudeModel(),
                           router=build_router(load_questions())).ask(args.question)
        print(FilingsAgent.to_json(res))
    elif args.cmd == "eval-answers":
        from .agent import FilingsAgent
        from .evaluation import evaluate_answers
        from .llm import ClaudeModel

        from .evaluation import build_router
        agent = FilingsAgent(SearchIndex.load(), ClaudeModel(), router=build_router(load_questions()))
        summary = evaluate_answers(agent, load_questions()[:args.limit],
                                   RESULTS / "answer_eval.json", args.judge_model)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
