"""Tool-calling agent that answers questions about SEC filings.

Loop:
  1. Send the question plus two tool definitions to the model.
  2. If the model asks for a tool, run it and send the result back.
  3. Stop when the model gives a final answer or the step limit is hit.

Tools:
  search_filings(query, doc_name?, company?) - hybrid BM25 + dense retrieval, returns chunks
                                              tagged with [doc_name p.N] citations
  calculator(expression)                     - safe arithmetic for ratios and % changes

Guardrails: a hard step limit, answers must cite pages that were actually retrieved (any
citation the model invents is dropped and flagged), and the model is told to say it could
not find the answer instead of guessing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import config
from .calculator import CalculatorError, evaluate
from .index import SearchIndex
from .llm import ChatModel
from .query import Router, process

SYSTEM_PROMPT = """You answer questions about public companies' SEC filings (10-K, 10-Q, 8-K).
Rules:
- Use the search_filings tool to find evidence. Search again with different wording if the first results do not contain the figure you need.
- If the question names a specific filing, pass its doc_name (for example 3M_2018_10K) to narrow the search.
- Use the calculator tool for every arithmetic step. Do not do arithmetic in your head.
- Answer only from retrieved text. Cite every figure with the bracketed source tag exactly as it appears, e.g. [3M_2018_10K p.59].
- If the filings you retrieved do not contain the answer, say "I could not find this in the filings." Do not guess.
- Keep the final answer short: the answer first, then one or two sentences showing where it came from."""

TOOLS = [
    {
        "name": "search_filings",
        "description": "Search SEC filing pages. Returns the most relevant text chunks, each tagged with a citation like [DOC_NAME p.N].",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'purchases of property, plant and equipment 2018'"},
                "doc_name": {"type": "string", "description": "Optional filing id to restrict the search, e.g. 3M_2018_10K"},
                "company": {"type": "string", "description": "Optional company id to restrict the search, e.g. 3M"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression with + - * / ** % and parentheses. Example: (1577 / 32765) * 100",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

_CITE = re.compile(r"\[([A-Za-z0-9&\-]+_\d{4}(?:Q\d)?_[A-Za-z0-9\-]+) p\.(\d+)\]")


@dataclass
class AgentResult:
    answer: str
    citations: list[str] = field(default_factory=list)          # cited AND retrieved
    unsupported_citations: list[str] = field(default_factory=list)  # cited but never retrieved
    retrieved: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    steps: int = 0
    stopped_early: bool = False
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


class FilingsAgent:
    def __init__(self, index: SearchIndex, model: ChatModel, k: int = 5,
                 max_steps: int = config.AGENT_MAX_STEPS, chunk_chars: int = 1500,
                 router: Router | None = None):
        self.index = index
        self.router = router
        self.model = model
        self.k = k
        self.max_steps = max_steps
        self.chunk_chars = chunk_chars

    # ---------- tools ----------
    def _search(self, args: dict, retrieved: list[str]) -> str:
        doc, company = args.get("doc_name") or None, args.get("company") or None
        if not doc and not company and self.router:
            # the model did not filter: fall back to the company/period read from the query
            route = self.router.route(args["query"])
            doc, company = route.doc_name, route.company
        hits = self.index.search(process(args["query"]), k=self.k, method="hybrid",
                                 doc_name=doc, company=company)
        if not hits:
            return "No results. Try a broader query or remove the filter."
        parts = []
        for h in hits:
            retrieved.append(h.chunk.citation)
            parts.append(f"[{h.chunk.citation}]\n{h.chunk.text[:self.chunk_chars]}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _calculate(args: dict) -> str:
        try:
            return repr(round(evaluate(args["expression"]), 6))
        except CalculatorError as e:
            return f"error: {e}"

    def _run_tool(self, name: str, args: dict, retrieved: list[str]) -> str:
        if name == "search_filings":
            return self._search(args, retrieved)
        if name == "calculator":
            return self._calculate(args)
        return f"error: unknown tool {name}"

    # ---------- loop ----------
    def ask(self, question: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": question}]
        result = AgentResult(answer="")
        for step in range(1, self.max_steps + 1):
            result.steps = step
            resp = self.model(SYSTEM_PROMPT, messages, TOOLS)
            for key in result.usage:
                result.usage[key] += resp.get("usage", {}).get(key, 0)
            blocks = resp["content"]
            messages.append({"role": "assistant", "content": blocks})
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            if resp.get("stop_reason") != "tool_use" or not tool_uses:
                result.answer = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                break
            tool_results = []
            for b in tool_uses:
                output = self._run_tool(b["name"], b.get("input", {}), result.retrieved)
                result.tool_calls.append({"name": b["name"], "input": b.get("input", {})})
                tool_results.append({"type": "tool_result", "tool_use_id": b["id"], "content": output})
            messages.append({"role": "user", "content": tool_results})
        else:
            result.stopped_early = True
            result.answer = "I could not finish within the step limit."
        self._check_citations(result)
        return result

    @staticmethod
    def _check_citations(result: AgentResult) -> None:
        retrieved = set(result.retrieved)
        for doc, page in _CITE.findall(result.answer):
            cite = f"{doc} p.{page}"
            bucket = result.citations if cite in retrieved else result.unsupported_citations
            if cite not in bucket:
                bucket.append(cite)

    @staticmethod
    def to_json(result: AgentResult) -> str:
        return json.dumps(result.__dict__, indent=2)
