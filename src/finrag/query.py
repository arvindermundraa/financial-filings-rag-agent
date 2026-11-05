"""Query processing before retrieval.

1. clean()   - drop instruction boilerplate ("Assume that you are a public equities analyst...")
               that matches nothing useful in a filing.
2. expand()  - add the line-item wording filings actually use. Analysts say "capex"; a cash
               flow statement says "purchases of property, plant and equipment". BM25 cannot
               bridge that gap on its own.
3. route()   - read the company and fiscal period out of the question and turn them into a
               metadata filter, so the search runs over one filing instead of all 84.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BOILERPLATE = [
    r"assume (that )?you are an? [^.?]*?analyst[.,]?",
    r"answer the following question( by)?",
    r"give a response to the question by",
    r"(please )?(by )?(primarily )?(relying on|using|based on) (the )?(details|information) (that is )?shown in (the )?",
    r"if [^.?]*(is not|isn't) (a )?(useful|relevant)[^.?]*[.?]",
    r"please state that and explain why\.?",
    r"round your answer[^.?]*[.?]",
    r"answer in [^.?]*[.?]",
]

# Analyst wording -> wording used in filings. Keys are matched as whole phrases.
SYNONYMS = {
    "capital expenditure": "purchases of property plant and equipment capital spending",
    "capex": "purchases of property plant and equipment capital expenditures",
    "cash flow statement": "consolidated statements of cash flows operating investing financing activities",
    "balance sheet": "consolidated balance sheets total assets total liabilities equity",
    "income statement": "consolidated statements of income operations net sales revenue",
    "p&l": "consolidated statements of income operations",
    "statement of income": "consolidated statements of income operations",
    "revenue": "net sales revenues",
    "cogs": "cost of sales cost of goods sold",
    "gross margin": "gross profit cost of sales net sales",
    "operating margin": "operating income net sales",
    "operating income": "operating income operating profit",
    "ebitda": "operating income depreciation and amortization",
    "d&a": "depreciation and amortization",
    "free cash flow": "net cash provided by operating activities purchases of property plant and equipment",
    "quick ratio": "cash and cash equivalents accounts receivable total current liabilities",
    "current ratio": "total current assets total current liabilities",
    "working capital": "total current assets total current liabilities",
    "inventory turnover": "inventories cost of sales",
    "dividend": "cash dividends paid per share",
    "dividends": "cash dividends paid per share",
    "net income": "net income attributable earnings",
    "eps": "earnings per share diluted basic",
    "debt": "long-term debt notes borrowings",
    "leverage": "long-term debt total equity",
    "segment": "segment results operating segments",
    "effective tax rate": "effective tax rate provision for income taxes",
    "share repurchase": "repurchases of common stock treasury",
    "buyback": "repurchases of common stock treasury",
    "roa": "net income total assets",
    "return on assets": "net income total assets",
    "days payable": "accounts payable cost of sales",
    "days sales outstanding": "accounts receivable net sales",
}
_SYN_RE = re.compile(r"(?<![a-z])(" + "|".join(re.escape(k) for k in sorted(SYNONYMS, key=len, reverse=True)) + r")(?![a-z])")


def clean(question: str) -> str:
    q = question
    for pat in _BOILERPLATE:
        q = re.sub(pat, " ", q, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", q).strip()


def expand(query: str) -> str:
    found = []
    for m in _SYN_RE.finditer(query.lower()):
        extra = SYNONYMS[m.group(1)]
        if extra not in found:
            found.append(extra)
    return query if not found else f"{query} {' '.join(found)}"


def process(question: str) -> str:
    return expand(clean(question))


# ---------------- metadata routing ----------------
def company_key(doc_name: str) -> str:
    """'JOHNSON_JOHNSON_2022_10K' -> 'JOHNSONJOHNSON', 'Pfizer_2023Q2_10Q' -> 'PFIZER'."""
    m = re.match(r"^(.*?)_(\d{4})", doc_name)
    return re.sub(r"[^A-Z0-9]", "", (m.group(1) if m else doc_name).upper())


def norm_company(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


_DOC_RE = re.compile(r"^(?P<co>.*?)_(?P<year>\d{4})(?:Q(?P<q>\d))?_(?P<kind>[A-Za-z0-9]+)")

# Common short names analysts use. Added on top of the official names in the filing index.
EXTRA_ALIASES = {"J&J": "JOHNSONJOHNSON", "AMEX": "AMERICANEXPRESS", "JPM": "JPMORGAN",
                 "COKE": "COCACOLA", "MGM": "MGMRESORTS", "ULTA": "ULTABEAUTY",
                 "CVS": "CVSHEALTH", "LOCKHEED": "LOCKHEEDMARTIN", "KRAFT": "KRAFTHEINZ",
                 "ACTIVISION": "ACTIVISIONBLIZZARD", "SQUARE": "BLOCK"}


@dataclass
class Route:
    company: str | None = None
    doc_name: str | None = None


class Router:
    def __init__(self, doc_names: list[str], company_names: dict[str, str]):
        """company_names maps a display name ('Johnson & Johnson') to its key ('JOHNSONJOHNSON')."""
        self.docs: dict[str, list[dict]] = {}
        for d in sorted(set(doc_names)):
            m = _DOC_RE.match(d)
            if not m:
                continue
            self.docs.setdefault(company_key(d), []).append(
                {"doc": d, "year": int(m["year"]), "q": int(m["q"]) if m["q"] else None,
                 "kind": m["kind"].upper()})
        aliases = {**{n: k for n, k in company_names.items()}, **EXTRA_ALIASES}
        # longest alias first so "American Water Works" beats "American"
        self.aliases = sorted(((a, k) for a, k in aliases.items() if k in self.docs),
                              key=lambda x: -len(x[0]))

    def _company(self, question: str) -> str | None:
        for alias, key in self.aliases:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", question, re.I):
                return key
        return None

    def route(self, question: str) -> Route:
        key = self._company(question)
        if not key:
            return Route()
        years = [int(y) for y in re.findall(r"(?:FY\s?'?|\b)(20\d{2})\b", question, re.I)]
        q_match = re.search(r"\bQ([1-4])\b", question, re.I)
        quarter = int(q_match.group(1)) if q_match else None
        candidates = self.docs[key]
        if years:
            year = max(years)  # "FY2022 vs FY2021" questions live in the later filing
            candidates = [c for c in candidates if c["year"] == year]
        if quarter:
            candidates = [c for c in candidates if c["q"] == quarter] or candidates
        else:
            candidates = [c for c in candidates if c["kind"] == "10K"] or candidates
        if years and len(candidates) == 1:
            return Route(company=key, doc_name=candidates[0]["doc"])
        return Route(company=key)
