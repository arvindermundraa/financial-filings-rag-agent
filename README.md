# Financial Filings RAG Agent

Ask questions about public companies' SEC filings and get answers that cite the exact filing page.
A tool-calling LLM agent searches 84 real 10-K / 10-Q / 8-K filings (about 12,000 pages), does its
arithmetic with a calculator tool instead of guessing, and has to cite pages it actually retrieved.
Retrieval is measured against **FinanceBench**, a public benchmark of 150 analyst questions with
gold answers and gold evidence pages.

```
question ─► query processing ─► router ─► hybrid retrieval ─► agent loop ─► answer + [doc p.N] citations
            (clean + finance     (company   (BM25 + dense,     (search_filings,
             synonyms)            + period)   RRF fusion)        calculator, step limit)
```

## Results (retrieval, offline, reproducible)

150 FinanceBench questions, 84 filings, 11,948 pages, 34,113 chunks. A hit means a gold evidence
page is among the top-k *pages* returned. Full table: [`results/retrieval_eval.md`](results/retrieval_eval.md).

| setup | hit@1 | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|---|
| Raw question, search all filings, hybrid | 0.053 | 0.113 | 0.160 | 0.082 |
| + query cleaning and finance synonyms | 0.040 | 0.140 | 0.213 | 0.084 |
| + company / fiscal-period router (**default pipeline**) | **0.287** | **0.560** | **0.627** | **0.407** |
| Upper bound: true filing given, processed query, hybrid | 0.340 | 0.687 | 0.767 | 0.495 |

What the numbers say:
- **Searching everything at once fails.** Filings from different companies and years use the same
  headings and line items, so the right page is buried among thousands of near-duplicates.
- **Routing is the biggest lever.** Reading the company and fiscal period out of the question and
  filtering to that filing raises hit@5 from 0.14 to 0.56. The router finds the company for 89% of
  questions and the exact filing for 68%; the remaining gap to the 0.687 upper bound is routing misses.
- **Query processing matters once the search is scoped.** With the true filing given, cleaning plus
  finance synonyms ("capex" → "purchases of property, plant and equipment") lifts hybrid hit@5 from
  0.393 to 0.687.
- **Hybrid wins on ranking quality.** In the routed setting, hybrid has the best hit@1 and MRR. BM25
  alone is close because filings reuse exact line-item wording.
- Retrieval latency is a few milliseconds per query on a laptop CPU (p50 under 4 ms).

Caveat: the synonym list was written from standard financial-statement vocabulary, not tuned on
individual questions, but there is no separate held-out set; all numbers are on the same 150 questions.

## Answer quality (needs an API key)

`python -m finrag.cli eval-answers` runs the agent on every question, then a separate LLM call grades
each answer against the gold answer (CORRECT / INCORRECT / REFUSED). It also reports how often the
answer cites a gold page and how often the model cited a page it never retrieved (these citations are
stripped and flagged). Results are written to `results/answer_eval.json`.

## How it works

| Piece | File | What it does |
|---|---|---|
| Data | `src/finrag/data.py` | Downloads FinanceBench and the 84 filings, extracts text per page |
| Chunking | `src/finrag/chunking.py` | 250-word windows with 50-word overlap; each chunk keeps doc + page |
| BM25 | `src/finrag/index.py` | Okapi BM25 stored as a sparse matrix of term weights |
| Dense | `src/finrag/index.py` | TF-IDF (1-2 grams) reduced to 256 dims with truncated SVD, cosine similarity; fully offline |
| Fusion | `src/finrag/index.py` | Reciprocal rank fusion (k = 60) of the two ranked lists |
| Query processing | `src/finrag/query.py` | Strips prompt boilerplate, adds filing wording for analyst terms |
| Router | `src/finrag/query.py` | Maps company names and FY / quarter mentions to a filing filter |
| Agent | `src/finrag/agent.py` | Claude tool-use loop: `search_filings`, `calculator`; step limit; citation check |
| Calculator | `src/finrag/calculator.py` | AST-based evaluator: numbers and + - * / ** % only |
| API | `src/finrag/api.py` | FastAPI: `/health`, `/search` (no key), `/ask` (agent) |
| Evaluation | `src/finrag/evaluation.py` | hit@k, MRR@10, per question type, latency; LLM-as-judge for answers |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
make setup          # pip install -r requirements-dev.txt
make data           # download FinanceBench + filings (~165 MB) and extract pages
make index          # build BM25 + dense index (~1.5 min)
make eval           # retrieval metrics -> results/
make test           # 51 unit tests, no network or API key needed
make serve          # http://localhost:8000/docs

export ANTHROPIC_API_KEY=...
PYTHONPATH=src python -m finrag.cli ask "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
```

## Deploy to Google Cloud Run

The Docker image downloads the data and builds the index at build time, so the container starts
ready to serve.

```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/finrag
gcloud run deploy finrag --image gcr.io/$PROJECT_ID/finrag --region us-central1 \
  --memory 2Gi --allow-unauthenticated --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

## Limitations and next steps

- Dense vectors are LSA, chosen so everything runs offline. A neural embedding model plus a
  cross-encoder reranker should close part of the gap at hit@1.
- Tables lose their structure in PDF text extraction; table-aware parsing would help numeric questions.
- The router uses names and years in the question. Questions with no company name fall back to
  searching everything.

## Data

FinanceBench: Islam et al., 2023, *FinanceBench: A New Benchmark for Financial Question Answering*,
Patronus AI. Licensed CC BY-NC 4.0; this repo downloads it and does not redistribute it.
