# Interview guide: Financial Filings RAG Agent

How to explain this project in 30 seconds, in 3 minutes, and under follow-up questions.

## 30-second version

"I built a question-answering agent over SEC filings. It searches 84 real 10-Ks and 10-Qs with
hybrid retrieval, BM25 plus dense vectors fused with reciprocal rank fusion, and a Claude agent calls a
search tool and a calculator tool and has to cite the pages it used. I measured retrieval on
FinanceBench, 150 analyst questions with gold evidence pages. Searching all filings at once only
found the right page in the top 5 for 11% of questions. Adding a router that reads the company and
fiscal year from the question, plus finance synonym expansion, took that to 58%."

## 3-minute walkthrough (follow the data)

1. **Data.** FinanceBench gives questions, gold answers, and the exact filing page holding the
   evidence. I download the 84 filings and extract text page by page, about 12,000 pages.
2. **Chunking.** 250-word windows, 50 words of overlap, each chunk tagged with its filing and page.
   Overlap keeps a table row and its column headers together more often.
3. **Two retrievers.**
   - BM25 rewards exact term matches. Filings reuse exact line-item names, so it is strong here.
   - Dense: TF-IDF vectors compressed to 256 dimensions with truncated SVD (latent semantic analysis).
     Terms that co-occur end up close together, so it catches related wording BM25 misses. It runs
     fully offline.
4. **Fusion.** Reciprocal rank fusion: each chunk scores the sum of 1 / (60 + rank) across the two
   lists. It uses only ranks, so I never have to calibrate BM25 scores against cosine scores.
5. **Query processing.** Remove prompt boilerplate, then add the wording filings use: "capex"
   becomes "purchases of property, plant and equipment".
6. **Router.** Find the company name and FY or quarter in the question and filter to that filing.
   This was the biggest single gain, because pages from different companies look alike.
7. **Agent.** Claude gets two tools. `search_filings` runs the pipeline above. `calculator` does
   arithmetic through a restricted AST evaluator, because LLMs make multi-digit math mistakes.
   There is a step limit, and any citation to a page the agent never retrieved is flagged.
8. **Evaluation.** hit@k and MRR at page level for retrieval. For answers, an LLM judge compares to
   the gold answer, and I track refusals separately from wrong answers.

## Numbers to remember

| | hit@5 | MRR@10 |
|---|---|---|
| raw question, all filings, hybrid | 0.107 | 0.081 |
| processed + routed, hybrid (default) | 0.580 | 0.408 |
| true filing given (upper bound) | 0.720 | 0.511 |

Router: finds the company 89% of the time, the exact filing 68%. Retrieval p50 is under 4 ms.

## Likely questions and answers

**Why not just a vector database?** At this size (34k chunks) a NumPy matrix product is a few
milliseconds. A vector database adds value at millions of vectors or when you need filtering and
updates at scale. The index class is small enough to swap for pgvector or Qdrant.

**Why LSA instead of neural embeddings?** It keeps the project reproducible with no model download
or API key. The weakness is real: on raw queries over all filings, dense alone was worst. With a neural
embedding model I would expect better dense recall, and adding a cross-encoder reranker is next.

**Why does searching all filings do so badly?** 84 filings share the same structure. "Consolidated
Statements of Cash Flows" appears in every 10-K, so without a company and year the right page
competes with dozens of near-identical ones.

**How do you know the synonym list isn't overfit?** I wrote it from standard financial-statement
terms, not from failed questions, but there is no held-out split; I say that in the README. The
honest next step is a held-out set or cross-validation over question types.

**How do you stop hallucinated numbers?** The system prompt requires citations; the code checks each
cited page against what was actually retrieved and flags the rest; arithmetic goes through a tool; and
the model is told to say it could not find the answer rather than guess. Refusals are counted
separately from wrong answers in the evaluation.

**Why hit@k at page level, not chunk level?** FinanceBench labels pages. Several chunks can come from
one page, so I collapse chunks to unique pages before scoring, otherwise one page could fill the top 5.

**What is reciprocal rank fusion?** score(d) = sum over lists of 1 / (k + rank). k = 60 dampens the
gap between rank 1 and 2. A document ranked well by both retrievers rises to the top.

**How would you deploy and monitor it?** The Dockerfile builds the index into the image; Cloud Run
serves it. In production I would log query, route, retrieved pages, latency, tokens, and judge scores,
then alert on refusal rate and unsupported-citation rate the way I used SLI/SLO alerting at Dun &
Bradstreet.

**What would you improve next?** Table-aware PDF parsing, neural embeddings plus a reranker, a
held-out evaluation split, and letting the agent ask a clarifying question when the router finds no
company.

## Commands to demo live

```bash
make test                      # 51 tests
make eval                      # regenerates results/retrieval_eval.md
make serve                     # open http://localhost:8000/docs, try POST /search
```
