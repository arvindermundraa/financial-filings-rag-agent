from finrag.evaluation import evaluate_retrieval, gold_pages, retrieval_markdown


def q(question, doc, page, qtype="metrics-generated"):
    return {"financebench_id": question, "question": question, "doc_name": doc, "company": doc.split("_")[0].title(),
            "question_type": qtype, "answer": "x",
            "evidence": [{"doc_name": doc, "evidence_page_num": page}]}


def test_gold_pages():
    assert gold_pages(q("a", "3M_2018_10K", 59)) == {("3M_2018_10K", 59)}


def test_retrieval_report_shape_and_bounds(tiny_index):
    questions = [q("purchases of property plant and equipment cash flow", "3M_2018_10K", 59),
                 q("Creative Cloud subscription revenue", "ADOBE_2017_10K", 40, "novel-generated")]
    report = evaluate_retrieval(tiny_index, questions)
    assert report["n_questions"] == 2
    assert len(report["results"]) == 15  # 5 configs x 3 methods
    for r in report["results"]:
        assert 0 <= r["hit@1"] <= r["hit@3"] <= r["hit@5"] <= r["hit@10"] <= 1
        assert 0 <= r["mrr@10"] <= 1
    filing_bm25 = next(r for r in report["results"] if r["scope"] == "filing" and r["method"] == "bm25" and r["query"] == "raw")
    assert filing_bm25["hit@1"] == 1.0
    assert "| raw | filing | bm25 |" in retrieval_markdown(report)
