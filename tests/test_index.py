import pytest

from finrag.index import SearchIndex, rrf_fuse, tokenize


def test_tokenize_keeps_numbers_and_decimals():
    assert tokenize("Capex was $1,577.5M in FY2018") == ["capex", "was", "1", "577.5", "m", "in", "fy2018"]


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    fused = rrf_fuse([[1, 2, 3], [2, 1, 4]], k=60)
    assert {fused[0][0], fused[1][0]} == {1, 2}
    assert fused[-1][0] in {3, 4}


def test_rrf_single_list_keeps_order():
    assert [d for d, _ in rrf_fuse([[7, 3, 9]])] == [7, 3, 9]


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_each_method_returns_ranked_hits(tiny_index, method):
    hits = tiny_index.search("purchases of property plant and equipment", k=3, method=method)
    assert 1 <= len(hits) <= 3
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_bm25_finds_exact_line_item(tiny_index):
    top = tiny_index.search("purchases of property, plant and equipment (PP&E)", k=1, method="bm25")[0]
    assert top.chunk.doc_name == "3M_2018_10K" and top.chunk.page == 59


def test_doc_filter_restricts_results(tiny_index):
    hits = tiny_index.search("purchases of property", k=5, doc_name="ADOBE_2017_10K")
    assert hits and all(h.chunk.doc_name == "ADOBE_2017_10K" for h in hits)


def test_company_filter_restricts_results(tiny_index):
    hits = tiny_index.search("capital expenditures", k=5, company="amcor")
    assert hits and all(h.chunk.company == "AMCOR" for h in hits)


def test_unknown_method_rejected(tiny_index):
    with pytest.raises(ValueError):
        tiny_index.search("x", method="magic")


def test_save_and_load_round_trip(tiny_index, tmp_path):
    tiny_index.save(tmp_path)
    loaded = SearchIndex.load(tmp_path)
    q = "net sales operating income"
    assert [h.chunk.chunk_id for h in loaded.search(q, k=3)] == \
           [h.chunk.chunk_id for h in tiny_index.search(q, k=3)]
