import pytest

from finrag.chunking import Chunk, chunk_page, chunk_pages, normalize


def test_normalize_collapses_pdf_whitespace():
    assert normalize("Net sales\n \n32,765 \n\t31,657") == "Net sales 32,765 31,657"


def test_short_page_is_one_chunk():
    assert chunk_page("a b c", size=10, overlap=2) == ["a b c"]


def test_empty_page_has_no_chunks():
    assert chunk_page("  \n ", size=10, overlap=2) == []


def test_windows_overlap_and_cover_every_word():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_page(" ".join(words), size=10, overlap=3)
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_page("a b c", size=3, overlap=3)


def test_chunks_keep_source_and_citation():
    chunks = chunk_pages([{"doc_name": "3M_2018_10K", "page": 59, "text": "capex 1577"}], 10, 2)
    assert chunks == [Chunk(0, "3M_2018_10K", 59, "capex 1577")]
    assert chunks[0].company == "3M"
    assert chunks[0].citation == "3M_2018_10K p.59"
