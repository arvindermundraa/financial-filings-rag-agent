from finrag.query import Router, clean, company_key, expand, process

ROUTER = Router(
    ["3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q", "JOHNSON_JOHNSON_2022_10K",
     "AMERICANWATERWORKS_2021_10K", "AMERICANWATERWORKS_2022_10K", "Pfizer_2023Q2_10Q"],
    {"3M": "3M", "Johnson & Johnson": "JOHNSONJOHNSON", "American Water Works": "AMERICANWATERWORKS",
     "Pfizer": "PFIZER"},
)


def test_clean_removes_instruction_boilerplate():
    q = ("Assume that you are a public equities analyst. Answer the following question by primarily "
         "using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M?")
    out = clean(q)
    assert "analyst" not in out and "Answer the following" not in out
    assert "balance sheet" in out and "FY2018" in out


def test_expand_adds_filing_wording():
    out = expand("What was FY2018 capex for 3M?")
    assert "purchases of property plant and equipment" in out


def test_expand_matches_whole_phrases_only():
    assert expand("Describe the prevenue model") == "Describe the prevenue model"


def test_process_keeps_question_content():
    assert "3M" in process("What is the FY2018 capital expenditure amount for 3M?")


def test_company_key():
    assert company_key("JOHNSON_JOHNSON_2022_10K") == "JOHNSONJOHNSON"
    assert company_key("Pfizer_2023Q2_10Q") == "PFIZER"


def test_route_company_and_year_to_10k():
    r = ROUTER.route("What is the FY2018 capital expenditure amount for 3M?")
    assert r.company == "3M" and r.doc_name == "3M_2018_10K"


def test_route_quarter_to_10q():
    assert ROUTER.route("3M quick ratio for Q2 of FY2023?").doc_name == "3M_2023Q2_10Q"


def test_route_uses_later_year_in_comparisons():
    r = ROUTER.route("How did American Water Works revenue change from FY2021 to FY2022?")
    assert r.doc_name == "AMERICANWATERWORKS_2022_10K"


def test_route_company_only_when_no_year():
    r = ROUTER.route("Is 3M a capital-intensive business?")
    assert r.company == "3M" and r.doc_name is None


def test_route_aliases_and_no_match():
    assert ROUTER.route("J&J FY2022 dividends?").doc_name == "JOHNSON_JOHNSON_2022_10K"
    assert ROUTER.route("What was revenue last year?").company is None
