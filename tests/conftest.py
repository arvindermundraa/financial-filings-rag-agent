import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag.chunking import chunk_pages  # noqa: E402
from finrag.index import SearchIndex  # noqa: E402

PAGES = [
    {"doc_name": "3M_2018_10K", "page": 59, "text": "Consolidated Statement of Cash Flows. "
     "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420). "
     "Net cash used in investing activities."},
    {"doc_name": "3M_2018_10K", "page": 12, "text": "Risk factors. The company faces competition "
     "and changes in foreign currency exchange rates affect reported results."},
    {"doc_name": "3M_2018_10K", "page": 57, "text": "Consolidated Statement of Income. Net sales "
     "32,765 31,657 30,109. Operating income 7,207."},
    {"doc_name": "ADOBE_2017_10K", "page": 40, "text": "Revenue by segment. Digital Media "
     "subscription revenue grew as customers moved to Creative Cloud."},
    {"doc_name": "ADOBE_2017_10K", "page": 75, "text": "Cash flows from investing activities. "
     "Purchases of property and equipment (178) (204)."},
    {"doc_name": "AMCOR_2020_10K", "page": 30, "text": "Liquidity. Capital expenditures for "
     "plant and equipment were funded by operating cash flow."},
]


@pytest.fixture(scope="session")
def tiny_index() -> SearchIndex:
    return SearchIndex.build(chunk_pages(PAGES, size=40, overlap=10), lsa_dim=4)
