"""Build the LanceDB index: one row per vacancy (vector + native FTS on text).

Chunking: one row per vacancy; bge-m3's 8192-token limit truncates long texts.
Pending real token stats (`python -m copilot stats`); if p90 exceeds 8192, split
into a `chunks` table and aggregate by max score at search time.
"""
from dataclasses import asdict

import lancedb
from lancedb.index import FTS

from copilot.models import embed

# ponytail: fixed FX rates, fetch daily if salary filtering needs precision
EUR_USD = 1.08
UAH_USD = 0.024
_RATE = {"USD": 1.0, "EUR": EUR_USD, "UAH": UAH_USD}
_TO_MONTH = {"month": 1.0, "hour": 168.0, "year": 1 / 12}


def to_usd_month(amount: float | None, currency: str | None, period: str | None) -> float | None:
    if amount is None or currency not in _RATE or period not in _TO_MONTH:
        return None
    return amount * _RATE[currency] * _TO_MONTH[period]


def build_index(vacancies, db_path: str, embed_fn=embed) -> None:
    texts = [f"{v.title}\n\n{v.description}" for v in vacancies]
    vectors = embed_fn(texts)
    rows = [
        {
            **asdict(v),
            "text": text,
            "vector": vec.tolist(),
            "salary_usd_month": to_usd_month(
                v.salary_min if v.salary_min is not None else v.salary_max,
                v.salary_currency,
                v.salary_period,
            ),
        }
        for v, text, vec in zip(vacancies, texts, vectors)
    ]
    table = lancedb.connect(db_path).create_table("vacancies", rows, mode="overwrite")
    table.create_index("text", config=FTS(), replace=True)
