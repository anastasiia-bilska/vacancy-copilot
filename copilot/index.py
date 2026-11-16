"""Build the LanceDB index: one row per vacancy (vector + native FTS on text).

Chunking: one row per vacancy; bge-m3's 8192-token limit truncates long texts.
Pending real token stats (`python -m copilot stats`); if p90 exceeds 8192, split
into a `chunks` table and aggregate by max score at search time.
"""
import shutil
from dataclasses import asdict
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from copilot.dedup import group_duplicates
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


_S, _F, _L = pa.string(), pa.float64(), pa.list_(pa.string())
# Explicit schema: all-null metadata columns must still be typed, and vector must be fixed-size.
SCHEMA = pa.schema([
    ("id", _S), ("source", _S), ("external_id", _S), ("raw_item_id", pa.int64()),
    ("title", _S), ("link", _S), ("description", _S), ("role", _S), ("seniority", _S),
    ("salary_min", _F), ("salary_max", _F), ("salary_currency", _S), ("salary_period", _S),
    ("work_format", _S), ("locations", _L), ("english", _S), ("domain", _S), ("skills", _L),
    ("last_seen", pa.timestamp("us")), ("text", _S),
    ("vector", pa.list_(pa.float32(), 1024)), ("salary_usd_month", _F), ("group_id", _S),
])


def _previous_vectors(previous_db_path: str | None) -> dict:
    """(id, raw_item_id) -> vector from an earlier index; empty if there is none."""
    if not previous_db_path or not Path(previous_db_path).is_dir():
        return {}
    db = lancedb.connect(previous_db_path)
    if "vacancies" not in db.list_tables().tables:
        return {}
    cols = db.open_table("vacancies").to_arrow().select(["id", "raw_item_id", "vector"]).to_pylist()
    return {(r["id"], r["raw_item_id"]): r["vector"] for r in cols}


def build_index(vacancies, db_path: str, embed_fn=embed, previous_db_path: str | None = None) -> None:
    """Rebuild db_path from scratch; reuse vectors from previous_db_path for unchanged vacancies."""
    texts = [f"{v.title}\n\n{v.description}" for v in vacancies]
    prev = _previous_vectors(previous_db_path)
    todo = [i for i, v in enumerate(vacancies) if (v.id, v.raw_item_id) not in prev]
    fresh = dict(zip(todo, embed_fn([texts[i] for i in todo]))) if todo else {}
    vectors = np.array([fresh[i] if i in fresh else prev[(v.id, v.raw_item_id)] for i, v in enumerate(vacancies)],
                       dtype="float32")
    print(f"reused {len(vacancies) - len(todo)}, embedded {len(todo)}")
    groups = group_duplicates([v.id for v in vacancies], np.asarray(vectors), [v.title for v in vacancies])
    rows = [
        {
            **asdict(v),
            "text": text,
            "group_id": groups[v.id],
            "vector": vec.tolist(),
            "salary_usd_month": to_usd_month(
                v.salary_min if v.salary_min is not None else v.salary_max,
                v.salary_currency,
                v.salary_period,
            ),
        }
        for v, text, vec in zip(vacancies, texts, vectors)
    ]
    # Fresh directory: overwrite would keep old LanceDB versions, which publish would upload.
    if not previous_db_path or Path(db_path).resolve() != Path(previous_db_path).resolve():
        shutil.rmtree(db_path, ignore_errors=True)
    table = lancedb.connect(db_path).create_table(
        "vacancies", pa.Table.from_pylist(rows, schema=SCHEMA), mode="overwrite"
    )
    table.create_index("text", config=FTS(), replace=True)
