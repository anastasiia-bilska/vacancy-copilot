import hashlib
from datetime import datetime

import lancedb
import numpy as np

from copilot import index
from copilot.index import build_index, to_usd_month
from copilot.sync import Vacancy


def fake_embed(texts):
    out = []
    for t in texts:
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big"))
        v = rng.standard_normal(1024).astype("float32")
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def vac(i, title, **kw):
    base = dict(
        id=f"dou:{i}", source="dou", external_id=str(i), raw_item_id=i, title=title,
        link=f"https://x/{i}", description="generic description", role=None, seniority=None,
        salary_min=None, salary_max=None, salary_currency=None, salary_period=None,
        work_format=None, locations=[], english=None, domain=None, skills=["python"],
        last_seen=datetime(2026, 1, 1),
    )
    return Vacancy(**{**base, **kw})


def test_build_index_roundtrip(tmp_path):
    vs = [
        vac(1, "Python Backend Developer", salary_min=1200.0, salary_currency="USD", salary_period="month"),
        vac(2, "Kubernetes Platform Engineer", salary_max=10.0, salary_currency="USD", salary_period="hour"),
        vac(3, "React Frontend Developer"),
    ]
    build_index(vs, str(tmp_path), embed_fn=fake_embed)
    t = lancedb.connect(str(tmp_path)).open_table("vacancies")
    assert t.count_rows() == 3
    assert len(t.to_arrow()["vector"][0]) == 1024
    hits = t.search("Kubernetes", query_type="fts").limit(5).to_list()
    assert [h["id"] for h in hits] == ["dou:2"]
    sal = {r["id"]: r["salary_usd_month"] for r in t.to_arrow().to_pylist()}
    assert sal == {"dou:1": 1200.0, "dou:2": 1680.0, "dou:3": None}


def test_to_usd_month():
    assert to_usd_month(1200, "USD", "month") == 1200
    assert to_usd_month(10, "USD", "hour") == 1680
    assert to_usd_month(12000, "USD", "year") == 1000
    assert to_usd_month(100, "EUR", "month") == 100 * index.EUR_USD
    assert to_usd_month(1000, "UAH", "month") == 1000 * index.UAH_USD
    assert to_usd_month(None, None, None) is None
