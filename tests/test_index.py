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
        vac(2, "Kubernetes Platform Engineer", salary_min=5.0, salary_max=10.0, salary_currency="USD",
            salary_period="hour"),
        vac(3, "React Frontend Developer"),
    ]
    build_index(vs, str(tmp_path), embed_fn=fake_embed)
    t = lancedb.connect(str(tmp_path)).open_table("vacancies")
    assert t.count_rows() == 3
    assert len(t.to_arrow()["vector"][0]) == 1024
    hits = t.search(index.stem("Kubernetes"), query_type="fts").limit(5).to_list()
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


def test_incremental_build_reuses_unchanged_vectors(tmp_path, capsys):
    prev, new = str(tmp_path / "prev"), str(tmp_path / "new")
    build_index([vac(1, "Python Dev"), vac(2, "Go Dev")], prev, embed_fn=fake_embed)
    seen = []

    def counting(texts):
        seen.extend(texts)
        return fake_embed([t + " v2" for t in texts])  # distinct vectors expose any re-embedding

    vs = [vac(1, "Python Dev"), vac(2, "Go Dev", raw_item_id=20), vac(3, "Rust Dev")]
    build_index(vs, new, embed_fn=counting, previous_db_path=prev)
    assert len(seen) == 2
    old = {r["id"]: r["vector"] for r in lancedb.connect(prev).open_table("vacancies").to_arrow().to_pylist()}
    cur = {r["id"]: r["vector"] for r in lancedb.connect(new).open_table("vacancies").to_arrow().to_pylist()}
    assert cur["dou:1"] == old["dou:1"] and cur["dou:2"] != old["dou:2"]
    assert "reused 1, embedded 2" in capsys.readouterr().out


def test_missing_previous_index_embeds_all(tmp_path):
    seen = []
    build_index([vac(1, "A")], str(tmp_path / "i"), embed_fn=lambda t: seen.extend(t) or fake_embed(t),
                previous_db_path=str(tmp_path / "nope"))
    assert len(seen) == 1


def test_rebuild_does_not_accumulate_versions(tmp_path):
    p = str(tmp_path)
    build_index([vac(1, "A")], p, embed_fn=fake_embed)
    n = len(lancedb.connect(p).open_table("vacancies").list_versions())
    build_index([vac(1, "A")], p, embed_fn=fake_embed)
    assert len(lancedb.connect(p).open_table("vacancies").list_versions()) == n


def test_changed_fingerprint_reembeds_everything(tmp_path, monkeypatch):
    prev, new = str(tmp_path / "prev"), str(tmp_path / "new")
    build_index([vac(1, "Python Dev")], prev, embed_fn=fake_embed)
    monkeypatch.setattr(index, "EMBED_FINGERPRINT", "other-model|max_seq=1024|tpl=1")
    seen = []
    build_index([vac(1, "Python Dev")], new, embed_fn=lambda t: seen.extend(t) or fake_embed(t),
                previous_db_path=prev)
    assert len(seen) == 1


def test_description_stored_once(tmp_path):
    build_index([vac(1, "A")], str(tmp_path), embed_fn=fake_embed)
    assert "description" not in lancedb.connect(str(tmp_path)).open_table("vacancies").schema.names
