import pytest
import lancedb
import pyarrow as pa

from copilot.index import build_index
from copilot.search import Filters, Hit, filters_to_where, rrf, search
from tests.test_index import fake_embed, vac


class Calls:
    n = 0

    def embed(self, texts):
        self.n += 1
        return fake_embed(texts)

    def rerank(self, q, texts):
        self.n += 1
        return [float(len(t)) for t in texts]


def make(tmp_path, vs):
    build_index(vs, str(tmp_path), embed_fn=fake_embed)
    return lancedb.connect(str(tmp_path)).open_table("vacancies")


def test_rrf():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"


def test_filters_escape_quotes(tmp_path):
    f = Filters(location="Kyiv's")
    assert "'Kyiv''s'" in filters_to_where(f)
    t = make(tmp_path, [vac(1, "Python Developer", locations=["Kyiv's"])])
    c = Calls()
    for mode in ("bm25", "vector", "hybrid", "rerank"):
        hits = search(t, "Python", f, mode=mode, embed_fn=c.embed, rerank_fn=c.rerank)
        assert [h.id for h in hits] == ["dou:1"]


def test_empty_query_returns_nothing_without_models(tmp_path):
    t = make(tmp_path, [vac(1, "Python Developer")])
    c = Calls()
    assert search(t, "   ", embed_fn=c.embed, rerank_fn=c.rerank) == []
    assert c.n == 0


def test_null_metadata_found_without_filter_excluded_with_filter(tmp_path):
    t = make(tmp_path, [vac(1, "Quantum Gardener"), vac(2, "Other Job", work_format="remote")])
    c = Calls()
    kw = dict(mode="hybrid", embed_fn=c.embed, rerank_fn=c.rerank)
    assert "dou:1" in [h.id for h in search(t, "Quantum Gardener", **kw)]
    assert "dou:1" not in [h.id for h in search(t, "Quantum Gardener", Filters(work_format="remote"), **kw)]


def test_unknown_mode_is_rejected(tmp_path):
    t = make(tmp_path, [vac(1, "Dev")])
    with pytest.raises(ValueError, match="unknown mode"):
        search(t, "Dev", mode="junk")


def test_location_matches_any_of_the_given_values(tmp_path):
    t = make(tmp_path, [vac(1, "Dev A", locations=["Київ"]), vac(2, "Dev B", locations=["Lviv"]),
                        vac(3, "Dev C", locations=["Kyiv"])])
    hits = search(t, "Dev", Filters(location=["Kyiv", "Київ"]), mode="bm25")
    assert sorted(h.id for h in hits) == ["dou:1", "dou:3"]


def test_salary_filter_means_can_pay_at_least(tmp_path):
    t = make(tmp_path, [
        vac(1, "Dev A", salary_min=2000.0, salary_max=5000.0, salary_currency="USD", salary_period="month"),
        vac(2, "Dev B", salary_min=2000.0, salary_max=3000.0, salary_currency="USD", salary_period="month"),
        vac(3, "Dev C"),  # no salary: hidden while the salary filter is on
    ])
    hits = search(t, "Dev", Filters(min_salary_usd_month=4000), mode="bm25")
    assert [h.id for h in hits] == ["dou:1"]


def test_rerank_sees_skills_first(tmp_path):
    seen = []
    t = make(tmp_path, [vac(1, "Dev", skills=["kafka", "go"])])
    search(t, "Dev", mode="rerank", embed_fn=fake_embed, rerank_fn=lambda q, xs: seen.extend(xs) or [1.0] * len(xs))
    assert seen[0].startswith("kafka, go\nDev")


def test_salary_filter_across_currencies(tmp_path):
    t = make(tmp_path, [
        vac(1, "Dev A", salary_min=40.0, salary_currency="USD", salary_period="hour"),
        vac(2, "Dev B", salary_min=2000.0, salary_currency="USD", salary_period="month"),
    ])
    hits = search(t, "Dev", Filters(min_salary_usd_month=3000), mode="bm25")
    assert [h.id for h in hits] == ["dou:1"]


def test_hit_has_no_description():
    assert "description" not in Hit.__dataclass_fields__


def test_index_schema_all_null_metadata(tmp_path):
    t = make(tmp_path, [vac(1, "Only Null")])
    s = t.schema
    assert s.field("vector").type == pa.list_(pa.float32(), 1024)
    assert s.field("role").type == pa.string() and s.field("salary_min").type == pa.float64()
    assert s.field("locations").type == pa.list_(pa.string())


def test_bm25_matches_ukrainian_word_forms_and_short_tokens(tmp_path):
    t = make(tmp_path, [vac(1, "Python розробник", description="шукаємо інженера"),
                        vac(2, "Go Developer"), vac(3, "QA Engineer")])
    ids = lambda q: [h.id for h in search(t, q, mode="bm25")]
    assert ids("розробники") == ["dou:1"] and ids("інженер") == ["dou:1"]
    assert ids("developers") == ["dou:2"] and ids("Go") == ["dou:2"] and ids("QA") == ["dou:3"]
    assert ids("!!!") == []
