import numpy as np

from copilot.dedup import group_duplicates
from copilot.search import Filters, filters_to_where, search
from tests.test_index import fake_embed, vac
from tests.test_search import Calls, make
import pytest


def test_same_vector_and_title_grouped():
    v = fake_embed(["a", "b"])
    v = np.array([v[0], v[0], v[1]])
    g = group_duplicates(["dou:1", "djinni:9", "dou:2"], v, ["Python Developer", "python developer", "Go Dev"])
    assert g == {"dou:1": "djinni:9", "djinni:9": "djinni:9", "dou:2": "dou:2"}


def test_same_vector_unrelated_titles_not_grouped():
    v = np.array([fake_embed(["a"])[0]] * 2)
    g = group_duplicates(["dou:1", "djinni:9"], v, ["Python Developer", "Chief Gardener"])
    assert g["dou:1"] != g["djinni:9"]


def test_search_returns_one_hit_with_both_links(tmp_path):
    t = make(tmp_path, [
        vac(1, "Python Developer"),
        vac(9, "Python Developer", id="djinni:9", source="djinni", link="https://d/9"),
        vac(2, "Rust Engineer"),
    ])
    c = Calls()
    hits = search(t, "Python Developer", mode="hybrid", k=2, embed_fn=c.embed, rerank_fn=c.rerank)
    assert hits[0].id == "djinni:9" and sorted(hits[0].links) == ["https://d/9", "https://x/1"]
    assert hits[0].link in hits[0].links and len({h.id for h in hits}) == len(hits) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_salary_rejected(bad):
    with pytest.raises(ValueError):
        filters_to_where(Filters(min_salary_usd_month=bad))
