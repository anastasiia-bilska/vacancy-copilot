import csv
import json

from copilot.evaluate import apply_labels, mrr, pool, recall_at_k, run_eval
from tests.test_index import fake_embed, vac
from tests.test_search import Calls, make

VARIANTS = ["bm25", "vector", "hybrid", "rerank"]


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"c", "d"}, 2) == 0.0
    assert recall_at_k(["a", "b", "c"], {"c", "d"}, 3) == 0.5
    assert mrr(["a", "b"], {"b"}) == 0.5
    assert mrr(["a"], {"z"}) == 0.0


def _table(tmp_path):
    return make(tmp_path, [vac(1, "Python Backend Developer"), vac(2, "React Frontend Developer"),
                           vac(3, "Kubernetes Engineer")])


def test_run_eval_writes_results(tmp_path):
    t, c = _table(tmp_path / "idx"), Calls()
    qs = [
        {"id": "q1", "query": "Python", "filters": None, "relevant": ["dou:1"]},
        {"id": "q2", "query": "React", "filters": {"work_format": None}, "relevant": ["dou:2"]},
        {"id": "q3", "query": "Kubernetes", "filters": None, "relevant": []},  # unlabelled: skipped
    ]
    out = tmp_path / "out"
    res = run_eval(qs, t, out_dir=str(out), embed_fn=c.embed, rerank_fn=c.rerank)
    saved = json.loads((out / "results.json").read_text())
    assert saved == res and saved["queries_used"] == 2 and saved["snapshot"].endswith("-3")
    assert list(saved["variants"]) == VARIANTS
    assert set(saved["variants"]["hybrid"]["metrics"]) == {"recall@5", "recall@10", "recall@20", "mrr"}
    assert set(saved["variants"]["bm25"]["ranked"]) == {"q1", "q2"}
    assert saved["variants"]["bm25"]["metrics"]["recall@20"] == 1.0
    keys = set()
    json.loads((out / "results.json").read_text(), object_hook=lambda d: keys.update(d) or d)
    assert not keys & {"description", "text"}
    assert all(v in (out / "results.md").read_text() for v in VARIANTS)


def test_pool_and_apply_labels(tmp_path):
    t, c = _table(tmp_path / "idx"), Calls()
    qs = tmp_path / "queries.jsonl"
    qs.write_text(json.dumps({"id": "q1", "query": "Python", "filters": None, "relevant": []}) + "\n")
    csv_path = tmp_path / "pool.csv"
    rows = pool([json.loads(qs.read_text())], t, depth=20, out=str(csv_path),
                embed_fn=c.embed, rerank_fn=c.rerank)
    assert {r["vacancy_id"] for r in rows} == {"dou:1", "dou:2", "dou:3"}
    assert set(rows[0]) == {"query_id", "vacancy_id", "title", "link", "relevant"}
    got = list(csv.DictReader(csv_path.open()))
    for r in got:
        r["relevant"] = "Yes" if r["vacancy_id"] == "dou:1" else ""
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(got[0]))
        w.writeheader()
        w.writerows(got)
    apply_labels(str(csv_path), str(qs))
    assert json.loads(qs.read_text())["relevant"] == ["dou:1"]


def test_apply_labels_keeps_labels_of_queries_not_in_csv(tmp_path):
    qs = tmp_path / "queries.jsonl"
    qs.write_text("".join(json.dumps(q) + "\n" for q in (
        {"id": "q1", "query": "a", "filters": None, "relevant": ["dou:1"]},
        {"id": "q2", "query": "b", "filters": None, "relevant": []},
    )))
    p = tmp_path / "pool.csv"
    p.write_text("query_id,vacancy_id,title,link,relevant\nq2,dou:2,t,l,+\n")
    apply_labels(str(p), str(qs))
    got = {q["id"]: q["relevant"] for q in map(json.loads, qs.read_text().splitlines())}
    assert got == {"q1": ["dou:1"], "q2": ["dou:2"]}


def test_run_eval_uses_given_snapshot(tmp_path):
    t, c = _table(tmp_path / "idx"), Calls()
    qs = [{"id": "q1", "query": "Python", "filters": None, "relevant": ["dou:1"]}]
    res = run_eval(qs, t, out_dir=str(tmp_path / "o"), embed_fn=c.embed, rerank_fn=c.rerank, snapshot="eval-2026-11-16")
    assert res["snapshot"] == "eval-2026-11-16"
    assert "`eval-2026-11-16`" in (tmp_path / "o" / "results.md").read_text()
