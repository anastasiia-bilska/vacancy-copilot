"""Eval harness: pool candidates for labelling, then score the four search variants."""
import csv
import json
import os
from datetime import date

from copilot.models import embed, rerank
from copilot.search import Filters, search

VARIANTS = ["bm25", "vector", "hybrid", "rerank"]
KS = (5, 10, 20)
TRUE = {"1", "y", "yes", "true", "+"}


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    return next((1 / (i + 1) for i, r in enumerate(ranked) if r in relevant), 0.0)


def load_queries(path: str = "eval/queries.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _search(table, q, mode, k, embed_fn, rerank_fn):
    f = Filters(**q["filters"]) if q.get("filters") else None
    return search(table, q["query"], f, mode=mode, k=k, embed_fn=embed_fn, rerank_fn=rerank_fn)


def pool(queries, table, depth=20, out="eval/pool.csv", embed_fn=embed, rerank_fn=rerank) -> list[dict]:
    """Union of the top `depth` hits of every variant per query, for the owner to label."""
    rows = []
    for q in queries:
        seen: dict[str, object] = {}
        for mode in VARIANTS:
            for h in _search(table, q, mode, depth, embed_fn, rerank_fn):
                seen.setdefault(h.id, h)
        rows += [{"query_id": q["id"], "vacancy_id": h.id, "title": h.title, "link": h.link, "relevant": ""}
                 for h in seen.values()]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "vacancy_id", "title", "link", "relevant"])
        w.writeheader()
        w.writerows(rows)
    return rows


def apply_labels(csv_path: str, queries_path: str) -> None:
    with open(csv_path, newline="") as f:
        marked: dict[str, list[str]] = {}  # only queries present in the CSV are touched
        for r in csv.DictReader(f):
            marked.setdefault(r["query_id"], [])
            if r["relevant"].strip().lower() in TRUE:
                marked[r["query_id"]].append(r["vacancy_id"])
    queries = load_queries(queries_path)
    for q in queries:
        if q["id"] in marked:
            q["relevant"] = marked[q["id"]]
    with open(queries_path, "w") as f:
        f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in queries)


def run_eval(queries, table, out_dir="eval", embed_fn=embed, rerank_fn=rerank) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    used = [q for q in queries if q.get("relevant")]
    # ponytail: snapshot = today + row count; Task 7 replaces it with the HF tag
    res = {"snapshot": f"{date.today()}-{table.count_rows()}", "queries_used": len(used), "variants": {}}
    for mode in VARIANTS:
        ranked = {q["id"]: [h.id for h in _search(table, q, mode, max(KS), embed_fn, rerank_fn)] for q in used}
        rel = {q["id"]: set(q["relevant"]) for q in used}
        n = len(used) or 1
        metrics = {f"recall@{k}": sum(recall_at_k(ranked[i], rel[i], k) for i in ranked) / n for k in KS}
        metrics["mrr"] = sum(mrr(ranked[i], rel[i]) for i in ranked) / n
        res["variants"][mode] = {"metrics": metrics, "ranked": ranked}
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    head = ["recall@5", "recall@10", "recall@20", "mrr"]
    labels = head[:3] + ["MRR@20"]
    md = [f"Snapshot `{res['snapshot']}`, {len(used)} labelled queries.", "",
          "| variant | " + " | ".join(labels) + " |", "|---|" + "---|" * len(head)]
    md += [f"| {m} | " + " | ".join(f"{res['variants'][m]['metrics'][h]:.3f}" for h in head) + " |"
           for m in VARIANTS]
    with open(f"{out_dir}/results.md", "w") as f:
        f.write("\n".join(md) + "\n")
    return res
