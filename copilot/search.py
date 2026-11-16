"""Hybrid search over the vacancies table: BM25 + vector, fused by RRF, optionally reranked."""
import math
from dataclasses import dataclass, field
from typing import Literal

from copilot.models import embed, rerank

POOL = 50  # candidates taken from each list before fusion
RERANK_TOP = 30  # reranker sees at most this many fused candidates
COLUMNS = [
    "id", "title", "link", "source", "role", "seniority", "work_format", "locations", "english",
    "domain", "salary_min", "salary_max", "salary_currency", "salary_period", "salary_usd_month",
    "skills", "text", "group_id",
]


@dataclass
class Filters:
    role: str | None = None
    seniority: str | None = None
    work_format: str | None = None
    location: str | None = None
    english: str | None = None
    domain: str | None = None
    min_salary_usd_month: float | None = None


@dataclass
class Hit:  # no description field: description text must never reach results
    id: str
    title: str
    link: str
    source: str
    score: float
    links: list[str] = field(default_factory=list)
    role: str | None = None
    seniority: str | None = None
    work_format: str | None = None
    locations: list[str] | None = None
    english: str | None = None
    domain: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    salary_usd_month: float | None = None
    skills: list[str] | None = None


def _q(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def filters_to_where(f: Filters) -> str | None:
    parts = [f"{c} = {_q(v)}" for c in ("role", "seniority", "work_format", "english", "domain")
             if (v := getattr(f, c)) is not None]
    if f.location is not None:
        parts.append(f"array_has(locations, {_q(f.location)})")
    if f.min_salary_usd_month is not None:
        if not math.isfinite(f.min_salary_usd_month):
            raise ValueError("min_salary_usd_month must be finite")
        parts.append(f"salary_usd_month >= {float(f.min_salary_usd_month)}")
    return " AND ".join(parts) or None


def _rrf_scores(rankings: list[list[str]], k: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id_ in enumerate(ranking):
            scores[id_] = scores.get(id_, 0.0) + 1 / (k + rank + 1)
    return scores


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores = _rrf_scores(rankings, k)
    return sorted(scores, key=lambda i: -scores[i])


def search(
    table, query: str, filters: Filters | None = None,
    mode: Literal["bm25", "vector", "hybrid", "rerank"] = "rerank", k: int = 20,
    embed_fn=embed, rerank_fn=rerank,
) -> list[Hit]:
    if not query.strip():
        return []
    where = filters_to_where(filters) if filters else None

    def run(q, score_col):
        if where:
            q = q.where(where, prefilter=True)
        return q.select([*COLUMNS, score_col]).limit(POOL).to_list()

    rows: dict[str, dict] = {}
    rankings: list[list[str]] = []
    scores: dict[str, float] = {}
    if mode != "vector":
        got = run(table.search(query, query_type="fts"), "_score")
        rows.update({r["id"]: r for r in got})
        rankings.append([r["id"] for r in got])
        scores = {r["id"]: r["_score"] for r in got}
    if mode != "bm25":
        got = run(table.search(embed_fn([query])[0]), "_distance")
        for r in got:
            rows.setdefault(r["id"], r)
        rankings.append([r["id"] for r in got])
        if mode == "vector":
            scores = {r["id"]: -r["_distance"] for r in got}

    if mode in ("hybrid", "rerank"):
        scores = _rrf_scores(rankings, 60)
    ids = rrf(rankings)
    if mode == "rerank" and ids:
        ids = ids[:RERANK_TOP]
        scores = dict(zip(ids, rerank_fn(query, [rows[i]["text"] for i in ids])))
        ids.sort(key=lambda i: -scores[i])
    elif mode in ("bm25", "vector"):
        ids = rankings[0]
    # Group after ranking, before cutting to k: k hits are k distinct groups.
    best: dict[str, str] = {}  # group id -> best-ranked member id
    for i in ids:
        best.setdefault(rows[i]["group_id"], i)
    top = list(best.items())[:k]
    links: dict[str, list[str]] = {}
    if top:
        where_g = "group_id IN (" + ", ".join(_q(g) for g, _ in top) + ")"
        for r in table.search().where(where_g).select(["id", "link", "group_id"]).limit(None).to_list():
            links.setdefault(r["group_id"], []).append(r)
    hits = []
    for g, i in top:
        r = rows[i]
        ls = [m["link"] for m in sorted(links.get(g, []), key=lambda m: m["id"])]
        ls = [r["link"]] + [x for x in ls if x != r["link"]]
        hits.append(Hit(**{c: r[c] for c in COLUMNS if c not in ("text", "group_id", "id")},
                        id=g, links=ls, score=float(scores[i])))
    return hits
