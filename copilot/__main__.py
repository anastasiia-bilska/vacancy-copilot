import os
import sys

from copilot.sync import fetch_vacancies, load, report, save


def sync() -> None:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(os.environ["COPILOT_DATABASE_URL"], row_factory=dict_row,
                         options="-c default_transaction_read_only=on") as conn:
        vacancies = fetch_vacancies(conn)
    save(vacancies)
    report(vacancies)


def stats() -> None:
    from transformers import AutoTokenizer  # lazy: heavy, only this command needs it

    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    lengths = sorted(len(tok(v.description, add_special_tokens=False)["input_ids"]) for v in load())
    if not lengths:
        sys.exit("no vacancies; run sync first")
    for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99)):
        print(f"{name}: {lengths[min(len(lengths) - 1, int(q * len(lengths)))]} tokens")


def index() -> None:
    import time

    from copilot.index import build_index

    t0 = time.time()
    vacancies = load()
    if not vacancies:
        sys.exit("no vacancies to index; run sync first")
    prev = "data/index-prev" if os.path.isdir("data/index-prev") else None
    build_index(vacancies, "data/index", previous_db_path=prev)
    print(f"indexed {len(vacancies)} vacancies in {time.time() - t0:.1f}s")


def search_cmd() -> None:
    import argparse

    import lancedb

    from copilot.search import search

    ap = argparse.ArgumentParser(prog="python -m copilot search")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["bm25", "vector", "hybrid", "rerank"], default="rerank")
    args = ap.parse_args(sys.argv[2:])
    if not args.query.strip():
        sys.exit("empty query")
    table = lancedb.connect("data/index").open_table("vacancies")
    for h in search(table, args.query, mode=args.mode):
        print(f"{h.score:8.3f}  {h.title}\n          {h.link}")


def _table(path: str = "data/index"):
    import lancedb

    return lancedb.connect(path).open_table("vacancies")


def download_snapshot(tag: str) -> str:
    """Download the tagged index from HF_INDEX_REPO; returns the local dir."""
    from huggingface_hub import snapshot_download

    repo = os.environ.get("HF_INDEX_REPO")
    if not repo:
        sys.exit("HF_INDEX_REPO is not set; --snapshot needs the index repo to download the tag")
    local_dir = f"data/eval-{tag}"
    snapshot_download(repo, repo_type="dataset", revision=tag, local_dir=local_dir)
    return local_dir


def pool_cmd() -> None:
    from copilot.evaluate import load_queries, pool

    rows = pool(load_queries(), _table())
    print(f"wrote {len(rows)} rows to eval/pool.csv")


def label_cmd() -> None:
    from copilot.evaluate import apply_labels

    apply_labels("eval/pool.csv", "eval/queries.jsonl")


def publish_cmd() -> None:
    import argparse

    from copilot.publish import publish

    ap = argparse.ArgumentParser(prog="python -m copilot publish")
    ap.add_argument("--tag", help="snapshot tag, e.g. eval-2026-11-16 (never overwritten)")
    args = ap.parse_args(sys.argv[2:])
    repo = os.environ.get("HF_INDEX_REPO")
    if not repo:
        sys.exit("HF_INDEX_REPO is not set")
    sha = publish("data/index", repo, tag=args.tag)
    print(f"published commit {sha[:8]}" + (f" tagged {args.tag}" if args.tag else ""))


def eval_cmd() -> None:
    import argparse

    from copilot.evaluate import load_queries, run_eval

    ap = argparse.ArgumentParser(prog="python -m copilot eval")
    ap.add_argument("--snapshot", help="HF index tag to download and score (default: local data/index, date-rowcount)")
    tag = ap.parse_args(sys.argv[2:]).snapshot
    table = _table(download_snapshot(tag)) if tag else _table()
    res = run_eval(load_queries(), table, snapshot=tag)
    print(f"evaluated {res['queries_used']} labelled queries; see eval/results.md")


COMMANDS = {"sync": sync, "stats": stats, "index": index, "search": search_cmd,
            "pool": pool_cmd, "label": label_cmd, "eval": eval_cmd, "publish": publish_cmd}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS or (len(sys.argv) != 2 and sys.argv[1] not in ("search", "eval", "publish")):
        sys.exit("usage: python -m copilot sync|stats|index|search|pool|label|eval|publish")
    COMMANDS[sys.argv[1]]()
