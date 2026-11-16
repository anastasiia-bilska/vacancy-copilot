import os
import sys

from copilot.sync import fetch_vacancies, load, report, save


def sync() -> None:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(os.environ["COPILOT_DATABASE_URL"], row_factory=dict_row) as conn:
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
    build_index(vacancies, "data/index")
    print(f"indexed {len(vacancies)} vacancies in {time.time() - t0:.1f}s")


COMMANDS = {"sync": sync, "stats": stats, "index": index}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit("usage: python -m copilot sync|stats|index")
    COMMANDS[sys.argv[1]]()
