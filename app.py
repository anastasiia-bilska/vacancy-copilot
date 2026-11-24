"""Gradio demo: hybrid vacancy search + eval results. Index loads at startup; models lazily."""
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

from copilot.render import notice, page, render_hits
from copilot.search import MODES, Filters, search

INDEX_DIR = "data/index"
FILTER_COLUMNS = ("role", "seniority", "work_format", "english", "domain")
log = logging.getLogger("copilot.app")
CSS = (Path(__file__).parent / "copilot" / "ui.css").read_text()
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700'
         '&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap">')
NOTE = ("Живий пошук працює на новішому індексі, ніж eval-знімок нижче. / "
        "Live search runs on a newer index than the eval snapshot below.")


def ensure_index() -> None:
    repo = os.environ.get("HF_INDEX_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, repo_type="dataset", local_dir=INDEX_DIR, token=os.environ.get("HF_TOKEN"))


@lru_cache(maxsize=1)
def _table():
    import lancedb
    ensure_index()
    return lancedb.connect(INDEX_DIR).open_table("vacancies")


def choices(table) -> dict[str, list[str]]:
    """Filter values that actually occur in the index, most frequent first (locations come in
    several languages, e.g. "Kyiv" and "Київ", so free text would mostly match nothing)."""
    import pyarrow.compute as pc

    data = table.search().select([*FILTER_COLUMNS, "locations"]).limit(None).to_arrow()
    out = {}
    for col in (*FILTER_COLUMNS, "locations"):
        values = pc.list_flatten(data[col]) if col == "locations" else data[col]
        counts = pc.value_counts(pc.drop_null(values)).to_pylist()
        out[col] = [c["values"] for c in sorted(counts, key=lambda c: -c["counts"])]
    return out


def run_search(query, mode, role, seniority, work_format, locations, english, domain, min_salary):
    t0 = time.perf_counter()
    try:
        f = Filters(role or None, seniority or None, work_format or None, locations or None,
                    english or None, domain or None, min_salary if min_salary else None)
        hits = search(_table(), query, f, mode=mode)
    except ValueError as e:
        return page(notice("Некоректний запит / Invalid request", str(e), kind="err"))
    # Counts and timings only: never the query text or results.
    log.info("search mode=%s hits=%d ms=%d", mode, len(hits), (time.perf_counter() - t0) * 1000)
    return render_hits(hits, query)


def eval_markdown() -> str:
    p = Path("eval/results.md")
    body = p.read_text() if p.exists() else "Результати з'являться після першого розміченого eval-запуску. / Results will appear after the first labelled eval run."
    return f"{NOTE}\n\n{body}"


def build_ui(opts: dict[str, list[str]]) -> gr.Blocks:
    with gr.Blocks(title="Vacancy Copilot") as demo:
        with gr.Column(elem_classes=["vc"]):
            with gr.Tab("Пошук / Search"):
                q = gr.Textbox(label="Запит / Query")
                # hybrid by default: rerank is slow on a free 2-vCPU Space; it stays selectable
                mode = gr.Radio(list(MODES), value="hybrid", label="Mode")
                with gr.Row():
                    dds = [gr.Dropdown([""] + opts[c], value="", label=c) for c in FILTER_COLUMNS]
                with gr.Row():
                    loc = gr.Dropdown(opts["locations"], multiselect=True, label="location (any of)")
                    sal = gr.Number(label="can pay at least, USD/month (hides vacancies without salary)",
                                    value=None)
                out = gr.HTML()
                inputs = [q, mode, dds[0], dds[1], dds[2], loc, dds[3], dds[4], sal]
                gr.Button("Search", variant="primary").click(run_search, inputs, out)
                q.submit(run_search, inputs, out)
            with gr.Tab("Як ми міряли / How we measured"):
                gr.Markdown(eval_markdown)
    return demo


def main() -> None:
    from copilot.models import embed, rerank

    logging.basicConfig(level=logging.INFO)
    table = _table()  # download the index and open it before serving, not on the first search
    embed(["warm-up"]), rerank("warm-up", ["warm-up"])  # load both models before the first user
    build_ui(choices(table)).queue(default_concurrency_limit=1, max_size=20).launch(css=CSS, head=FONTS)


if __name__ == "__main__":
    main()
