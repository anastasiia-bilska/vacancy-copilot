"""Gradio demo: hybrid vacancy search + eval results. Models and index load lazily."""
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr

from copilot.render import render_hits
from copilot.search import Filters, search

INDEX_DIR = "data/index"
# Values mirror the CHECK constraints in it-pulse-ua db/migrations/003_extraction.sql.
CHOICES = {
    "role": ["backend", "frontend", "fullstack", "mobile", "ml_ai", "data_engineering", "data_analytics",
             "devops", "qa", "security", "gamedev", "embedded", "design", "product", "project",
             "business_analysis", "marketing", "sales", "hr", "support", "management", "other"],
    "seniority": ["intern", "junior", "middle", "senior", "lead"],
    "work_format": ["remote", "office", "hybrid"],
    "english": ["none", "basic", "intermediate", "upper", "fluent"],
    "domain": ["fintech", "igaming", "gamedev", "ecommerce", "healthtech", "edtech", "adtech", "martech",
               "defense", "govtech", "logistics", "telecom", "automotive", "cybersecurity", "ai", "other"],
}
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


def run_search(query, mode, role, seniority, work_format, location, english, domain, min_salary):
    try:
        f = Filters(role or None, seniority or None, work_format or None, (location or "").strip() or None,
                    english or None, domain or None, min_salary if min_salary else None)
        return render_hits(search(_table(), query, f, mode=mode))
    except ValueError as e:
        return f"Некоректний фільтр / Invalid filter: {e}"


def eval_markdown() -> str:
    p = Path("eval/results.md")
    body = p.read_text() if p.exists() else "Результати з'являться після першого розміченого eval-запуску. / Results will appear after the first labelled eval run."
    return f"{NOTE}\n\n{body}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Vacancy Copilot") as demo:
        with gr.Tab("Пошук / Search"):
            q = gr.Textbox(label="Запит / Query")
            mode = gr.Radio(["bm25", "vector", "hybrid", "rerank"], value="rerank", label="Mode")
            with gr.Row():
                dds = [gr.Dropdown([""] + v, value="", label=k) for k, v in CHOICES.items()]
            with gr.Row():
                loc = gr.Textbox(label="location")
                sal = gr.Number(label="min salary, USD/month", value=None)
            out = gr.Markdown()
            inputs = [q, mode, dds[0], dds[1], dds[2], loc, dds[3], dds[4], sal]
            gr.Button("Search", variant="primary").click(run_search, inputs, out)
            q.submit(run_search, inputs, out)
        with gr.Tab("Як ми міряли / How we measured"):
            gr.Markdown(eval_markdown)
    return demo


def main() -> None:
    build_ui().launch()


if __name__ == "__main__":
    main()
