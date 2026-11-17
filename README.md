---
title: Vacancy Copilot
emoji: 🔎
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# Vacancy Copilot

Hybrid semantic search over current Ukrainian IT vacancies (DOU and Djinni), with metadata
filters and a reproducible retrieval evaluation. Queries can be Ukrainian or English.

**Why.** Keyword search on job boards misses paraphrases and mixed-language postings. This
project measures how much BM25, dense vectors, hybrid fusion and cross-encoder reranking each
help, on a hand-labelled query set, and serves the best variant as a live demo.

Retrieval: BM25 + `BAAI/bge-m3` vectors fused (RRF) in LanceDB, optional rerank of the top 30
with `BAAI/bge-reranker-v2-m3`. Near-duplicate postings are grouped; the hit links to the
best-ranked member.

## Data rules

- Vacancies are read from IT Pulse's Neon database through a read-only role. Corpus: vacancies
  seen in the last 30 days, sources `dou` and `djinni` only.
- **No vacancy text is ever public.** The demo shows title, metadata, matched skills and the
  link to the original posting. Full texts exist only in the local index and in a **private**
  Hugging Face dataset repo that the Space downloads at startup with a token secret. This
  GitHub repo and the Space repo hold code only.
- Logs (Actions, Space) carry counts only.
- DOU terms forbid automated collection without consent; consent is **pending**. Features that
  quote vacancy text stay local until it is given.
- `data/`, `cv/`, `.env` are gitignored; test fixtures are synthetic.

## Evaluation

30-50 hand-labelled queries (`eval/queries.jsonl`); metrics recall@5, recall@10, recall@20 and
MRR for `bm25`, `vector`, `hybrid`, `rerank`. Results are written to `eval/results.md` and
`eval/results.json` and shown in the demo's second tab. The numbers are tied to a snapshot: a
git tag on the private index repo (`eval-YYYY-MM-DD`) that publish never overwrites, so they can
be reproduced from the committed query set and that tag. See `eval/README.md` for labelling.

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
export COPILOT_DATABASE_URL=...        # read-only role `copilot`
uv run python -m copilot sync          # Neon -> data/ (counts only)
uv run python -m copilot index         # data/index (downloads models on first run)
                                       # refuses a corpus under 50% of data/index-prev; --allow-shrink if real
uv run python -m copilot search "python backend remote" --mode rerank
uv run python app.py                   # Gradio UI

uv run python -m copilot pool          # candidate pool -> eval/pool.csv
uv run python -m copilot label         # apply marked labels to eval/queries.jsonl
uv run python -m copilot eval [--snapshot eval-YYYY-MM-DD]
uv run pytest
```

Without `HF_INDEX_REPO` the app uses the local `data/index`.

Filters in the app come from values present in the index. Location is "any of", because IT Pulse
stores cities in the language of the posting ("Kyiv" and "Київ" are separate values). The salary
filter means "can pay at least X USD/month" (top of the range, or the only number given) and hides
vacancies without a salary. The app defaults to `hybrid`; `rerank` is slower on a free CPU.

## Deploy

Done by the owner; nothing here is run by CI setup or by the agent.

1. Create a private dataset repo for the index and a Gradio Space (free CPU).
   Before the first sync, check that role `copilot` can `SELECT` from `vacancy_current`, `raw_item` and
   `extraction_skill_resolved` (and the tables/views under it): `select 1 from extraction_skill_resolved limit 1`
   as `copilot`.
2. GitHub repo secrets: `COPILOT_DATABASE_URL`, `HF_TOKEN` (write), `HF_INDEX_REPO`
   (`user/name` of the dataset), `HF_SPACE_ID` (`user/space`, used for `restart_space`).
3. Space secrets: `HF_TOKEN` (read), `HF_INDEX_REPO`.
4. Push this repo to the Space (`git remote add space https://huggingface.co/spaces/<user>/<space>`,
   `git push space main`). The front-matter above is the Space config; `requirements.txt` is
   exported with `uv export --no-hashes --no-dev -o requirements.txt` (re-run after changing
   dependencies, then re-add the first line `--extra-index-url https://download.pytorch.org/whl/cpu`,
   which pip needs for the CPU-only torch build).
5. Publish the eval snapshot once, then record it:
   ```bash
   HF_INDEX_REPO=... uv run python -m copilot publish --tag eval-<date of the snapshot>
   uv run python -m copilot eval --snapshot eval-<date of the snapshot>
   ```
   An existing tag is refused, never moved.
   Do not re-run `sync` or `index` between `pool`/`label` and `publish --tag`: labels are dedup
   group ids of the current local index, and a rebuild can change them, so labels and tagged index diverge.
6. Run the `publish` workflow by hand (Actions, workflow_dispatch). Expected: green, logs show
   counts only, the Space reloads and answers a query. After that it runs daily at 04:00 UTC
   (sync, index, publish, restart Space). GitHub disables scheduled workflows in public repos
   after 60 days without repo activity; re-enable it in Actions if that happens.
