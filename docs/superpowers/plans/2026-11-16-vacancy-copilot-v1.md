# Vacancy Copilot v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hybrid semantic search over IT Pulse's DOU and Djinni vacancies, with an eval harness (recall@k, MRR for four variants) and a public Gradio demo on Hugging Face Spaces.

**Architecture:** A Python package `copilot` syncs vacancies read-only from Neon into Parquet, builds a LanceDB index (bge-m3 vectors plus BM25 full-text), and searches with filters, reciprocal rank fusion and a bge-reranker cross-encoder. A daily GitHub Action rebuilds the index and uploads it to a private HF dataset; the Gradio Space downloads it at startup.

**Tech Stack:** Python 3.12, uv, psycopg 3, pandas/pyarrow, LanceDB, sentence-transformers (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`), Gradio, huggingface_hub, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-vacancy-copilot-design.md`

## Global Constraints

- Never commit, log or display vacancy description text. Logs carry counts only. The public UI shows title, metadata, matched skills and link only.
- Read Neon only through role `copilot` (`neon roles create --name copilot --access read_only`, no extra grants). Connection string in env `COPILOT_DATABASE_URL`.
- Corpus: vacancies with `last_seen >= now() - interval '30 days'`.
- Sources: `dou`, `djinni` only.
- Models: `BAAI/bge-m3` (embeddings), `BAAI/bge-reranker-v2-m3` (reranker). Reranker sees at most the top 30 fused candidates.
- Eval metrics: recall@5, recall@10, recall@20, MRR, for variants `bm25`, `vector`, `hybrid`, `rerank`.
- Index lives in private HF dataset repo (env `HF_INDEX_REPO`, token `HF_TOKEN`); the eval snapshot is a git tag on that repo that publish never overwrites.
- `data/`, `cv/`, `.env` are gitignored. Test fixtures are synthetic.
- Budget $0: HF Spaces free CPU, GitHub Actions free minutes.

## Review Focus

1. Filter values from the UI (e.g. a location `Kyiv's`) are interpolated into a LanceDB `where` string: they must be escaped, never break the query or inject SQL. Test in Task 3.
2. An empty or whitespace-only query returns no results and a message, without loading or calling models. Test in Task 3.
3. Description text must not leak into the rendered search results or CLI output. Tests in Tasks 1 and 6.
4. Vacancies without an extraction row (metadata all null) are indexed and found; they are excluded only when a filter on that field is set. Test in Task 3.
5. Salary filter compares across currencies (USD, EUR, UAH) and periods (hour, month, year) by normalising to USD per month. Test in Task 3.

---

### Task 0: Owner setup (manual, owner only)

- [ ] Create the Neon role: `neon roles create --name copilot --access read_only`. Grant nothing else.
- [ ] Create a private HF dataset repo (e.g. `<user>/vacancy-copilot-index`) and a public Gradio Space (`<user>/vacancy-copilot`).
- [ ] Create an HF token with write access to both.
- [ ] Put `COPILOT_DATABASE_URL`, `HF_TOKEN`, `HF_INDEX_REPO` in local `.env`. GitHub secrets and Space secrets come in Task 7.

Tasks 1–6 need only `COPILOT_DATABASE_URL` for real data; their tests run without any secret.

### Task 1: Project scaffold and sync

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `copilot/__init__.py`, `copilot/__main__.py`, `copilot/sync.py`
- Test: `tests/test_sync.py`

**Interfaces:**
- Produces: `Vacancy` dataclass in `copilot/sync.py` with fields `id: str` (`"{source}:{external_id}"`), `source`, `external_id`, `raw_item_id: int`, `title`, `link`, `description`, `role`, `seniority`, `salary_min: float | None`, `salary_max`, `salary_currency`, `salary_period`, `work_format`, `locations: list[str]`, `english`, `domain`, `skills: list[str]`, `last_seen: datetime`. Optional fields are `None` when there is no extraction.
- Produces: `row_to_vacancy(row: dict) -> Vacancy`; `fetch_vacancies(conn) -> list[Vacancy]`; `save(vacancies, path="data/vacancies.parquet") -> None`; `load(path=...) -> list[Vacancy]`.
- Produces: CLI `python -m copilot sync` and `python -m copilot stats`.

- [ ] **Step 1: Write failing tests** in `tests/test_sync.py`:
  - `test_row_to_vacancy_builds_id`: row with `source="dou", external_id="123"` gives `id == "dou:123"`.
  - `test_row_to_vacancy_without_extraction`: row with all extraction columns `None` and `skills=None` gives `skills == []`, `locations == []`, `role is None`.
  - `test_save_load_roundtrip(tmp_path)`: two synthetic vacancies survive `save`/`load` unchanged.
  - `test_sync_output_has_no_description(capsys)`: run the CLI's sync reporting function `report(vacancies)` on a vacancy whose description is `"SECRET-TEXT"`; `"SECRET-TEXT" not in capsys.readouterr().out`.
- [ ] **Step 2:** `uv run pytest tests/test_sync.py -v`. Expected: FAIL, module not found.
- [ ] **Step 3: Implement** `copilot/sync.py`. One SQL query: `vacancy_current` filtered by the 30-day `last_seen` window, joined to `raw_item` for `payload->>'description'`, with `array_agg` of canonical skill names from `extraction_skill` (check `db/migrations/003_extraction.sql` and `004_skill_taxonomy.sql` in `../it-pulse-ua` for the column names). `report()` prints counts per source only. `stats` prints description length percentiles (p50, p90, p99) in bge-m3 tokens.
- [ ] **Step 4:** `uv run pytest tests/test_sync.py -v`. Expected: PASS.
- [ ] **Step 5: Real-data check** (needs Task 0): `uv run python -m copilot sync && uv run python -m copilot stats`. Expected: counts per source, a non-empty `data/vacancies.parquet`, and token percentiles. Record the percentiles in the commit message; Task 2's chunking decision depends on them.
- [ ] **Step 6: Commit** `feat: sync vacancies from IT Pulse read-only`.

### Task 2: Index

**Files:**
- Create: `copilot/index.py`, `copilot/models.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: `Vacancy`, `load()` from Task 1.
- Produces: `copilot/models.py`: `embed(texts: list[str]) -> np.ndarray` (normalised, float32, dim 1024) and `rerank(query: str, texts: list[str]) -> list[float]`, both loading their model lazily once.
- Produces: `build_index(vacancies: list[Vacancy], db_path: str, embed_fn=embed) -> None`; creates LanceDB table `vacancies` with one row per vacancy: all `Vacancy` fields, `text` (title + description, used for FTS and reranking), `vector`, `salary_usd_month: float | None`, and a native FTS index on `text`.
- Produces: `to_usd_month(amount, currency, period) -> float | None` (rates as module constants; `# ponytail: fixed FX rates, fetch daily if salary filtering needs precision`). Hour→month uses 168 hours, year→month divides by 12.
- Produces: CLI `python -m copilot index`.

- [ ] **Step 1: Failing tests:**
  - `test_build_index_roundtrip(tmp_path)`: three synthetic vacancies, `embed_fn` a fake returning deterministic unit vectors of dim 1024; table has 3 rows, vector dim 1024, FTS query for a word only in one title returns that row.
  - `test_to_usd_month`: `to_usd_month(1200, "USD", "month") == 1200`; `to_usd_month(10, "USD", "hour") == 1680`; `to_usd_month(None, None, None) is None`; EUR and UAH use the module constants.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement.** Chunking: if Task 1's p90 fits in 8192 tokens, embed whole vacancies (one row per vacancy) and note it in the module docstring; otherwise split by blank-line sections into a `chunks` table and aggregate by max score at search time. Keep salary normalised from `salary_min` (or `salary_max` when min is null).
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5: Real-data check:** `uv run python -m copilot index`. Expected: row count equal to the synced count, the time taken printed.
- [ ] **Step 6: Commit** `feat: LanceDB index with bge-m3 vectors and BM25`.

### Task 3: Search

**Files:**
- Create: `copilot/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: the `vacancies` table and `embed`, `rerank` from Task 2.
- Produces: `Filters` dataclass: `role`, `seniority`, `work_format`, `location`, `english`, `domain` (all `str | None`), `min_salary_usd_month: float | None`.
- Produces: `filters_to_where(f: Filters) -> str | None` (escapes single quotes by doubling; `location` matches with `array_has(locations, ...)`).
- Produces: `rrf(rankings: list[list[str]], k: int = 60) -> list[str]`.
- Produces: `Hit` dataclass: `id, title, link, source, score: float`, plus metadata fields, `skills`. **No description field.**
- Produces: `search(table, query: str, filters: Filters | None = None, mode: Literal["bm25", "vector", "hybrid", "rerank"] = "rerank", k: int = 20) -> list[Hit]`. Hybrid fuses the top 50 from each list. Rerank reranks the top 30 fused.
- Produces: CLI `python -m copilot search "query" [--mode ...]` printing title, link and score.

- [ ] **Step 1: Failing tests** (fake embed and rerank, synthetic table via `build_index`):
  - `test_rrf`: `rrf([["a","b"],["b","c"]])[0] == "b"`.
  - `test_filters_escape_quotes`: `filters_to_where(Filters(location="Kyiv's"))` contains `'Kyiv''s'`, and a search with it runs without error.
  - `test_empty_query_returns_nothing_without_models`: `search(table, "   ")` returns `[]`, and fake embed/rerank were not called.
  - `test_null_metadata_found_without_filter_excluded_with_filter`: a vacancy with no extraction is found by its title without filters, and is absent with `Filters(work_format="remote")`.
  - `test_salary_filter_across_currencies`: vacancies at 40 USD/hour and 2000 USD/month; `min_salary_usd_month=3000` keeps only the hourly one.
  - `test_hit_has_no_description`: `"description" not in Hit.__dataclass_fields__`.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement** with LanceDB's `search(query_type="fts")` and `search(vector)`, each with `.where(filters_to_where(...), prefilter=True)`.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5: Real-data check:** `uv run python -m copilot search "remote frontend, checkout і платежі"`. Expected: 20 hits, top ones plausibly about frontend and payments. Note the latency of the rerank mode.
- [ ] **Step 6: Commit** `feat: hybrid search with filters, RRF and reranker`.

### Task 4: Cross-source dedup

**Files:**
- Create: `copilot/dedup.py`
- Modify: `copilot/index.py` (store `group_id`), `copilot/search.py` (one hit per group, `links: list[str]` on `Hit`)
- Test: `tests/test_dedup.py`

**Interfaces:**
- Produces: `group_duplicates(ids: list[str], vectors: np.ndarray, titles: list[str], threshold: float = 0.95) -> dict[str, str]` mapping each id to its group id (the smallest id in the group). Pairs must exceed the cosine threshold **and** have normalised titles with token Jaccard ≥ 0.5.
- `Hit` gains `links: list[str]` (all links in the group); `link` stays the best-ranked one.

- [ ] **Step 1: Failing tests:** identical vectors and titles from `dou:1` and `djinni:9` land in one group; identical vectors with unrelated titles don't; a search returning both yields one hit with two links.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement** with a numpy similarity matrix (`# ponytail: O(n²) over ~10k vacancies, use ANN if the corpus grows past ~50k`) and union-find.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5: Real-data check:** run index, print the number of groups with more than one member, and inspect 20 by hand (titles and links only). Adjust the threshold if wrong; note the value in the commit.
- [ ] **Step 6: Commit** `feat: group duplicate vacancies across DOU and Djinni`.

### Task 5: Eval harness

**Files:**
- Create: `copilot/evaluate.py`, `eval/queries.jsonl` (owner fills in), `eval/README.md` (one paragraph on method)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `search`, `Filters` from Task 3.
- `eval/queries.jsonl` line format: `{"id": "q01", "query": "...", "filters": {...} | null, "relevant": ["dou:123", ...]}`. Ids are group ids after dedup.
- Produces: `recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float`, `mrr(ranked, relevant) -> float`.
- Produces: `pool(queries, table, depth=20) -> list[dict]` writing `eval/pool.csv` (`query_id, vacancy_id, title, link, relevant`) for labelling; `apply_labels(csv_path, queries_path) -> None` writes `relevant` back into `queries.jsonl`.
- Produces: `run_eval(queries, table) -> dict` writing `eval/results.json` (`snapshot`, per-variant metrics, per-query ranked ids; no vacancy text) and `eval/results.md` (the table).
- Produces: CLI `python -m copilot pool`, `python -m copilot label`, `python -m copilot eval`.

- [ ] **Step 1: Failing tests:** `recall_at_k(["a","b","c"], {"c","d"}, 2) == 0.0`, `... k=3 == 0.5`; `mrr(["a","b"], {"b"}) == 0.5`; `mrr(["a"], {"z"}) == 0.0`; `run_eval` on a synthetic table and two queries writes a `results.json` with all four variants and no field named `description` or `text`.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement.** Snapshot id is the index build date plus row count, until Task 7 replaces it with the HF tag.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5: Owner labels (manual):** the owner writes 30–50 queries in `eval/queries.jsonl` (Ukrainian and English, some with filters), runs `pool`, marks `relevant` in `eval/pool.csv`, runs `label`. `pool.csv` holds titles and links only and is gitignored.
- [ ] **Step 6:** `uv run python -m copilot eval`. Expected: `eval/results.md` table with 4 rows by 4 metrics.
- [ ] **Step 7: Commit** `feat: eval harness with pooling, recall@k and MRR` (include `queries.jsonl`, `results.json`, `results.md`).

### Task 6: Gradio app

**Files:**
- Create: `app.py`, `copilot/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `search`, `Filters`, `Hit`; `eval/results.md`.
- Produces: `render_hits(hits: list[Hit]) -> str` (markdown: linked title, source, salary, format, location, top skills).
- Produces: `app.py` with two tabs: **Пошук / Search** (query box, filter dropdowns with values taken from the CHECK constraints in IT Pulse migration `003_extraction.sql`, mode selector defaulting to `rerank`) and **Як ми міряли / How we measured** (renders `eval/results.md`, the snapshot date, and a note that live search runs on a newer index). On startup, if `HF_INDEX_REPO` is set, `snapshot_download` the index to `data/index`; otherwise use the local index.

- [ ] **Step 1: Failing tests:** `render_hits` output contains the title and link; with a vacancy whose description is `"SECRET-TEXT"` passed through `search` on a synthetic table, `"SECRET-TEXT" not in render_hits(hits)`; empty hits render a "nothing found" line.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5: Manual check:** `uv run python app.py`, open the local URL, run three queries, switch tabs. Expected: results in under ~5 s on the Mac, no description text visible.
- [ ] **Step 6: Commit** `feat: Gradio demo with search and eval tabs`.

### Task 7: Publish

**Files:**
- Create: `copilot/publish.py`, `.github/workflows/publish.yml`, `README.md`, `requirements.txt` (for the Space, exported by `uv export`)
- Test: `tests/test_publish.py`

**Interfaces:**
- Produces: `publish(index_dir: str, repo_id: str, tag: str | None = None) -> str` uploading the folder to the private dataset with `HfApi.upload_folder`, optionally creating a tag; returns the commit sha. `python -m copilot publish [--tag eval-YYYY-MM-DD]`.
- Workflow: daily cron plus `workflow_dispatch`; steps `sync`, `index`, `publish`, then `HfApi.restart_space`. Secrets `COPILOT_DATABASE_URL`, `HF_TOKEN`, `HF_INDEX_REPO`. Cache the HF model directory between runs.

- [ ] **Step 1: Failing test:** `publish` with a mocked `HfApi` calls `upload_folder(repo_type="dataset", ...)` and `create_tag` only when `tag` is given; a tag that already exists raises instead of moving.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3: Implement**, plus the workflow and the README (what, why, eval table, data rules, how to run locally).
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5: Owner setup (manual):** add the three secrets to GitHub and `HF_TOKEN`, `HF_INDEX_REPO` to the Space. Push the code to the Space.
- [ ] **Step 6:** `uv run python -m copilot publish --tag eval-<date of Task 5 snapshot>`, then re-run `eval` so `results.json` records that tag. Trigger the workflow by hand.
  Expected: the workflow is green, the logs show counts only, and the Space loads and answers a query.
- [ ] **Step 7: Commit** `feat: daily publish to HF and Space`.
