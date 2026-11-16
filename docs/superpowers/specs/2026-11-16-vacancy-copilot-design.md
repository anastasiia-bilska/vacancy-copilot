# Vacancy Copilot: design

_2026-11-16. Status: draft for approval (builder check done)._

## Intent

Vacancy Copilot is retrieval-augmented search over the IT vacancies that
[IT Pulse UA](../../../../it-pulse-ua) already collects from DOU and Djinni. It finds vacancies
by meaning rather than keywords, and later matches a CV against them, answers questions about
the market with links to the vacancies it used, and drafts cover letters.

Why it exists:

1. **Portfolio for AI Engineer roles.** The strongest artefact is measured retrieval quality:
   recall@k and MRR on a hand-labelled query set, compared across approaches (BM25, vectors,
   hybrid, hybrid plus reranker). A recruiter must be able to try the search live and see those
   numbers next to it.
2. **The owner's own job search.** She will use it herself, so the local version may do more
   than the public one (see Data rules).

## Goals (v1)

- Semantic and hybrid search over current DOU and Djinni vacancies, in Ukrainian and English
  queries, with metadata filters (role, seniority, work format, location, salary, English level,
  domain).
- An eval harness: 30–50 hand-labelled queries, recall@k and MRR for four retrieval variants,
  one command to rerun.
- A public live demo on Hugging Face Spaces (Gradio) with a second tab showing the eval results.

## Later phases (not v1, same retrieval core)

- **v2: CV match.** Upload a CV, get top-N vacancies and the skills it lacks, using IT Pulse's
  `extraction_skill` and skill taxonomy. The uploaded CV is processed in memory and never stored.
- **v3: Market Q&A.** Questions like "most frequent requirements for Vue developers" answered by
  an LLM over retrieved vacancies, every claim cited with vacancy links.
- **v4: Cover letter draft.** Local only (see Data rules).

## Non-goals

- New sources (LinkedIn, EU job boards). DOU and Djinni are enough for now.
- Collecting or extracting data. IT Pulse owns that; this project only reads.
- A Vue/Nuxt or React frontend. Gradio keeps the whole thing in Python.
- Accounts, saved searches, notifications.

## Anti-goals (failure even if it works)

- Raw vacancy text appearing anywhere public: the Space UI, the public GitHub repo, Actions logs,
  a public HF repo.
- Any write access to IT Pulse's production database.
- The owner's CV ending up in git or on the Space.
- Eval numbers that can't be reproduced from the committed query set.

## Constraints

- **Data rules (inherited from IT Pulse ADR 0002 and its CLAUDE.md).** DOU terms (clause 2.5)
  forbid automated collection without consent; content is CC BY-NC-SA. IT Pulse publishes only
  extracted facts and links, never full texts. This project follows the same rule in public:
  - The public demo shows title, metadata, matched skills and the link to the original vacancy.
    It never shows description text or quotes from it. This is deliberately looser than IT Pulse
    ADR 0011 (whose aggregate dashboard shows no titles or links): the owner decided on
    2026-11-16 that titles and links are fine, since the RSS feeds already distribute exactly
    that and the links send traffic back to DOU and Djinni. Consent from DOU is still pending.
  - Full texts may be indexed (they are needed for BM25 and embeddings), but the index lives in a
    **private** Hugging Face dataset repo that the Space downloads at startup with a token
    secret. The Space's own repo holds code only.
  - Logs (Actions, Space) carry counts only.
  - Features that quote vacancy text (cover letter, Q&A with quotes) run locally only until DOU
    gives consent.
- **Budget $0.** HF Spaces free CPU tier (2 vCPU, 16 GB RAM), GitHub Actions free minutes
  (public repo), Neon free tier read through IT Pulse's existing `read_only` role.
- **Latency on free CPU.** Query embedding plus reranking must stay within a few seconds; the
  reranker sees at most the top ~30 hybrid results.
- Languages in the data: mostly Ukrainian and English, some Russian.

## Approaches considered

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | Separate Python repo, local LanceDB index synced from Neon, local `sentence-transformers` models | Python is native ground for embeddings, cross-encoders and eval; models are free; IT Pulse's production pipeline is untouched. Costs a second language next to IT Pulse's TypeScript. |
| B | Inside IT Pulse: pgvector in Neon, TypeScript, API embeddings | One stack and could live on the dashboard, but `bge-m3` and cross-encoders are awkward in JS, Postgres full-text is not real BM25 and has no Ukrainian stemming, and Neon free-tier storage is tight. |
| C | Python, but vectors in pgvector in Neon | One shared index, but adds tables and roles to the production database and keeps the BM25 problem. |

The owner picked A. For hosting she picked Hugging Face Spaces with Gradio plus a published eval
page, over API-model hosting on Cloudflare/Fly (fast but no longer local) and a video-only demo.

## Decided

- Repo: `~/projects/vacancy-copilot`, its own public GitHub repo, Python.
- Data: read-only from IT Pulse's `vacancy_current` view (title, link, metadata) joined with
  `raw_item.payload ->> 'description'` and `extraction_skill`. DOU and Djinni only.
- Credential: a new Neon role `copilot`, created by the owner with
  `neon roles create --name copilot --access read_only` and given **no** extra grants (the
  existing `collector`, `extractor` and `agent` roles all have INSERT/UPDATE). Blocks sync until
  created. _(Builder-check answer.)_
- Corpus: vacancies with `last_seen` within the last 30 days, the same window IT Pulse's agent
  uses (ADR 0021). Nothing marks a vacancy closed, so this is the proxy for "open".
  _(My call.)_
- Index: LanceDB (embedded, file on disk).
- Embeddings: `BAAI/bge-m3` via `sentence-transformers`.
- Lexical: LanceDB's built-in full-text (BM25) index.
- Reranker: `BAAI/bge-reranker-v2-m3` cross-encoder.
- Fusion: hybrid of BM25 and vectors, reranker on top.
- Eval: recall@k (k = 5, 10, 20) and MRR for four variants: BM25 only, vectors only, hybrid,
  hybrid plus reranker.
- UI: Gradio, two tabs: Search, and How we measured (eval table and method).
- Hosting: HF Spaces free CPU; index in a private HF dataset repo.
- LLM (from v3): Gemini API, as in IT Pulse. Ollama may be added later as an alternative.
- The owner's CV stays local and gitignored.

## Pipeline

1. **sync**: pull current vacancies from Neon into a local Parquet/LanceDB staging table.
   Incremental by `raw_item_id`.
2. **index**: chunk, embed new or changed vacancies, rebuild the BM25 index, write LanceDB.
3. **dedup**: group near-identical vacancies across DOU and Djinni so one result row stands for
   the group (both links shown).
4. **search**: filters → BM25 and vector candidates → fusion → reranker → grouped results.
5. **eval**: run every query in `eval/queries.jsonl` through each variant, write
   `eval/results.json` (query text, ranked ids, metrics; no vacancy text) and a markdown table.
6. **publish**: a scheduled GitHub Action (daily) runs sync and index, uploads the LanceDB
   directory to the private HF dataset, and restarts the Space.

## Labelling the eval set

The owner writes 30–50 queries of the kinds she'd really type (role plus domain, stack plus
format, Ukrainian and English, some with filters). Relevance is judged by **pooling**: the union
of the top 20 from all four variants is shown to her for each query, and she marks each candidate
relevant or not. Candidates outside the pool count as not relevant. The labelled set is frozen
against one dated snapshot of the index, so numbers stay reproducible as new vacancies arrive;
the snapshot id is recorded in `results.json`.

Reproducibility is for the owner, not the public _(my call)_: the eval snapshot is a pinned
revision (tag) of the private HF dataset that the daily publish job never overwrites. The public
gets the code, the query set, the labels (vacancy ids) and `results.json`. The eval tab states
the snapshot date and that the live search runs on a newer index.

## Left to the builder

- Chunking: first measure description lengths in the real data (RSS descriptions may be short).
  If most fit in bge-m3's context, embed whole vacancies; otherwise chunk by section and
  aggregate to vacancy level (max score). Report which, with numbers.
- Fusion method (reciprocal rank fusion is the default) and candidate counts.
- Dedup threshold and signals (embedding similarity plus title/company), checked by hand on a
  sample.
- Project layout, CLI shape, dependency manager, tests.
- How the labelling UI looks (a small local Gradio page or a spreadsheet export is fine).
- Whether embedding runs in GitHub Actions or locally before upload, as long as it fits free
  limits.
