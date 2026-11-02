# Project Map and Cleanup Notes

This map distinguishes executable project code from generated data. Commands in
the README assume the Python entrypoints stay in the project root, and Uvicorn
starts the API with `main:app`.

## Source and entrypoints

| File or folder | Role | Status |
|---|---|---|
| `run.py`, `start.bat` | Start FastAPI and Vite together on Windows | Active app launchers |
| `main.py` | FastAPI app, startup lifecycle, search/compare/label/benchmark endpoints | Active API entrypoint |
| `src/` | React interface and styles | Active frontend |
| `embeddings.py` | Local and OpenAI embedding providers | Shared by indexing and retrieval |
| `search_engine.py` | FAISS, BM25, SQLite retrieval and metadata | Active retrieval core |
| `ingest_kaggle.py` | Stream and sample the large Kaggle JSONL corpus | Primary corpus importer in README |
| `ingest.py` | Fetch a smaller corpus directly from the arXiv API | Alternative importer; still functional, not called by the app |
| `build_index.py` | Build dense and sparse indexes from the selected SQLite database | Active offline indexing command |
| `labels_store.py`, `overlap.py`, `metrics.py`, `benchmark_runner.py` | Label persistence, comparison stats, and evaluation | Shared API/CLI support |
| `evaluate.py`, `export_labels.py` | Run evaluation and export human judgments | Active command-line utilities |
| `tests/` | Unit tests for search metrics, label storage, benchmark runner, and generated README results | Active tests |
| `requirements.txt`, `package.json`, `package-lock.json`, `vite.config.js`, `tailwind.config.js`, `postcss.config.js`, `index.html` | Python/JavaScript dependencies and frontend build configuration | Required project configuration |

## Generated or user data

These are not source code and should not be moved or deleted as part of routine
source cleanup. `.gitignore` intentionally excludes most of them.

| Artifact | Finding |
|---|---|
| `papers.db` | Active Kaggle-derived corpus with 150,000 rows, dated 2018-01-01 through 2026-10-29. The other categories in the source data remain in the database; the importer selected cs.AI, cs.LG, and cs.CL papers. |
| `faiss_local.index`, `faiss_bge.index` | Active MiniLM and BGE FAISS indexes, each validated at 150,000 vectors. |
| `bm25_index/` | Active BM25 sparse index, validated with a 150,000-row SQLite ID mapping. |
| `papers_1k.db`, `faiss_local_1k.index`, `faiss_phase2_check.index`, `bm25_index_1k/` | Older generated/demo/check artifacts. They are not used by the active app pipeline; retain as local files unless deliberately cleaning up backups. |
| `emb_cache/` | Resumable embedding chunks used by `build_index.py`; useful during long indexing runs. |
| `labels.db` | Human benchmark store with 16 query records, 16 persistent pools, and 815 real judgments (542 Relevant, 273 Not relevant). All pool candidates are judged. |
| `.venv/`, `node_modules/`, `dist/`, `__pycache__/`, `.pytest_cache/` | Re-creatable environments, build products, and caches. They are not project source and are ignored by Git. |

## Cleanup guidance

- Do not delete `ingest.py`: it is a separate, smaller arXiv API ingestion
  option even though the README documents `ingest_kaggle.py` as the main path.
- Do not delete the `*_1k` database or indexes without deciding that the
  rollback copy is no longer needed.
- `faiss_phase2_check.index` is the only listed artifact with no source
  references. It is likely disposable, but it is a binary generated file and
  should be removed only after confirming it is not needed as a local backup.
- Source entrypoints remain in the root to preserve the documented commands
  (`python run.py`, `python build_index.py`, and `python evaluate.py`) and the
  `uvicorn main:app` import path. Shared runtime logic is already separated into
  focused modules; moving entrypoints into packages would require changing
  those commands and users' existing launch habits.

## Current project state

The UI/API/index pipeline and the 150,000-paper Kaggle-derived corpus are
present. Both dense indexes and the sparse index are aligned to the corpus.
Sixteen benchmark queries have complete, persistent candidate pools and 815
human relevance judgments. `labels.json`, `benchmark_results.json`, the
generated README results block, and three UI screenshots are present. Run
`python status.py` to check their current freshness and alignment.
