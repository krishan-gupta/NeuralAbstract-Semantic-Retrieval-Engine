# ArXivLens

ArXivLens searches arXiv machine learning and AI papers with complementary dense and sparse retrieval. FAISS and local Sentence Transformers models find semantically related work, BM25 matches terms, and Reciprocal Rank Fusion combines rankings. A blind labeling workflow supports a human-judged retrieval benchmark.

See [PROJECT_MAP.md](PROJECT_MAP.md) for source responsibilities and generated artifact notes.

## Architecture

```mermaid
flowchart LR
    K[Kaggle arXiv JSONL snapshot] --> I[ingest_kaggle.py]
    I --> DB[(papers.db)]
    DB --> B[build_index.py]
    B --> F[FAISS MiniLM and BGE indexes]
    B --> S[BM25 index and SQLite ID map]
    F --> API[FastAPI]
    S --> API
    DB --> API
    API --> UI[React and Vite]
    UI --> L[Blind human judgments]
    L --> LD[(labels.db and persistent pools)]
    LD --> M[Recall@K and NDCG@K]
```

## Current workspace state

The local working directory contains a 150,000-paper seeded sample from the Kaggle arXiv snapshot, published from 2018 onward and selected from `cs.AI`, `cs.LG`, and `cs.CL`. The MiniLM, BGE, and BM25 indexes each cover all 150,000 rows. The raw snapshot and generated indexes are local ignored files and are not included in Git.

The 16 benchmark queries each have a persistent blind pool built from BM25, MiniLM dense and hybrid retrieval, and BGE dense and hybrid retrieval. Human labels have not yet been collected, so the measured benchmark remains pending. No relevance judgments or quality scores are generated automatically.

## Windows setup and run

Run these commands from the project directory. Python and frontend dependencies are listed in `requirements.txt` and `package-lock.json`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm ci
```

Start the API and Vite UI together:

```powershell
python run.py
```

Open <http://localhost:5174> for the UI or <http://localhost:8000/docs> for the API. Keep the terminal open while using the application. `VITE_API_URL` can override the frontend API base URL.

## Reproduce the corpus and indexes

Place the Kaggle arXiv JSON-lines file at `data\arxiv-metadata-oai-snapshot.json`. The importer streams the file and stores a seeded random sample of up to 150,000 matching papers.

```powershell
python ingest_kaggle.py --input data\arxiv-metadata-oai-snapshot.json --db papers.db --max-papers 150000 --min-year 2018 --categories cs.AI cs.LG cs.CL --seed 42 --overwrite
python build_index.py --chunk-size 2000 --local-files-only
python build_index.py --model BAAI/bge-small-en-v1.5 --tag bge --skip-bm25 --local-files-only
python status.py
```

The first build of each embedding model requires that model to be present in the local Hugging Face cache. Once cached, `--local-files-only` prevents model downloads. Embedding chunks in `emb_cache\` allow interrupted indexing runs to resume.

## Search, labeling, and evaluation

Search offers dense, sparse, and hybrid retrieval. Compare mode reports how much the result lists agree; **overlap measures agreement, not accuracy**. The Label tab uses a fixed blind pool order. `R`, `N`, `S`, and `Z` label, skip, and undo the active paper; completed judgments are saved to `labels.db`.

Build or refresh all query pools in the Label tab after adding retrieval systems. Pools only grow, and existing judgments remain attached to their query and arXiv paper ID. Export judgments from the UI or run:

```powershell
python export_labels.py
```

After at least 12 queries have a Relevant judgment, run and save the real benchmark:

```powershell
python evaluate.py --save benchmark_results.json
```

The API Benchmark tab and CLI share the same evaluation runner. Results include macro Recall@5/10, NDCG@5/10, latency, per-embedder scores, pool-coverage warnings, per-query NDCG@10 wins/ties/losses, and a seeded 95% bootstrap interval for the paired Hybrid-minus-BM25 NDCG@10 difference. Queries without any Relevant judgment are excluded from averages. Unjudged papers in the candidate pool are treated as not relevant.

## Results

<!-- RESULTS:START -->
Results pending: run the labeling workflow and python evaluate.py --save benchmark_results.json.
<!-- RESULTS:END -->

Run `python scripts\update_readme_results.py` after producing `benchmark_results.json` to regenerate this section. The generated metrics and interpretation are read from that file.

## Checks

```powershell
python status.py
python -m pytest -q
npm run build
```

`status.py` checks the corpus, index counts, labels, benchmark artifacts, README results marker, screenshots, and tests against local files and databases.

## Limitations

- The corpus is a seeded 2018+ random sample from three arXiv categories (`cs.AI`, `cs.LG`, and `cs.CL`), not a complete or curated arXiv collection.
- Relevance judgments come from one labeler and cover a pooled set of retrieved papers.
- MiniLM truncates long inputs at its supported token limit, so some abstracts are only partially represented in its dense index.
- Unlabeled papers are counted as not relevant under the pooled-judgment evaluation assumption.
- Building 150,000-paper indexes is CPU and disk intensive; runtime depends on the local machine.
