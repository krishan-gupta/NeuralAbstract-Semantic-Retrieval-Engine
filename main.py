"""FastAPI application for ArXivLens search, comparisons, and human benchmarks."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from benchmark_runner import run_labeled_benchmark
from embeddings import OpenAIEmbeddingProvider
from labels_store import (
    close_labels,
    connect_labels,
    delete_label,
    export_labels,
    list_labels,
    list_queries,
    save_label,
    upsert_query,
)
from overlap import compute_overlap
from pool_builder import build_query_pool, read_query_pool
from search_engine import HybridSearchEngine

LOGGER = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    method: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=1000)
    embedder: str = "local"


class ComparisonRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    embedder: str = "local"


class SearchResponse(BaseModel):
    query: str
    method: Literal["dense", "sparse", "hybrid"]
    results: list[dict]
    embedder: str = "local"


class BenchmarkQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    query_id: int | None = Field(default=None, ge=1)


class LabelRequest(BaseModel):
    query_id: int = Field(..., ge=1)
    arxiv_id: str = Field(..., min_length=1)
    relevant: bool


class LabelDeleteRequest(BaseModel):
    query_id: int = Field(..., ge=1)
    arxiv_id: str = Field(..., min_length=1)


class PoolRefreshRequest(BaseModel):
    query_id: int = Field(..., ge=1)


class BenchmarkRunRequest(BaseModel):
    k_values: list[int] = Field(
        default_factory=lambda: [5, 10], min_length=1, max_length=5
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared search and label resources once per server process."""
    try:
        database_path = os.getenv("PAPERS_DB", "papers.db")
        local_index_path = os.getenv("FAISS_INDEX", "faiss_local.index")
        sparse_path = os.getenv("BM25_INDEX", "bm25_index")
        index_paths = {"local": local_index_path}
        loaded_embedders = {}
        bge_path = Path(os.getenv("FAISS_BGE_INDEX", "faiss_bge.index"))
        if bge_path.is_file():
            index_paths["bge"] = bge_path
        openai_path = Path(os.getenv("FAISS_OPENAI_INDEX", "faiss_openai.index"))
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_path.is_file() and openai_key:
            try:
                stats = json.loads(Path("index_stats.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stats = {}
            openai_model = os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                stats.get("openai", {}).get("model", "text-embedding-3-small"),
            )
            index_paths["openai"] = openai_path
            loaded_embedders["openai"] = OpenAIEmbeddingProvider(
                openai_model, api_key=openai_key
            )
        app.state.search_engine = HybridSearchEngine(
            database_path=database_path,
            dense_index_path=local_index_path,
            sparse_index_path=sparse_path,
            dense_indexes=index_paths,
            embedders=loaded_embedders,
        )
        app.state.labels_connection = connect_labels(
            os.getenv("LABELS_DB", "labels.db")
        )
    except Exception as exc:
        LOGGER.exception("Could not initialize ArXivLens")
        raise RuntimeError(
            "ArXivLens initialization failed; check the database and index files"
        ) from exc
    try:
        yield
    finally:
        engine = getattr(app.state, "search_engine", None)
        if engine is not None:
            engine.close()
        labels_connection = getattr(app.state, "labels_connection", None)
        if labels_connection is not None:
            close_labels(labels_connection)


app = FastAPI(title="ArXivLens Search API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    """Show the correct local UI and API documentation links."""
    return {
        "service": "ArXivLens API",
        "docs": "http://localhost:8000/docs",
        "ui": "http://localhost:5174",
    }


@app.get("/embedders")
def get_embedders(request: Request) -> dict[str, list[str]]:
    """List dense models that have both a FAISS index and query embedder loaded."""
    return {"embedders": list(request.app.state.search_engine.dense_indexes)}


@app.post("/search", response_model=SearchResponse)
def search(request_body: SearchRequest, request: Request) -> SearchResponse:
    """Run dense, sparse, or hybrid retrieval and return ordered metadata."""
    engine: HybridSearchEngine = request.app.state.search_engine
    try:
        search_method = {
            "dense": lambda: engine.search_dense(
                request_body.query, request_body.top_k, request_body.embedder
            ),
            "sparse": lambda: engine.search_sparse(
                request_body.query, request_body.top_k
            ),
            "hybrid": lambda: engine.search_hybrid(
                request_body.query, request_body.top_k, request_body.embedder
            ),
        }[request_body.method]
        ids = search_method()
        return SearchResponse(
            query=request_body.query,
            method=request_body.method,
            results=engine.get_metadata(ids),
            embedder=request_body.embedder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Search request failed")
        raise HTTPException(status_code=500, detail="Search failed") from exc


@app.post("/compare")
def compare(request_body: ComparisonRequest, request: Request) -> dict:
    """Return Hybrid and BM25 results with a live agreement summary."""
    engine: HybridSearchEngine = request.app.state.search_engine
    try:
        hybrid_ids = engine.search_hybrid(
            request_body.query, request_body.top_k, request_body.embedder
        )
        bm25_ids = engine.search_sparse(request_body.query, request_body.top_k)
        hybrid_rows = engine.get_metadata(hybrid_ids)
        bm25_rows = engine.get_metadata(bm25_ids)
        hybrid_set = {paper["id"] for paper in hybrid_rows}
        bm25_set = {paper["id"] for paper in bm25_rows}
        for paper in hybrid_rows:
            paper["unique"] = paper["id"] not in bm25_set
        for paper in bm25_rows:
            paper["unique"] = paper["id"] not in hybrid_set
        return {
            "query": request_body.query,
            "hybrid": hybrid_rows,
            "bm25": bm25_rows,
            "overlap": compute_overlap(hybrid_ids, bm25_ids),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Comparison request failed")
        raise HTTPException(status_code=500, detail="Comparison failed") from exc


@app.get("/benchmark/queries")
def get_benchmark_queries(request: Request) -> dict:
    connection = request.app.state.labels_connection
    return {"queries": list_queries(connection)}


@app.post("/benchmark/queries")
def post_benchmark_query(body: BenchmarkQueryRequest, request: Request) -> dict:
    connection = request.app.state.labels_connection
    try:
        query_id = upsert_query(connection, body.text, body.query_id)
        return {"id": query_id, "text": " ".join(body.text.split())}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Could not save benchmark query")
        raise HTTPException(
            status_code=409, detail="Query text may already exist"
        ) from exc


@app.get("/labels/pool")
def get_label_pool(
    request: Request,
    query_id: int = Query(..., ge=1),
) -> dict:
    """Read the persisted, blind pool without recomputing its contents."""
    engine: HybridSearchEngine = request.app.state.search_engine
    connection = request.app.state.labels_connection
    try:
        return read_query_pool(engine, connection, query_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Benchmark query not found"
        ) from exc


def _refresh_one_pool(request: Request, query_id: int) -> dict:
    """Union the top 20 from BM25 and all loaded dense/hybrid systems."""
    engine: HybridSearchEngine = request.app.state.search_engine
    connection = request.app.state.labels_connection
    try:
        _, added_count = build_query_pool(engine, connection, query_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Benchmark query not found"
        ) from exc
    except Exception as exc:
        LOGGER.exception("Could not refresh pool for query %s", query_id)
        raise HTTPException(status_code=500, detail="Pool refresh failed") from exc

    result = get_label_pool(request, query_id)
    result["added_count"] = added_count
    return result


@app.post("/labels/pool/refresh")
def refresh_label_pool(body: PoolRefreshRequest, request: Request) -> dict:
    """Add newly retrieved candidates to one query's persistent pool."""
    return _refresh_one_pool(request, body.query_id)


@app.post("/labels/pool/refresh-all")
def refresh_all_label_pools(request: Request) -> dict:
    """Build or extend stored pools for all seeded benchmark queries."""
    results = [
        _refresh_one_pool(request, int(query["id"]))
        for query in list_queries(request.app.state.labels_connection)
    ]
    return {
        "queries_processed": len(results),
        "papers_added": sum(int(result["added_count"]) for result in results),
        "pools": [
            {
                "query_id": result["query_id"],
                "pool_count": result["pool_count"],
                "added_count": result["added_count"],
            }
            for result in results
        ],
    }


@app.get("/labels")
def get_labels(request: Request, query_id: int = Query(..., ge=1)) -> dict:
    return {
        "query_id": query_id,
        "labels": list_labels(request.app.state.labels_connection, query_id),
    }


@app.post("/labels")
def post_label(body: LabelRequest, request: Request) -> dict[str, str | int | bool]:
    engine: HybridSearchEngine = request.app.state.search_engine
    if not engine.has_arxiv_id(body.arxiv_id):
        raise HTTPException(
            status_code=404, detail="Paper arXiv ID not found in the corpus"
        )
    try:
        save_label(
            request.app.state.labels_connection,
            body.query_id,
            body.arxiv_id,
            body.relevant,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "query_id": body.query_id,
        "arxiv_id": body.arxiv_id,
        "relevant": body.relevant,
    }


@app.delete("/labels")
def remove_label(
    body: LabelDeleteRequest, request: Request
) -> dict[str, str | int | bool]:
    try:
        removed = delete_label(
            request.app.state.labels_connection, body.query_id, body.arxiv_id
        )
    except Exception as exc:
        LOGGER.exception("Could not undo label")
        raise HTTPException(status_code=500, detail="Could not undo label") from exc
    return {"query_id": body.query_id, "arxiv_id": body.arxiv_id, "removed": removed}


@app.post("/labels/export")
def post_export_labels(request: Request) -> dict[str, str | int]:
    output_path = Path(os.getenv("LABELS_EXPORT", "labels.json"))
    try:
        exported = export_labels(request.app.state.labels_connection, output_path)
        modified = datetime.fromtimestamp(output_path.stat().st_mtime, UTC).isoformat()
    except Exception as exc:
        LOGGER.exception("Could not export human labels")
        raise HTTPException(status_code=500, detail="Could not export labels") from exc
    return {
        "count": exported,
        "path": str(output_path),
        "exported_at": modified,
    }


@app.post("/benchmark/run")
def run_benchmark(request: Request, body: BenchmarkRunRequest | None = None) -> dict:
    k_values = body.k_values if body else [5, 10]
    if any(k < 1 or k > 1000 for k in k_values):
        raise HTTPException(
            status_code=422, detail="k_values must be between 1 and 1000"
        )
    try:
        return run_labeled_benchmark(
            request.app.state.search_engine,
            request.app.state.labels_connection,
            k_values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Human-labeled benchmark failed")
        raise HTTPException(status_code=500, detail="Benchmark failed") from exc
