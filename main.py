"""FastAPI application exposing the hybrid arXiv paper search engine."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from search_engine import HybridSearchEngine

LOGGER = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    method: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=1000)


class PaperMetadata(BaseModel):
    title: str
    abstract: str
    authors: str
    categories: str
    published: str
    url: str


class SearchResponse(BaseModel):
    query: str
    method: Literal["dense", "sparse", "hybrid"]
    results: list[PaperMetadata]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared indexes once and close resources on shutdown."""
    try:
        app.state.search_engine = HybridSearchEngine(
            database_path=os.getenv("PAPERS_DB", "papers.db"),
            dense_index_path=os.getenv("FAISS_INDEX", "faiss_local.index"),
            sparse_index_path=os.getenv("BM25_INDEX", "bm25_index"),
        )
    except Exception as exc:
        LOGGER.exception("Could not initialize the search engine")
        raise RuntimeError("Search engine initialization failed") from exc
    try:
        yield
    finally:
        engine = getattr(app.state, "search_engine", None)
        if engine is not None:
            engine.close()


app = FastAPI(title="arXiv Hybrid Search API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search", response_model=SearchResponse)
def search(request_body: SearchRequest, request: Request) -> SearchResponse:
    """Run dense, sparse, or hybrid retrieval and return paper metadata."""
    engine: HybridSearchEngine = request.app.state.search_engine
    try:
        search_method = {
            "dense": engine.search_dense,
            "sparse": engine.search_sparse,
            "hybrid": engine.search_hybrid,
        }[request_body.method]
        ids = search_method(request_body.query, request_body.top_k)
        papers = engine.get_metadata(ids)
        return SearchResponse(
            query=request_body.query,
            method=request_body.method,
            results=[PaperMetadata(**paper) for paper in papers],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Search request failed")
        raise HTTPException(status_code=500, detail="Search failed") from exc
