"""api/main.py — FastAPI application entry point."""
from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routers.workflow import make_workflow_router
from context_engine.store import ContextStore
from core.config import settings
from core.llm_factory import build_embeddings
from core.logger import get_logger
from hitl.gateway import HITLGateway
from hitl.router import make_hitl_router

logger = get_logger(__name__)

_UI_DIR = pathlib.Path(__file__).parent.parent / "ui"

# ── App-level singletons ──────────────────────────────────────────────────────
_embeddings = build_embeddings(settings)
_store = ContextStore(
    persist_dir=settings.chroma_persist_dir,
    collection_name=settings.context_collection_name,
    lc_embeddings=_embeddings,
)
_gateway = HITLGateway(mode="async")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("contextengg API starting up")
    yield
    logger.info("contextengg API shutting down")


app = FastAPI(
    title="contextengg — Agentic AI with Incremental Domain Context & HITL",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(make_workflow_router(store=_store, gateway=_gateway))
app.include_router(make_hitl_router(gateway=_gateway))


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


# ── Static UI ─────────────────────────────────────────────────────────────────
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(_UI_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
