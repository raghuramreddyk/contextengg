"""api/routers/workflow.py — FastAPI router for workflow execution and context management."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import (
    RunWorkflowRequest,
    RunWorkflowResponse,
    SeedContextRequest,
    SeedContextResponse,
)
from context_engine.store import ContextStore
from core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/workflow", tags=["Workflow"])


def make_workflow_router(store: ContextStore, gateway) -> APIRouter:  # type: ignore[type-arg]
    """Inject dependencies and return the configured router."""

    @router.post("/run", response_model=RunWorkflowResponse)
    async def run_workflow(req: RunWorkflowRequest, background_tasks: BackgroundTasks):
        """Trigger the Actor-Critic pipeline. In async HITL mode returns 202 immediately."""
        from uuid import uuid4

        run_id = str(uuid4())
        logger.info(f"[api] /workflow/run — run_id={run_id}, domain={req.domain}")

        # For async mode: enqueue and return immediately
        entry = gateway.enqueue(
            run_id=run_id,
            payload={
                "query": req.query,
                "domain": req.domain,
                "topic_label": req.topic_label,
                "actor_response": "(processing...)",
                "critic_score": 0.0,
                "critic_reasoning": "",
                "confidence_tier": "LOW",
                "context_version_used": None,
            },
        )

        from pipeline.orchestrator import run_async

        # Trigger background task asynchronously on the main event loop
        background_tasks.add_task(
            run_async,
            run_id=run_id,
            query=req.query,
            domain=req.domain,
            topic_label=req.topic_label,
            gateway=gateway,
        )

        return RunWorkflowResponse(
            run_id=run_id,
            status="pending_hitl",
            message="Workflow queued. Await HITL review at GET /hitl/pending.",
        )

    @router.post("/seed", response_model=SeedContextResponse)
    def seed_context(req: SeedContextRequest):
        """Seed initial domain context for a topic."""
        try:
            entry = store.seed(
                domain=req.domain,
                topic_label=req.topic_label,
                content=req.content,
                role=req.role,
                tasks=req.tasks,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return SeedContextResponse(
            domain=entry.domain,
            topic_label=entry.topic_label,
            version=entry.version,
            entry_id=entry.entry_id,
            role=entry.role,
            tasks=entry.tasks,
        )

    @router.get("/versions/{domain}/{topic_label}")
    def list_versions(domain: str, topic_label: str):
        """List all stored versions for a domain topic."""
        versions = store.list_versions(domain=domain, topic_label=topic_label)
        return {"domain": domain, "topic_label": topic_label, "versions": versions}

    @router.get("/topics/{domain}")
    def list_topics(domain: str):
        """Return all topic_labels that have at least one version in a domain."""
        # Scan all documents in the collection filtered by domain
        result = store._collection.get(
            where={"domain": domain},
            include=["metadatas"],
        )
        if not result["ids"]:
            return {"domain": domain, "topics": []}
        seen: set[str] = set()
        topics = []
        for meta in result["metadatas"]:
            tl = meta.get("topic_label", "")
            if tl and tl not in seen:
                seen.add(tl)
                topics.append(tl)
        return {"domain": domain, "topics": sorted(topics)}

    @router.get("/context/metrics/{domain}")
    def context_metrics(domain: str):
        """Return per-topic metrics: versions list, latest version, content size, role, tasks."""
        result = store._collection.get(
            where={"domain": domain},
            include=["metadatas", "documents"],
        )
        if not result["ids"]:
            return {"domain": domain, "topics": []}

        import json as _json
        from datetime import datetime as _dt

        # Group by topic_label
        topics_map: dict = {}
        for doc, meta in zip(result["documents"], result["metadatas"]):
            tl = meta.get("topic_label", "unknown")
            ver = meta.get("version", "0.0")
            if tl not in topics_map:
                tasks_raw = meta.get("tasks", "[]")
                try:
                    tasks = _json.loads(tasks_raw) if isinstance(tasks_raw, str) else tasks_raw
                except Exception:
                    tasks = []
                topics_map[tl] = {
                    "topic_label": tl,
                    "domain": domain,
                    "role": meta.get("role", ""),
                    "tasks": tasks,
                    "versions": [],
                    "latest_version": "0.0",
                    "content_chars": 0,
                    "source": meta.get("source", ""),
                    "approved_by": meta.get("approved_by") or None,
                    "created_at": meta.get("created_at", ""),
                }
            topics_map[tl]["versions"].append(ver)
            # Track latest content size
            cur_latest = topics_map[tl]["latest_version"]
            def _ver_key(v: str):
                try:
                    return tuple(int(x) for x in v.split("."))
                except Exception:
                    return (0,)
            if _ver_key(ver) >= _ver_key(cur_latest):
                topics_map[tl]["latest_version"] = ver
                topics_map[tl]["content_chars"] = len(doc)
                topics_map[tl]["source"] = meta.get("source", "")
                topics_map[tl]["approved_by"] = meta.get("approved_by") or None
                topics_map[tl]["created_at"] = meta.get("created_at", "")

        for tl in topics_map:
            topics_map[tl]["versions"] = sorted(
                topics_map[tl]["versions"],
                key=lambda v: tuple(int(x) for x in v.split("."))
            )
            topics_map[tl]["version_count"] = len(topics_map[tl]["versions"])

        return {"domain": domain, "topics": list(topics_map.values())}

    return router
