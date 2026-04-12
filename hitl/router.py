"""hitl/router.py — FastAPI router for async HITL decision submission."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from hitl.gateway import HITLGateway
from hitl.schemas import HITLDecision, HITLReviewPayload

router = APIRouter(prefix="/hitl", tags=["HITL"])


def make_hitl_router(gateway: HITLGateway) -> APIRouter:
    """Create a FastAPI router with the gateway injected."""

    @router.get("/pending", response_model=list[HITLReviewPayload])
    def list_pending_reviews():
        """Return all workflow results awaiting human review."""
        return gateway.list_pending()

    @router.post("/decision")
    def submit_decision(decision: HITLDecision):
        """Submit an approve / reject / edit decision for a pending run."""
        try:
            gateway.resolve_decision(decision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "accepted", "run_id": decision.workflow_run_id}

    return router
