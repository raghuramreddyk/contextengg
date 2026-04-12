"""api/schemas.py — Request/response models for the FastAPI layer."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RunWorkflowRequest(BaseModel):
    query: str
    domain: str
    topic_label: str
    reviewer_id: str = "api_user"


class RunWorkflowResponse(BaseModel):
    run_id: str
    status: str  # "pending_hitl" | "completed" | "rejected"
    message: str


class SeedContextRequest(BaseModel):
    domain: str
    topic_label: str
    content: str
    role: str = ""
    tasks: list[str] = Field(default_factory=list)
    # e.g. tasks=["Determine applicable Incoterms",
    #             "Identify risk transfer points",
    #             "Flag non-standard clauses"]


class SeedContextResponse(BaseModel):
    domain: str
    topic_label: str
    version: str
    entry_id: str
    role: str
    tasks: list[str]
