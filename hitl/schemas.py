"""hitl/schemas.py — Pydantic models for HITL decisions and queue entries."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class HITLDecision(BaseModel):
    """The human reviewer's decision on a pending workflow result."""

    workflow_run_id: str
    decision: Literal["approve", "reject", "edit"]
    edited_response: str | None = None  # Required when decision == "edit"
    comment: str | None = None
    reviewer_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HITLQueueEntry(BaseModel):
    """A workflow result queued for async human review."""

    queue_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    query: str
    domain: str
    topic_label: str
    actor_response: str
    critic_score: float
    critic_reasoning: str
    confidence_tier: Literal["HIGH", "MEDIUM", "LOW"]
    context_version_used: str | None
    status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    decision: HITLDecision | None = None
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: datetime | None = None


class HITLReviewPayload(BaseModel):
    """Payload returned to the reviewer UI for a pending item."""

    queue_id: str
    run_id: str
    query: str
    domain: str
    topic_label: str
    actor_response: str
    critic_score: float
    critic_reasoning: str
    confidence_tier: Literal["HIGH", "MEDIUM", "LOW"]
    context_version_used: str | None
    queued_at: datetime
