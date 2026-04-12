"""actor_critic/schemas.py — Pydantic I/O models for Actor-Critic workflow."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ActorOutput(BaseModel):
    response: str


class CriticOutput(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    confidence_tier: Literal["HIGH", "MEDIUM", "LOW"]


class WorkflowResult(BaseModel):
    """Full result of one Actor-Critic run, passed to HITL and context engine."""

    run_id: str
    query: str
    domain: str
    topic_label: str
    actor_response: str
    critic_score: float
    critic_reasoning: str
    confidence_tier: Literal["HIGH", "MEDIUM", "LOW"]
    context_version_used: str | None = None
