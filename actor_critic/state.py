"""actor_critic/state.py — LangGraph state for the Actor-Critic workflow."""
from __future__ import annotations

from typing import Literal, TypedDict

from context_engine.schemas import ContextEntry


class WorkflowState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    run_id: str
    query: str
    domain: str
    topic_label: str

    # ── Context layer (filled by context_retriever node) ──────────────────────
    retrieved_context: list[ContextEntry]
    context_version: str | None  # latest version of context used
    formatted_context: str  # ready-to-inject markdown string

    # ── Actor output ──────────────────────────────────────────────────────────
    actor_response: str

    # ── Critic output ─────────────────────────────────────────────────────────
    critic_score: float
    critic_reasoning: str
    confidence_tier: Literal["HIGH", "MEDIUM", "LOW"]

    # ── HITL layer (filled after gateway) ─────────────────────────────────────
    hitl_status: Literal["pending", "approved", "rejected", "edited"]
    hitl_comment: str | None
    human_edited_response: str | None
    reviewed_by: str | None

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: str
    context_delta_summary: str | None  # what was written back to context store
    new_context_version: str | None
