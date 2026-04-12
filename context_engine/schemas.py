"""context_engine/schemas.py — Pydantic models for domain context entries."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_serializer


class ContextEntry(BaseModel):
    """A single versioned domain knowledge chunk persisted in the context store.

    New fields:
    - role   : The agent's role/persona for this domain (e.g. "Global Trade Expert")
    - tasks  : Ordered list of tasks the agent is expected to perform in this domain
               (e.g. ["Classify HS codes", "Determine applicable Incoterms"])
    Both are injected into the Actor's system prompt to constrain its behaviour.
    """

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    topic_hash: str          # SHA-256 of normalised "domain::topic_label"
    topic_label: str         # Human-readable topic  e.g. "incoterms"
    domain: str              # e.g. "global_trade", "ap_invoice"
    content: str             # The enriched knowledge text
    version: str             # e.g. "1.0", "1.3", "2.0"
    source: Literal["seed", "hitl_approved", "merged"] = "seed"
    confidence_at_creation: float = Field(default=1.0, ge=0.0, le=1.0)

    # ── Role & Task enrichment ────────────────────────────────────────────────
    role: str = ""           # Agent persona  e.g. "Senior Global Trade Compliance Expert"
    tasks: list[str] = Field(default_factory=list)
    # e.g. ["Determine applicable Incoterms for the shipment",
    #        "Identify which party bears risk at each stage",
    #        "Flag non-standard trade clauses for review"]

    # ── Provenance ────────────────────────────────────────────────────────────
    approved_by: str | None = None
    approved_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("approved_at", "created_at")
    def _ser_dt(self, v: datetime | None) -> str | None:
        return v.isoformat() if v else None


class ContextDelta(BaseModel):
    """Knowledge delta extracted from a HITL-approved response."""

    original_query: str
    approved_response: str
    extracted_facts: list[str]   # New facts synthesised by LLM
    summary: str                 # Merged paragraph (existing + new)
    domain: str
    topic_label: str
    role: str = ""               # Carry forward role from parent entry
    tasks: list[str] = Field(default_factory=list)  # Carry forward tasks
    confidence: float
    approved_by: str
    approved_at: datetime = Field(default_factory=datetime.utcnow)


class ContextUpdateResult(BaseModel):
    """Result returned by the context update graph after a successful write."""

    previous_version: str
    new_version: str
    domain: str
    topic_label: str
    facts_added: int
    entry_id: str
