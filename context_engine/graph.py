"""context_engine/graph.py — LangGraph pipeline for incremental context updates."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from context_engine.merger import ContextMerger
from context_engine.schemas import ContextDelta, ContextEntry, ContextUpdateResult
from context_engine.store import ContextStore
from core.logger import get_logger

logger = get_logger(__name__)


# ── Graph State ───────────────────────────────────────────────────────────────

class ContextUpdateState(TypedDict):
    # Inputs
    query: str
    domain: str
    topic_label: str
    approved_response: str
    approved_by: str
    confidence: float
    bump: str  # "minor" | "major"

    # Intermediate
    existing_entry: ContextEntry | None
    delta: ContextDelta | None

    # Output
    result: ContextUpdateResult | None


# ── Node Functions ────────────────────────────────────────────────────────────

def make_nodes(store: ContextStore, merger: ContextMerger):
    """Closure to inject dependencies into node functions."""

    def load_existing(state: ContextUpdateState) -> dict:
        entry = store.retrieve_latest(state["domain"], state["topic_label"])
        logger.info(
            f"[context_update] Loaded existing context: "
            f"v{entry.version if entry else 'none'}"
        )
        return {"existing_entry": entry}

    def extract_delta(state: ContextUpdateState) -> dict:
        delta = merger.extract_delta(
            existing_entry=state["existing_entry"],
            approved_response=state["approved_response"],
            query=state["query"],
            domain=state["domain"],
            topic_label=state["topic_label"],
            approved_by=state["approved_by"],
            confidence=state["confidence"],
        )
        return {"delta": delta}

    def merge_and_store(state: ContextUpdateState) -> dict:
        delta = state["delta"]
        assert delta is not None

        if not delta.extracted_facts:
            logger.info("[context_update] No new facts — skipping write.")
            existing = state["existing_entry"]
            return {
                "result": ContextUpdateResult(
                    previous_version=existing.version if existing else "N/A",
                    new_version=existing.version if existing else "N/A",
                    domain=state["domain"],
                    topic_label=state["topic_label"],
                    facts_added=0,
                    entry_id=existing.entry_id if existing else "",
                )
            }

        merged_content = merger.build_merged_content(state["existing_entry"], delta)
        prev_version = state["existing_entry"].version if state["existing_entry"] else "1.0"

        new_entry = store.save_update(
            domain=state["domain"],
            topic_label=state["topic_label"],
            new_content=merged_content,
            approved_by=state["approved_by"],
            confidence=state["confidence"],
            bump=state["bump"],
        )

        result = ContextUpdateResult(
            previous_version=prev_version,
            new_version=new_entry.version,
            domain=state["domain"],
            topic_label=state["topic_label"],
            facts_added=len(delta.extracted_facts),
            entry_id=new_entry.entry_id,
        )
        logger.info(
            f"[context_update] v{prev_version} → v{new_entry.version} "
            f"({len(delta.extracted_facts)} facts added)"
        )
        return {"result": result}

    return load_existing, extract_delta, merge_and_store


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_context_update_graph(store: ContextStore, merger: ContextMerger) -> StateGraph:
    """Build and compile the context update LangGraph."""
    load_existing, extract_delta, merge_and_store = make_nodes(store, merger)

    builder = StateGraph(ContextUpdateState)
    builder.add_node("load_existing", load_existing)
    builder.add_node("extract_delta", extract_delta)
    builder.add_node("merge_and_store", merge_and_store)

    builder.add_edge(START, "load_existing")
    builder.add_edge("load_existing", "extract_delta")
    builder.add_edge("extract_delta", "merge_and_store")
    builder.add_edge("merge_and_store", END)

    return builder.compile()
