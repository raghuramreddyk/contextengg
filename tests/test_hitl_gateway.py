"""tests/test_hitl_gateway.py — Unit tests for HITL gateway decision routing."""
from __future__ import annotations

import pytest

from hitl.gateway import HITLGateway
from hitl.schemas import HITLDecision


@pytest.fixture()
def gateway():
    return HITLGateway(mode="async")


def _make_payload(run_id: str) -> dict:
    return {
        "query": "Test query",
        "domain": "test_domain",
        "topic_label": "test_topic",
        "actor_response": "Test response",
        "critic_score": 0.87,
        "critic_reasoning": "Looks good.",
        "confidence_tier": "HIGH",
        "context_version_used": "1.0",
    }


def test_enqueue_creates_pending_entry(gateway):
    entry = gateway.enqueue("run-001", _make_payload("run-001"))
    assert entry.run_id == "run-001"
    assert entry.status == "pending"


def test_list_pending_returns_only_pending(gateway):
    gateway.enqueue("run-002", _make_payload("run-002"))
    pending = gateway.list_pending()
    assert any(p.run_id == "run-002" for p in pending)


def test_resolve_approve(gateway):
    gateway.enqueue("run-003", _make_payload("run-003"))
    decision = HITLDecision(
        workflow_run_id="run-003",
        decision="approve",
        reviewer_id="tester",
    )
    gateway.resolve_decision(decision)
    # Entry should no longer be in pending
    pending_ids = [p.run_id for p in gateway.list_pending()]
    assert "run-003" not in pending_ids


def test_resolve_unknown_run_raises(gateway):
    decision = HITLDecision(
        workflow_run_id="nonexistent",
        decision="approve",
        reviewer_id="tester",
    )
    with pytest.raises(KeyError):
        gateway.resolve_decision(decision)


def test_resolve_edit_stores_edited_response(gateway):
    gateway.enqueue("run-004", _make_payload("run-004"))
    decision = HITLDecision(
        workflow_run_id="run-004",
        decision="edit",
        edited_response="Corrected response.",
        reviewer_id="editor",
    )
    gateway.resolve_decision(decision)
    from hitl.gateway import _queue

    entry = _queue["run-004"]
    assert entry.decision.edited_response == "Corrected response."
    assert entry.status == "edit"
