"""tests/test_actor_critic.py — Unit tests for Actor and Critic nodes with mocked LLMs."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from actor_critic.actor import ActorNode
from actor_critic.critic import CriticNode
from actor_critic.state import WorkflowState
from core.config import AppConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def config() -> AppConfig:
    return AppConfig(
        llm_provider="ollama",
        confidence_high=0.85,
        confidence_medium=0.60,
    )


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    return llm


def _base_state(**overrides) -> WorkflowState:
    base: WorkflowState = {  # type: ignore[typeddict-item]
        "run_id": "test-run-1",
        "query": "What is FOB?",
        "domain": "global_trade",
        "topic_label": "incoterms",
        "retrieved_context": [],
        "context_version": "1.1",
        "formatted_context": "## Domain Context\nFOB: seller loads vessel.",
        "actor_response": "",
        "critic_score": 0.0,
        "critic_reasoning": "",
        "confidence_tier": "LOW",
        "hitl_status": "pending",
        "hitl_comment": None,
        "human_edited_response": None,
        "reviewed_by": None,
        "final_response": "",
        "context_delta_summary": None,
        "new_context_version": None,
    }
    base.update(overrides)
    return base


# ── Actor tests ────────────────────────────────────────────────────────────────

def test_actor_uses_context(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="FOB means Free On Board.")
    actor = ActorNode(llm=mock_llm)
    state = _base_state()
    result = actor(state)

    assert result["actor_response"] == "FOB means Free On Board."
    # Verify context was injected into the system message
    call_args = mock_llm.invoke.call_args[0][0]
    system_content = call_args[0].content
    assert "Domain Context" in system_content


def test_actor_no_context(mock_llm):
    mock_llm.invoke.return_value = AIMessage(content="Generic response.")
    actor = ActorNode(llm=mock_llm)
    state = _base_state(formatted_context="", context_version=None)
    result = actor(state)
    assert result["actor_response"] == "Generic response."


# ── Critic tests ───────────────────────────────────────────────────────────────

def test_critic_high_confidence(mock_llm, config):
    mock_llm.invoke.return_value = AIMessage(
        content='{"confidence": 0.92, "reasoning": "Well grounded.", "confidence_tier": "HIGH"}'
    )
    critic = CriticNode(llm=mock_llm, config=config)
    state = _base_state(actor_response="FOB means Free On Board.")
    result = critic(state)

    assert result["critic_score"] == pytest.approx(0.92)
    assert result["confidence_tier"] == "HIGH"


def test_critic_low_confidence(mock_llm, config):
    mock_llm.invoke.return_value = AIMessage(
        content='{"confidence": 0.45, "reasoning": "Uncertain.", "confidence_tier": "LOW"}'
    )
    critic = CriticNode(llm=mock_llm, config=config)
    state = _base_state(actor_response="I am not sure.")
    result = critic(state)

    assert result["confidence_tier"] == "LOW"
    assert result["critic_score"] < config.confidence_medium


def test_critic_parse_failure_defaults_low(mock_llm, config):
    mock_llm.invoke.return_value = AIMessage(content="not valid json at all")
    critic = CriticNode(llm=mock_llm, config=config)
    state = _base_state(actor_response="Something.")
    result = critic(state)
    # Should not raise; should return graceful fallback
    assert "critic_score" in result
    assert result["confidence_tier"] == "LOW"


def test_critic_respects_config_thresholds(mock_llm, config):
    # Score is 0.75 — should be MEDIUM (between 0.60 and 0.85)
    mock_llm.invoke.return_value = AIMessage(
        content='{"confidence": 0.75, "reasoning": "Partially grounded.", "confidence_tier": "HIGH"}'
    )
    critic = CriticNode(llm=mock_llm, config=config)
    state = _base_state(actor_response="Partial answer.")
    result = critic(state)
    # Config thresholds override LLM's self-reported tier
    assert result["confidence_tier"] == "MEDIUM"
