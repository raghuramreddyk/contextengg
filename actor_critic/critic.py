"""actor_critic/critic.py — Critic node: scores the Actor's response."""
from __future__ import annotations

import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from actor_critic.schemas import CriticOutput
from actor_critic.state import WorkflowState
from core.config import AppConfig
from core.logger import get_logger

logger = get_logger(__name__)

_CRITIC_SYSTEM = """You are a rigorous quality evaluator for AI-generated responses.

Evaluate the response against the domain context and the query. Return ONLY a JSON object with this exact schema:
{
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one paragraph explaining the score>",
  "confidence_tier": "<HIGH|MEDIUM|LOW>"
}

Scoring rubric:
- 0.85–1.0 (HIGH): Response is accurate, well-grounded in context, complete, and clearly expressed.
- 0.60–0.84 (MEDIUM): Response is mostly correct but has minor gaps, ambiguities, or assumptions.
- 0.00–0.59 (LOW): Response has significant inaccuracies, is poorly grounded, or is incomplete.
"""


class CriticNode:
    """LangGraph node that evaluates the Actor's response with a confidence score."""

    def __init__(self, llm: BaseChatModel, config: AppConfig) -> None:
        self._llm = llm
        self._config = config

    def __call__(self, state: WorkflowState) -> dict:
        human_content = (
            f"QUERY: {state['query']}\n\n"
            f"DOMAIN CONTEXT USED:\n{state.get('formatted_context', '(none)')}\n\n"
            f"ACTOR RESPONSE:\n{state['actor_response']}"
        )

        raw = self._llm.invoke(
            [
                SystemMessage(content=_CRITIC_SYSTEM),
                HumanMessage(content=human_content),
            ]
        )

        try:
            parsed = self._parse_critic_output(raw.content)
        except Exception as exc:
            logger.warning(f"[critic] Parse failed ({exc}) — defaulting to LOW confidence")
            parsed = CriticOutput(
                confidence=0.5,
                reasoning="Critic output parsing failed; defaulting to medium-low score.",
                confidence_tier="LOW",
            )

        # Re-compute tier from thresholds in config (in case LLM tier is inconsistent)
        score = parsed.confidence
        if score >= self._config.confidence_high:
            tier = "HIGH"
        elif score >= self._config.confidence_medium:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        logger.info(f"[critic] Score={score:.2f} Tier={tier}")
        return {
            "critic_score": score,
            "critic_reasoning": parsed.reasoning,
            "confidence_tier": tier,
        }

    @staticmethod
    def _parse_critic_output(text: str) -> CriticOutput:
        """Extract JSON from LLM output (handles markdown fences)."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        data = json.loads(cleaned)
        return CriticOutput(
            confidence=float(data["confidence"]),
            reasoning=str(data["reasoning"]),
            confidence_tier=data.get("confidence_tier", "MEDIUM"),
        )
