"""actor_critic/actor.py — Actor node: generates a response grounded in domain context."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from actor_critic.state import WorkflowState
from core.logger import get_logger

logger = get_logger(__name__)

_BASE_SYSTEM = """You are a domain expert assistant. Your role is to provide accurate, well-reasoned responses grounded in the domain knowledge provided.

Guidelines:
- Base your response strictly on the domain context when available.
- Clearly state if something is outside the provided knowledge.
- Be precise and concise. Avoid speculation.
"""


class ActorNode:
    """LangGraph node that generates a response using domain context."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._parser = StrOutputParser()

    def __call__(self, state: WorkflowState) -> dict:
        system_prompt = _BASE_SYSTEM
        if state.get("formatted_context"):
            system_prompt += f"\n\n{state['formatted_context']}"
            logger.info(
                f"[actor] Injecting domain context v{state.get('context_version', 'N/A')}"
            )
        else:
            logger.warning("[actor] No domain context available — responding from base knowledge")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["query"]),
        ]
        raw = self._llm.invoke(messages)
        response = self._parser.invoke(raw)
        logger.info(f"[actor] Generated response ({len(response)} chars)")
        return {"actor_response": response}
