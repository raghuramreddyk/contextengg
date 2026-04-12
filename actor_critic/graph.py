"""actor_critic/graph.py — Full Actor-Critic LangGraph with context retrieval and HITL interrupt."""
from __future__ import annotations

from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from actor_critic.actor import ActorNode
from actor_critic.critic import CriticNode
from actor_critic.state import WorkflowState
from context_engine.retriever import ContextRetriever
from core.config import AppConfig
from core.logger import get_logger

logger = get_logger(__name__)


def build_actor_critic_graph(
    actor: ActorNode,
    critic: CriticNode,
    retriever: ContextRetriever,
    config: AppConfig,
) -> StateGraph:
    """Build and compile the Actor-Critic LangGraph.

    Graph topology:
        START
          → run_id_init
          → context_retriever
          → actor
          → critic
          → hitl_interrupt   (interrupt point — graph pauses here for human review)
          → decision_router  (conditional edge)
             ├── approve → finalize
             ├── reject  → discard
             └── edit    → re_actor → critic → hitl_interrupt (loop)
          → END
    """
    checkpointer = MemorySaver()

    # ── Node functions ────────────────────────────────────────────────────────

    def run_id_init(state: WorkflowState) -> dict:
        run_id = state.get("run_id") or str(uuid4())
        logger.info(f"[workflow] Starting run {run_id}")
        return {"run_id": run_id, "hitl_status": "pending"}

    def context_retriever_node(state: WorkflowState) -> dict:
        entries, formatted = retriever.retrieve(
            query=state["query"], domain=state["domain"]
        )
        latest_version = entries[0].version if entries else None
        return {
            "retrieved_context": entries,
            "context_version": latest_version,
            "formatted_context": formatted,
        }

    def actor_node(state: WorkflowState) -> dict:
        return actor(state)

    def critic_node(state: WorkflowState) -> dict:
        return critic(state)

    def hitl_interrupt_node(state: WorkflowState) -> dict:
        """This is the interrupt point. In sync mode the graph pauses here."""
        from langgraph.types import interrupt

        logger.info(
            f"[hitl] Pausing for human review — run {state['run_id']} "
            f"(confidence={state.get('critic_score', 0):.2f}, "
            f"tier={state.get('confidence_tier')})"
        )
        # interrupt() suspends execution; the pipeline orchestrator resumes
        # by calling graph.invoke() again with a Command(resume=decision)
        decision = interrupt(
            {
                "run_id": state["run_id"],
                "query": state["query"],
                "actor_response": state["actor_response"],
                "critic_score": state["critic_score"],
                "critic_reasoning": state["critic_reasoning"],
                "confidence_tier": state["confidence_tier"],
                "context_version_used": state.get("context_version"),
            }
        )
        # decision is the HITLDecision dict injected on resume
        return {
            "hitl_status": decision.get("decision", "rejected"),
            "hitl_comment": decision.get("comment"),
            "human_edited_response": decision.get("edited_response"),
            "reviewed_by": decision.get("reviewer_id"),
        }

    def finalize_node(state: WorkflowState) -> dict:
        final = state.get("human_edited_response") or state["actor_response"]
        logger.info(f"[workflow] Finalised run {state['run_id']}")
        return {"final_response": final}

    def discard_node(state: WorkflowState) -> dict:
        logger.info(f"[workflow] Run {state['run_id']} rejected by HITL")
        return {"final_response": ""}

    def re_actor_node(state: WorkflowState) -> dict:
        """Re-run Actor using the human-edited response as additional guidance."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import StrOutputParser

        edit_hint = state.get("human_edited_response", "")
        augmented_query = (
            f"{state['query']}\n\n[Editor note]: {edit_hint}"
            if edit_hint
            else state["query"]
        )
        modified_state: WorkflowState = {**state, "query": augmented_query}  # type: ignore[typeddict-item]
        return actor(modified_state)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route_decision(state: WorkflowState) -> str:
        status = state.get("hitl_status", "rejected")
        if status == "approved":
            return "finalize"
        if status == "edited":
            return "re_actor"
        return "discard"

    # ── Graph assembly ────────────────────────────────────────────────────────

    builder = StateGraph(WorkflowState)
    builder.add_node("run_id_init", run_id_init)
    builder.add_node("context_retriever", context_retriever_node)
    builder.add_node("actor", actor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("hitl_interrupt", hitl_interrupt_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("discard", discard_node)
    builder.add_node("re_actor", re_actor_node)

    builder.add_edge(START, "run_id_init")
    builder.add_edge("run_id_init", "context_retriever")
    builder.add_edge("context_retriever", "actor")
    builder.add_edge("actor", "critic")
    builder.add_edge("critic", "hitl_interrupt")

    builder.add_conditional_edges(
        "hitl_interrupt",
        route_decision,
        {"finalize": "finalize", "discard": "discard", "re_actor": "re_actor"},
    )
    builder.add_edge("re_actor", "critic")
    builder.add_edge("finalize", END)
    builder.add_edge("discard", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_interrupt"],
    )
