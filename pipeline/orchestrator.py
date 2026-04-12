"""pipeline/orchestrator.py — Master orchestrator tying Actor-Critic, HITL, and Context Engine together."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import typer
from langgraph.types import Command
from rich.console import Console

from actor_critic.actor import ActorNode
from actor_critic.critic import CriticNode
from actor_critic.graph import build_actor_critic_graph
from actor_critic.schemas import WorkflowResult
from actor_critic.state import WorkflowState
from context_engine.graph import build_context_update_graph
from context_engine.merger import ContextMerger
from context_engine.retriever import ContextRetriever
from context_engine.store import ContextStore
from core.config import AppConfig, settings
from core.llm_factory import build_embeddings, build_llm
from core.logger import get_logger
from hitl.gateway import HITLGateway
from hitl.schemas import HITLDecision

logger = get_logger(__name__)
console = Console()
app = typer.Typer(name="contextengg", help="Agentic AI — Actor-Critic with Domain Context & HITL")


# ── Dependency wiring ─────────────────────────────────────────────────────────

def _build_dependencies(config: AppConfig) -> dict[str, Any]:
    llm = build_llm(config)
    embeddings = build_embeddings(config)

    store = ContextStore(
        persist_dir=config.chroma_persist_dir,
        collection_name=config.context_collection_name,
        lc_embeddings=embeddings,
    )
    retriever = ContextRetriever(store=store, lc_embeddings=embeddings, top_k=config.top_k_context)
    merger = ContextMerger(llm=llm)

    actor = ActorNode(llm=llm)
    critic = CriticNode(llm=llm, config=config)

    ac_graph = build_actor_critic_graph(actor=actor, critic=critic, retriever=retriever, config=config)
    ctx_update_graph = build_context_update_graph(store=store, merger=merger)
    gateway = HITLGateway(mode=config.hitl_mode)

    return {
        "store": store,
        "retriever": retriever,
        "ac_graph": ac_graph,
        "ctx_update_graph": ctx_update_graph,
        "gateway": gateway,
    }


# ── Core run function ─────────────────────────────────────────────────────────

def run(
    query: str,
    domain: str,
    topic_label: str,
    config: AppConfig | None = None,
    reviewer_id: str = "human",
) -> WorkflowResult:
    """Execute one complete Actor-Critic → HITL → Context update cycle.

    For sync mode: blocks at the HITL interrupt, prompts the CLI reviewer.
    For async mode: callers should use run_async() instead.
    """
    cfg = config or settings
    deps = _build_dependencies(cfg)
    ac_graph = deps["ac_graph"]
    ctx_update_graph = deps["ctx_update_graph"]
    gateway: HITLGateway = deps["gateway"]

    run_id = str(uuid4())
    thread_config = {"configurable": {"thread_id": run_id}}

    initial_state: WorkflowState = {  # type: ignore[typeddict-item]
        "run_id": run_id,
        "query": query,
        "domain": domain,
        "topic_label": topic_label,
        "retrieved_context": [],
        "context_version": None,
        "formatted_context": "",
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

    # ── Phase 1: Run graph up to HITL interrupt ───────────────────────────────
    for event in ac_graph.stream(initial_state, config=thread_config, stream_mode="values"):
        pass  # consume events; graph will suspend at hitl_interrupt

    # Snapshot the interrupted state
    snapshot = ac_graph.get_state(thread_config)
    interrupted_state: dict = snapshot.values  # type: ignore[assignment]

    # ── Phase 2: HITL Sync decision ────────────────────────────────────────────
    if cfg.hitl_mode == "sync":
        decision: HITLDecision = HITLGateway.prompt_cli(
            {
                "run_id": run_id,
                "query": query,
                "actor_response": interrupted_state.get("actor_response", ""),
                "critic_score": interrupted_state.get("critic_score", 0.0),
                "critic_reasoning": interrupted_state.get("critic_reasoning", ""),
                "confidence_tier": interrupted_state.get("confidence_tier", "LOW"),
                "context_version_used": interrupted_state.get("context_version"),
            }
        )
    else:
        raise RuntimeError("Use run_async() for async HITL mode.")

    # ── Phase 3: Resume graph with decision ───────────────────────────────────
    resume_command = Command(
        resume={
            "decision": decision.decision,
            "edited_response": decision.edited_response,
            "comment": decision.comment,
            "reviewer_id": reviewer_id,
        }
    )
    final_state_events = list(
        ac_graph.stream(resume_command, config=thread_config, stream_mode="values")
    )
    final_state: dict = final_state_events[-1] if final_state_events else interrupted_state

    # ── Phase 4: Context update (only on approve / edit) ──────────────────────
    new_context_version = None
    if decision.decision in ("approve", "edit"):
        approved_response = decision.edited_response or final_state.get("actor_response", "")
        ctx_result = ctx_update_graph.invoke(
            {
                "query": query,
                "domain": domain,
                "topic_label": topic_label,
                "approved_response": approved_response,
                "approved_by": reviewer_id,
                "confidence": interrupted_state.get("critic_score", 0.5),
                "bump": "minor",
                "existing_entry": None,
                "delta": None,
                "result": None,
            }
        )
        if ctx_result.get("result"):
            new_context_version = ctx_result["result"].new_version
            logger.info(
                f"[orchestrator] Context updated to v{new_context_version} for '{topic_label}'"
            )

    return WorkflowResult(
        run_id=run_id,
        query=query,
        domain=domain,
        topic_label=topic_label,
        actor_response=final_state.get("final_response", ""),
        critic_score=interrupted_state.get("critic_score", 0.0),
        critic_reasoning=interrupted_state.get("critic_reasoning", ""),
        confidence_tier=interrupted_state.get("confidence_tier", "LOW"),
        context_version_used=interrupted_state.get("context_version"),
    )


async def run_async(
    run_id: str,
    query: str,
    domain: str,
    topic_label: str,
    gateway: HITLGateway,
    config: AppConfig | None = None,
) -> None:
    """Async execution of the Actor-Critic → HITL → Context update cycle."""
    import asyncio

    cfg = config or settings
    deps = _build_dependencies(cfg)
    ac_graph = deps["ac_graph"]
    ctx_update_graph = deps["ctx_update_graph"]

    thread_config = {"configurable": {"thread_id": run_id}}

    initial_state: WorkflowState = {  # type: ignore[typeddict-item]
        "run_id": run_id,
        "query": query,
        "domain": domain,
        "topic_label": topic_label,
        "retrieved_context": [],
        "context_version": None,
        "formatted_context": "",
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

    # ── Phase 1: Run graph up to HITL interrupt ───────────────────────────────
    def _run_phase1():
        for _ in ac_graph.stream(initial_state, config=thread_config, stream_mode="values"):
            pass
        return ac_graph.get_state(thread_config)

    snapshot = await asyncio.to_thread(_run_phase1)
    interrupted_state: dict = snapshot.values  # type: ignore[assignment]

    # ── Phase 2: Update queue and await HITL Async decision ────────────────────
    gateway.update_queue_entry(
        run_id,
        {
            "actor_response": interrupted_state.get("actor_response", ""),
            "critic_score": interrupted_state.get("critic_score", 0.0),
            "critic_reasoning": interrupted_state.get("critic_reasoning", ""),
            "confidence_tier": interrupted_state.get("confidence_tier", "LOW"),
            "context_version_used": interrupted_state.get("context_version"),
        }
    )

    decision = await gateway.wait_for_decision(run_id)

    # ── Phase 3: Resume graph with decision ───────────────────────────────────
    resume_command = Command(
        resume={
            "decision": decision.decision,
            "edited_response": decision.edited_response,
            "comment": decision.comment,
            "reviewer_id": decision.reviewer_id,
        }
    )

    def _run_phase3():
        events = list(ac_graph.stream(resume_command, config=thread_config, stream_mode="values"))
        return events[-1] if events else interrupted_state

    final_state = await asyncio.to_thread(_run_phase3)

    # ── Phase 4: Context update (only on approve / edit) ──────────────────────
    if decision.decision in ("approve", "edit"):
        approved_response = decision.edited_response or final_state.get("actor_response", "")

        def _run_phase4():
            return ctx_update_graph.invoke(
                {
                    "query": query,
                    "domain": domain,
                    "topic_label": topic_label,
                    "approved_response": approved_response,
                    "approved_by": decision.reviewer_id,
                    "confidence": interrupted_state.get("critic_score", 0.5),
                    "bump": "minor",
                    "existing_entry": None,
                    "delta": None,
                    "result": None,
                }
            )

        ctx_result = await asyncio.to_thread(_run_phase4)
        if ctx_result.get("result"):
            logger.info(
                f"[orchestrator] Context updated to v{ctx_result['result'].new_version} for '{topic_label}'"
            )

# ── Typer CLI ─────────────────────────────────────────────────────────────────

@app.command("run")
def cli_run(
    query: str = typer.Argument(..., help="The query to process"),
    domain: str = typer.Option("general", "--domain", "-d", help="Domain name"),
    topic: str = typer.Option("general", "--topic", "-t", help="Topic label"),
    reviewer: str = typer.Option("human", "--reviewer", "-r", help="Reviewer ID"),
):
    """Run the full Actor-Critic + HITL + Context-Update pipeline."""
    result = run(query=query, domain=domain, topic_label=topic, reviewer_id=reviewer)
    console.print(f"\n[bold green]Final Response:[/bold green]\n{result.actor_response}")
    console.print(f"\n[dim]Confidence: {result.critic_score:.2f} ({result.confidence_tier})[/dim]")


@app.command("seed")
def cli_seed(
    domain: str = typer.Argument(..., help="Domain name"),
    topic: str = typer.Argument(..., help="Topic label"),
    content_file: str = typer.Option(..., "--file", "-f", help="Path to seed content file"),
):
    """Seed initial domain context from a text file."""
    import pathlib

    content = pathlib.Path(content_file).read_text(encoding="utf-8")
    deps = _build_dependencies(settings)
    store: ContextStore = deps["store"]
    entry = store.seed(domain=domain, topic_label=topic, content=content)
    console.print(f"[green]Seeded '{topic}' [{domain}] at v{entry.version}[/green]")


@app.command("versions")
def cli_versions(
    domain: str = typer.Argument(...),
    topic: str = typer.Argument(...),
):
    """List all versions stored for a topic."""
    deps = _build_dependencies(settings)
    store: ContextStore = deps["store"]
    versions = store.list_versions(domain=domain, topic_label=topic)
    if not versions:
        console.print(f"[yellow]No context found for '{topic}' [{domain}][/yellow]")
    else:
        console.print(f"[bold]Versions for '{topic}' [{domain}]:[/bold]")
        for v in versions:
            console.print(f"  v{v}")


@app.command("rollback")
def cli_rollback(
    domain: str = typer.Argument(...),
    topic: str = typer.Argument(...),
    version: str = typer.Argument(..., help="Target version e.g. 1.2"),
):
    """Pin a topic to a prior version (read-only rollback view)."""
    deps = _build_dependencies(settings)
    store: ContextStore = deps["store"]
    entry = store.rollback(domain=domain, topic_label=topic, target_version=version)
    if entry:
        console.print(f"[green]Viewing v{version} for '{topic}':[/green]\n{entry.content}")
    else:
        console.print(f"[red]Version {version} not found for '{topic}'[/red]")


if __name__ == "__main__":
    app()
