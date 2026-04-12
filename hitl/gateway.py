"""hitl/gateway.py — HITL gateway: sync (CLI interrupt) and async queue modes."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from hitl.schemas import HITLDecision, HITLQueueEntry, HITLReviewPayload
from core.logger import get_logger

logger = get_logger(__name__)


# ── In-memory async queue (replace with DB in production) ────────────────────

_queue: dict[str, HITLQueueEntry] = {}
_decision_futures: dict[str, asyncio.Future[HITLDecision]] = {}


class HITLGateway:
    """Handles human review routing for both sync and async modes.

    Sync mode:
        The LangGraph interrupt() mechanism pauses the graph.
        The orchestrator presents a CLI prompt and resumes on human input.

    Async mode:
        Results are queued in memory (or DB). The reviewer calls POST /hitl/decision
        via the REST API. The orchestrator polls or uses a Future to resume.
    """

    def __init__(self, mode: str = "sync") -> None:
        self.mode = mode  # "sync" | "async"

    # ── Sync HITL (CLI) ───────────────────────────────────────────────────────

    @staticmethod
    def prompt_cli(interrupt_payload: dict[str, Any]) -> HITLDecision:
        """Present review UI on the terminal and collect decision."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Confirm, Prompt

        console = Console()
        console.print(
            Panel(
                f"[bold cyan]Query:[/bold cyan] {interrupt_payload['query']}\n\n"
                f"[bold green]Response:[/bold green]\n{interrupt_payload['actor_response']}\n\n"
                f"[bold yellow]Confidence:[/bold yellow] "
                f"{interrupt_payload['critic_score']:.2f} "
                f"({interrupt_payload['confidence_tier']})\n"
                f"[dim]Reasoning:[/dim] {interrupt_payload['critic_reasoning']}\n\n"
                f"[dim]Context version used:[/dim] "
                f"v{interrupt_payload.get('context_version_used', 'N/A')}",
                title="🔍 HITL Review",
                border_style="bright_blue",
            )
        )

        action = Prompt.ask(
            "Decision",
            choices=["approve", "reject", "edit"],
            default="approve",
        )
        edited = None
        if action == "edit":
            console.print("[dim]Enter the corrected response (empty line to finish):[/dim]")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            edited = "\n".join(lines)

        comment = Prompt.ask("Comment (optional)", default="")
        reviewer_id = Prompt.ask("Reviewer ID", default="human")

        decision = HITLDecision(
            workflow_run_id=interrupt_payload["run_id"],
            decision=action,
            edited_response=edited,
            comment=comment or None,
            reviewer_id=reviewer_id,
        )
        logger.info(f"[hitl] CLI decision: {action} by {reviewer_id}")
        return decision

    # ── Async HITL (REST API) ─────────────────────────────────────────────────

    def enqueue(self, run_id: str, payload: dict[str, Any]) -> HITLQueueEntry:
        """Queue a workflow result for async review."""
        entry = HITLQueueEntry(
            run_id=run_id,
            query=payload["query"],
            domain=payload.get("domain", ""),
            topic_label=payload.get("topic_label", ""),
            actor_response=payload["actor_response"],
            critic_score=payload["critic_score"],
            critic_reasoning=payload["critic_reasoning"],
            confidence_tier=payload["confidence_tier"],
            context_version_used=payload.get("context_version_used"),
        )
        _queue[run_id] = entry

        # Create a Future that will be resolved when the reviewer posts a decision
        loop = asyncio.get_event_loop()
        _decision_futures[run_id] = loop.create_future()

        logger.info(f"[hitl] Queued run {run_id} for async review")
        return entry

    def update_queue_entry(self, run_id: str, payload: dict[str, Any]) -> None:
        """Update an existing queue entry with computed data from the workflow."""
        if run_id not in _queue:
            logger.warning(f"[hitl] Cannot update non-existent run {run_id}")
            return
        entry = _queue[run_id]
        entry.actor_response = payload.get("actor_response", entry.actor_response)
        entry.critic_score = payload.get("critic_score", entry.critic_score)
        entry.critic_reasoning = payload.get("critic_reasoning", entry.critic_reasoning)
        entry.confidence_tier = payload.get("confidence_tier", entry.confidence_tier)
        entry.context_version_used = payload.get("context_version_used", entry.context_version_used)
        logger.info(f"[hitl] Updated queue entry for run {run_id}")

    async def wait_for_decision(self, run_id: str) -> HITLDecision:
        """Await the reviewer's decision for the given run_id."""
        if run_id not in _decision_futures:
            raise KeyError(f"No pending HITL review for run_id={run_id}")
        decision = await _decision_futures[run_id]
        logger.info(f"[hitl] Received async decision for run {run_id}: {decision.decision}")
        return decision

    def resolve_decision(self, decision: HITLDecision) -> None:
        """Called by the REST router to submit a reviewer's decision."""
        run_id = decision.workflow_run_id
        if run_id not in _queue:
            raise KeyError(f"run_id={run_id} not found in HITL queue")

        entry = _queue[run_id]
        entry.status = decision.decision  # type: ignore[assignment]
        entry.decision = decision
        entry.reviewed_at = datetime.utcnow()

        future = _decision_futures.get(run_id)
        if future and not future.done():
            future.set_result(decision)
        logger.info(f"[hitl] Decision resolved for run {run_id}: {decision.decision}")

    def list_pending(self) -> list[HITLReviewPayload]:
        """Return all pending queue entries for the reviewer dashboard."""
        return [
            HITLReviewPayload(
                queue_id=e.queue_id,
                run_id=e.run_id,
                query=e.query,
                domain=e.domain,
                topic_label=e.topic_label,
                actor_response=e.actor_response,
                critic_score=e.critic_score,
                critic_reasoning=e.critic_reasoning,
                confidence_tier=e.confidence_tier,
                context_version_used=e.context_version_used,
                queued_at=e.queued_at,
            )
            for e in _queue.values()
            if e.status == "pending"
        ]
