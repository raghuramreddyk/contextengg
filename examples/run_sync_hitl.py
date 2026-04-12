"""examples/run_sync_hitl.py — End-to-end demo with synchronous CLI HITL review."""
from __future__ import annotations

from rich.console import Console

from pipeline.orchestrator import run

console = Console()

if __name__ == "__main__":
    console.rule("[bold blue]contextengg — Sync HITL Demo[/bold blue]")
    console.print("[dim]Running Actor-Critic pipeline with synchronous CLI review[/dim]\n")

    result = run(
        query="What are the incoterms rules for sea freight?",
        domain="global_trade",
        topic_label="incoterms",
        reviewer_id="demo_reviewer",
    )

    console.rule("[bold green]Result[/bold green]")
    console.print(f"Run ID     : {result.run_id}")
    console.print(f"Confidence : {result.critic_score:.2f} ({result.confidence_tier})")
    console.print(f"Context v  : {result.context_version_used or 'none'}")
    console.print(f"\n[bold]Final Response:[/bold]\n{result.actor_response}")
