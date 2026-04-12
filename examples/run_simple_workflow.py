"""examples/run_simple_workflow.py — Simple generic Actor-Critic testing script using an external JSON context."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import typer
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.prompt import Prompt

from core.config import settings
from core.llm_factory import build_llm

app = typer.Typer(name="simple-workflow")
console = Console()
logger = logging.getLogger(__name__)

CONTEXT_FILE = Path(__file__).parent / "simple_context.json"

ACTOR_SYSTEM_PROMPT = """You are an AI assistant answering questions based on provided tasks and business rules.
Follow the business rules strictly. Use the provided tasks as grounding facts.

BUSINESS RULES:
{rules}

GROUNDING TASKS:
{tasks}
"""

CRITIC_SYSTEM_PROMPT = """You are a rigorous quality evaluator for AI-generated responses.
Evaluate if the ACTOR RESPONSE strictly adheres to the BUSINESS RULES and aligns with the expected answers in the GROUNDING TASKS.

BUSINESS RULES:
{rules}

GROUNDING TASKS:
{tasks}

Evaluate the response against the context and query. Return ONLY a JSON object with this exact schema:
{{
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one paragraph explaining the score>",
  "confidence_tier": "<HIGH|MEDIUM|LOW>"
}}

Scoring rubric:
- 0.85–1.0 (HIGH): Fully adheres to all rules and expected answers.
- 0.60–0.84 (MEDIUM): Mostly fits but misses minor strictness of the rules.
- 0.00–0.59 (LOW): Violates business rules or contradicts expected answers.
"""

def load_context() -> dict:
    if not CONTEXT_FILE.exists():
        console.print(f"[red]Error: Context file not found at {CONTEXT_FILE}[/red]")
        raise typer.Exit(1)
    with CONTEXT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_context(data: dict) -> None:
    with CONTEXT_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Context file updated at {CONTEXT_FILE}[/green]")

def bump_version(version_str: str) -> str:
    """Bump a version string from e.g. '1.0' to '1.1'."""
    try:
        parts = version_str.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except Exception:
        # fallback
        return str(float(version_str) + 0.1)

@app.command()
def run(
    query: str = typer.Argument(..., help="The query to ask the actor."),
):
    """Run a simple file-backed Actor-Critic + HITL workflow."""
    console.rule("[bold blue]Simple File-Backed Actor-Critic Workflow[/bold blue]")
    
    # 1. Load context
    ctx = load_context()
    version = ctx.get("version", "1.0")
    rules = ctx.get("business_rules", [])
    tasks = ctx.get("tasks", [])
    
    formatted_rules = "\n".join(f"- {r}" for r in rules)
    formatted_tasks = "\n".join(f"Q: {t['question']}\nA: {t['expected_answer']}" for t in tasks)
    
    console.print(f"[dim]Loaded Context Version: {version}[/dim]")
    
    llm = build_llm(settings)
    
    # 2. Actor Phase
    console.print("\n[bold]Running Actor...[/bold]")
    actor_sys = ACTOR_SYSTEM_PROMPT.format(rules=formatted_rules, tasks=formatted_tasks)
    actor_response = llm.invoke([
        SystemMessage(content=actor_sys),
        HumanMessage(content=f"QUERY: {query}")
    ]).content
    
    if not isinstance(actor_response, str):
        actor_response = str(actor_response)
        
    console.print(f"\n[cyan]Actor Response:[/cyan]\n{actor_response}")
    
    # 3. Critic Phase
    console.print("\n[bold]Running Critic...[/bold]")
    critic_sys = CRITIC_SYSTEM_PROMPT.format(rules=formatted_rules, tasks=formatted_tasks)
    human_msg = f"QUERY: {query}\n\nACTOR RESPONSE:\n{actor_response}"
    
    raw_critic = llm.invoke([
        SystemMessage(content=critic_sys),
        HumanMessage(content=human_msg)
    ]).content
    
    if not isinstance(raw_critic, str):
        raw_critic = str(raw_critic)
        
    try:
        import re
        cleaned = re.sub(r"```(?:json)?", "", raw_critic).replace("```", "").strip()
        critic_out = json.loads(cleaned)
    except Exception as exc:
        console.print(f"[yellow]Critic output parse failed: {exc}[/yellow]")
        critic_out = {"confidence": 0.5, "reasoning": "Parse failed", "confidence_tier": "LOW"}
    
    console.print(f"[magenta]Critic Score:[/magenta] {critic_out['confidence']} ({critic_out.get('confidence_tier')})")
    console.print(f"[magenta]Critic Reasoning:[/magenta] {critic_out.get('reasoning')}")
    
    # 4. HITL Phase
    console.rule("[bold green]HITL Review[/bold green]")
    choices = ["approve", "edit", "reject"]
    decision = Prompt.ask("Action", choices=choices, default="approve")
    
    if decision == "reject":
        console.print("[red]Workflow rejected by human.[/red]")
        raise typer.Exit(0)
    
    final_answer = actor_response
    if decision == "edit":
        console.print("\n[bold]Enter your corrected answer:[/bold]")
        # read multiline input or single line depending on implementation
        # simple single line for demo:
        new_answer = Prompt.ask("New Answer")
        final_answer = new_answer
        
        # 5. Update Context
        new_version = bump_version(version)
        ctx["version"] = new_version
        
        # Check if the exact task exists; if so, update, otherwise append
        existing_task = None
        for t in ctx["tasks"]:
            if t["question"] == query:
                existing_task = t
                break
                
        if existing_task:
            existing_task["expected_answer"] = final_answer
        else:
            ctx["tasks"].append({"question": query, "expected_answer": final_answer})
            
        save_context(ctx)
        console.print(f"[bold green]Context updated to version {new_version}[/bold green]")
        
    console.print(f"\n[bold green]Final Validated Answer:[/bold green]\n{final_answer}")

if __name__ == "__main__":
    app()
