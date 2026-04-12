"""examples/seed_context.py — Seed initial domain context from inline text."""
from __future__ import annotations

from rich.console import Console

from context_engine.store import ContextStore
from core.config import settings
from core.llm_factory import build_embeddings

console = Console()

SEED_DATA = {
    "domain": "global_trade",
    "topic_label": "incoterms",

    # ── Role ──────────────────────────────────────────────────────────────────
    "role": (
        "Senior Global Trade Compliance Expert with deep expertise in international "
        "commercial terms (Incoterms 2020), customs procedures, and cross-border "
        "logistics obligations."
    ),

    # ── Tasks ─────────────────────────────────────────────────────────────────
    "tasks": [
        "Determine the applicable Incoterm for a described shipment scenario",
        "Identify which party (buyer or seller) bears cost and risk at each stage",
        "Explain the precise point at which risk transfers between parties",
        "Flag any non-standard or ambiguous trade terms that require legal review",
        "Recommend the most appropriate Incoterm based on the mode of transport",
    ],

    # ── Domain Knowledge ──────────────────────────────────────────────────────
    "content": """
# Incoterms — International Commercial Terms (2020)

Incoterms are a set of 11 internationally recognised rules published by the
International Chamber of Commerce (ICC) that define the responsibilities of
sellers and buyers in international trade transactions.

## Key Rules

### Sea & Inland Waterway Only
- **FAS (Free Alongside Ship)**: Seller delivers when goods are placed alongside the vessel
  at the named port. Risk transfers from seller to buyer at that moment.
- **FOB (Free On Board)**: Seller bears cost and risk until goods are loaded on board the
  vessel at the named port of shipment. Most commonly used Incoterm.
- **CFR (Cost and Freight)**: Seller pays freight to destination port; risk transfers when
  goods are on board at origin.
- **CIF (Cost, Insurance and Freight)**: Like CFR but seller also arranges minimum marine
  insurance for the buyer's benefit.

### All Modes of Transport
- **EXW (Ex Works)**: Maximum obligation on buyer. Seller's responsibility ends at their
  own premises; buyer handles loading, export clearance, and all freight.
- **FCA (Free Carrier)**: Seller delivers to a named carrier at a named place. Often
  preferred over FOB for containerised cargo.
- **CPT (Carriage Paid To)**: Seller pays freight to named destination; risk transfers when
  goods are handed to the first carrier.
- **CIP (Carriage and Insurance Paid To)**: Like CPT but with higher insurance requirement
  (Institute Cargo Clauses A, i.e. all-risk).
- **DAP (Delivered at Place)**: Seller delivers at named destination, unloaded. Import
  duties remain with buyer.
- **DPU (Delivered at Place Unloaded)**: Seller unloads at destination. Only Incoterm where
  seller is responsible for unloading.
- **DDP (Delivered Duty Paid)**: Maximum obligation on seller. Seller delivers cleared for
  import, duties paid, at buyer's named location.

## Core Principle
Risk transfers at a specific, clearly defined point for each Incoterm.
Always specify the named place/port precisely to avoid disputes.

## 2020 Changes from 2010
- FCA now allows buyer to instruct carrier to issue on-board bill of lading to seller.
- CIP raised minimum insurance to ICC Clause A (all-risk); CIF remains Clause C.
- DAT renamed to DPU.
""".strip(),
}


if __name__ == "__main__":
    embeddings = build_embeddings(settings)
    store = ContextStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.context_collection_name,
        lc_embeddings=embeddings,
    )
    entry = store.seed(**SEED_DATA)
    console.print(
        f"\n[bold green]✓ Seeded context[/bold green]\n"
        f"  Topic    : {entry.topic_label} [{entry.domain}]\n"
        f"  Version  : v{entry.version}\n"
        f"  Role     : {entry.role[:80]}{'...' if len(entry.role) > 80 else ''}\n"
        f"  Tasks    : {len(entry.tasks)} defined\n"
        f"  Entry ID : {entry.entry_id[:8]}..."
    )
    for i, t in enumerate(entry.tasks, 1):
        console.print(f"  [dim]{i}. {t}[/dim]")
