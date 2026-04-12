"""examples/seed_global_trade.py — Seed all Global Trade domain topics from the rich JSON spec.

Reads examples/data/global_trade_seed.json and seeds one ContextStore entry per logical
topic block (incoterms, trade_finance, hs_codes, trade_agreements, logistics, sanctions,
documentation, risk_management).  Each entry carries:
  - content : formatted markdown knowledge block
  - role    : actor persona (from structural_role.actor_persona)
  - tasks   : topic-specific task list

Run:
    python examples/seed_global_trade.py
or via CLI:
    contextengg seed global_trade incoterms --file <path>   (for a single topic via file)
"""
from __future__ import annotations

import json
import pathlib
import textwrap

from rich.console import Console
from rich.table import Table

from context_engine.store import ContextStore
from core.config import settings
from core.llm_factory import build_embeddings

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
_SEED_FILE = _HERE / "data" / "global_trade_seed.json"

DOMAIN = "global_trade"


# ── Content builders — one per logical topic ──────────────────────────────────

def _build_incoterms(data: dict) -> str:
    inc = data["domain_knowledge"]["incoterms_2020"]
    lines = [
        "# Incoterms 2020 — International Commercial Terms",
        "",
        f"**Version:** {inc['version']}",
        f"**Publisher:** {inc['publisher']}",
        "",
        f"> ⚠️  {inc['critical_note']}",
        "",
        "## Rules",
        "",
    ]
    for rule in inc["rules"]:
        lines.append(f"### {rule['term']}")
        lines.append(f"- **Risk transfers:** {rule['risk_transfer']}")
        lines.append(f"- **Suitable for:** {rule.get('suitable_for', 'N/A')}")
        if "seller_obligation" in rule:
            lines.append(f"- **Seller obligation:** {rule['seller_obligation']}")
        if "buyer_obligation" in rule:
            lines.append(f"- **Buyer obligation:** {rule['buyer_obligation']}")
        if "cost_to_seller" in rule:
            lines.append(f"- **Cost to seller:** {rule['cost_to_seller']}")
        if "note" in rule:
            lines.append(f"- **Note:** {rule['note']}")
        if "common_misuse" in rule:
            lines.append(f"- **⚠️  Common misuse:** {rule['common_misuse']}")
        if "lc_usage" in rule:
            lines.append(f"- **LC usage:** {rule['lc_usage']}")
        lines.append("")

    # Relevant business rules
    rules_section = [
        r for r in data["domain_knowledge"]["business_rules"]
        if any(kw in r.lower() for kw in ["incoterm", "fob", "cif", "fca", "cip", "risk", "container"])
    ]
    if rules_section:
        lines.append("## Key Business Rules")
        for r in rules_section:
            lines.append(f"- {r}")
    return "\n".join(lines)


def _build_trade_finance(data: dict) -> str:
    instruments = data["domain_knowledge"]["entities"]["instruments"]
    parties = [
        p for p in data["domain_knowledge"]["entities"]["parties"]
        if "bank" in p["name"].lower() or "issuing" in p["name"].lower() or "advising" in p["name"].lower()
    ]
    lines = [
        "# Trade Finance Instruments",
        "",
        "## Instruments",
        "",
    ]
    for inst in instruments:
        lines.append(f"### {inst['name']}")
        if "governed_by" in inst:
            lines.append(f"- **Governed by:** {inst['governed_by']}")
        if "types" in inst:
            lines.append(f"- **Types:** {', '.join(inst['types'])}")
        if "key_principle" in inst:
            lines.append(f"- **Key principle:** {inst['key_principle']}")
        if "critical_rule" in inst:
            lines.append(f"- **⚠️  Critical rule:** {inst['critical_rule']}")
        if "risk_level" in inst:
            lines.append(f"- **Risk level:** {inst['risk_level']}")
        if "description" in inst:
            lines.append(f"- **Description:** {inst['description']}")
        if "risk" in inst:
            lines.append(f"- **Risk:** {inst['risk']}")
        if "mitigation" in inst:
            lines.append(f"- **Mitigation:** {inst['mitigation']}")
        lines.append("")

    if parties:
        lines.append("## Key Banking Parties")
        for p in parties:
            lines.append(f"### {p['name']}")
            lines.append(f"{p['role']}")
            if "governed_by" in p:
                lines.append(f"**Governed by:** {p['governed_by']}")
            lines.append("")

    # Relevant business rules
    finance_rules = [
        r for r in data["domain_knowledge"]["business_rules"]
        if any(kw in r.lower() for kw in ["letter of credit", "ucp", "bank", "document", "banking days", "irrevocable"])
    ]
    if finance_rules:
        lines.append("## Key Business Rules")
        for r in finance_rules:
            lines.append(f"- {r}")

    # Relevant glossary entries
    finance_terms = ["LC", "B/L", "AWB"]
    glossary = [
        g for g in data["domain_knowledge"]["terminology_glossary"]
        if g["term"] in finance_terms
    ]
    if glossary:
        lines.append("\n## Glossary")
        for g in glossary:
            lines.append(f"- **{g['term']}:** {g['definition']}")

    return "\n".join(lines)


def _build_hs_codes(data: dict) -> str:
    hs = data["domain_knowledge"]["hs_codes"]
    lines = [
        "# Harmonized System (HS) Codes — Tariff Classification",
        "",
        hs["description"],
        "",
        "## Key Facts",
        "",
    ]
    for fact in hs["key_facts"]:
        lines.append(f"- {fact}")
    lines.append("")
    lines.append("## Common HS Chapters")
    lines.append("")
    lines.append("| Chapter | Description |")
    lines.append("|---------|-------------|")
    for ch in hs["common_chapters"]:
        lines.append(f"| {ch['chapter']} | {ch['description']} |")

    hs_rules = [
        r for r in data["domain_knowledge"]["business_rules"]
        if any(kw in r.lower() for kw in ["hs code", "classification", "tariff", "advance ruling", "duty"])
    ]
    if hs_rules:
        lines.append("")
        lines.append("## Key Business Rules")
        for r in hs_rules:
            lines.append(f"- {r}")
    return "\n".join(lines)


def _build_trade_agreements(data: dict) -> str:
    agreements = data["domain_knowledge"]["trade_agreements"]
    lines = [
        "# Trade Agreements and Preferential Tariffs",
        "",
    ]
    for ag in agreements:
        lines.append(f"## {ag['name']}")
        if "full_name" in ag:
            lines.append(f"**Full name:** {ag['full_name']}")
        if "replaced" in ag:
            lines.append(f"**Replaced:** {ag['replaced']}")
        if "effective" in ag:
            lines.append(f"**Effective:** {ag['effective']}")
        if "launched" in ag:
            lines.append(f"**Launched:** {ag['launched']}")
        if "members" in ag:
            lines.append(f"**Members:** {', '.join(ag['members'])}")
        if "coverage" in ag:
            lines.append(f"**Coverage:** {ag['coverage']}")
        if "key_benefit" in ag:
            lines.append(f"**Key benefit:** {ag['key_benefit']}")
        if "origin_rule" in ag:
            lines.append(f"**Rules of Origin:** {ag['origin_rule']}")
        if "critical_document" in ag:
            lines.append(f"**Critical document:** {ag['critical_document']}")
        if "note" in ag:
            lines.append(f"**Note:** {ag['note']}")
        if "preferential_scheme" in ag:
            lines.append(f"**Preferential scheme:** {ag['preferential_scheme']}")
        if "status" in ag:
            lines.append(f"**Status:** {ag['status']}")
        lines.append("")

    # Glossary
    agreement_terms = ["FTA", "GSP", "MFN", "RVC", "TCC", "COO"]
    glossary = [
        g for g in data["domain_knowledge"]["terminology_glossary"]
        if g["term"] in agreement_terms
    ]
    if glossary:
        lines.append("## Glossary")
        for g in glossary:
            lines.append(f"- **{g['term']}:** {g['definition']}")

    return "\n".join(lines)


def _build_logistics(data: dict) -> str:
    glossary_terms = ["FCL", "LCL", "B/L", "AWB", "ATA Carnet", "TIR", "Inward Processing Relief"]
    glossary = [
        g for g in data["domain_knowledge"]["terminology_glossary"]
        if g["term"] in glossary_terms
    ]
    lines = [
        "# Logistics, Freight and Customs Transit",
        "",
        "## Cargo Modes and Load Types",
        "",
        "- **FCL (Full Container Load):** Shipper occupies entire 20ft or 40ft container exclusively.",
        "- **LCL (Less than Container Load):** Cargo consolidated with other shippers in a shared container.",
        "- **Air Freight:** Fastest but highest cost per kg. Non-negotiable AWB issued by carrier.",
        "- **Multimodal:** Combination of two or more transport modes under a single contract and document.",
        "- **Bonded Warehousing:** Goods stored under customs supervision; duties deferred until release.",
        "",
        "## Key Parties in Logistics",
        "",
    ]
    for p in data["domain_knowledge"]["entities"]["parties"]:
        if p["name"] in ("Freight Forwarder", "Carrier"):
            lines.append(f"### {p['name']}")
            lines.append(p["role"])
            if "key_obligations" in p:
                lines.append("**Obligations:**")
                for ob in p["key_obligations"]:
                    lines.append(f"  - {ob}")
            if "key_document" in p:
                lines.append(f"**Key document:** {p['key_document']}")
            if "types" in p:
                lines.append(f"**Types:** {', '.join(p['types'])}")
            lines.append("")

    demurrage_rule = next(
        (r for r in data["domain_knowledge"]["business_rules"] if "demurrage" in r.lower()), None
    )
    if demurrage_rule:
        lines.append("## Key Business Rules")
        lines.append(f"- {demurrage_rule}")
        lines.append("")

    if glossary:
        lines.append("## Glossary")
        for g in glossary:
            lines.append(f"- **{g['term']}:** {g['definition']}")

    return "\n".join(lines)


def _build_sanctions(data: dict) -> str:
    lines = [
        "# Sanctions, Export Controls and Dual-Use Regulations",
        "",
        "## Key Regulatory Regimes",
        "",
        "### OFAC (Office of Foreign Assets Control)",
        "US Treasury agency administering economic sanctions. SDN (Specially Designated Nationals) list",
        "must be screened before any transaction involving US persons, US dollars, or US-origin goods.",
        "",
        "### BIS EAR (Export Administration Regulations)",
        "US Department of Commerce regulations governing export of dual-use goods and technology.",
        "Items on the Commerce Control List (CCL) require an export licence to restricted destinations.",
        "",
        "### EU Dual-Use Regulation (2021/821)",
        "EU framework controlling export of dual-use items. Exporters must check the EU Common Military",
        "List and Annex I of the regulation. Catch-all controls apply for WMD-risk end uses.",
        "",
        "## Screening Requirements",
        "",
        "Sanctions screening must cover ALL transaction parties:",
        "- Exporter / Seller",
        "- Importer / Buyer",
        "- Ultimate Consignee",
        "- Freight Forwarder / Carrier",
        "- Issuing Bank and Correspondent Banks",
        "",
        "## Key Business Rules",
        "",
    ]
    sanctions_rules = [
        r for r in data["domain_knowledge"]["business_rules"]
        if any(kw in r.lower() for kw in ["sanction", "dual-use", "export licence", "export control"])
    ]
    for r in sanctions_rules:
        lines.append(f"- {r}")

    glossary = [
        g for g in data["domain_knowledge"]["terminology_glossary"]
        if g["term"] in ("OFAC", "EAR")
    ]
    if glossary:
        lines.append("")
        lines.append("## Glossary")
        for g in glossary:
            lines.append(f"- **{g['term']}:** {g['definition']}")

    return "\n".join(lines)


def _build_documentation(data: dict) -> str:
    lines = [
        "# Trade Documentation",
        "",
        "## Core Export Documents",
        "",
        "### Commercial Invoice",
        "Primary trade document stating price, quantity, and terms. Must match LC and packing list exactly.",
        "",
        "### Packing List",
        "Details each package: dimensions, weight, content. Required by customs and carrier.",
        "",
        "### Certificate of Origin (COO)",
        "Certifies country of manufacture. Required to claim preferential tariff rates under FTAs.",
        "Must be issued at or before time of export — backdated certificates are rejected.",
        "",
        "### Bill of Lading (B/L)",
        "Ocean carrier's receipt for goods, contract of carriage, and (if negotiable) document of title.",
        "An 'On Board' B/L confirms goods are loaded on the vessel — required under most LCs.",
        "",
        "### Air Waybill (AWB)",
        "Non-negotiable transport document for air freight. Consignee may collect goods without presenting the original.",
        "",
        "### Export Declaration / Customs Entry",
        "Filed with customs authority at origin (and destination). Requires HS code, value, origin, and Incoterm.",
        "",
        "## Key Business Rules",
        "",
    ]
    doc_rules = [
        r for r in data["domain_knowledge"]["business_rules"]
        if any(kw in r.lower() for kw in ["certificate", "document", "invoice", "declaration", "customs entry"])
    ]
    for r in doc_rules:
        lines.append(f"- {r}")

    glossary_terms = ["B/L", "AWB", "COO"]
    glossary = [
        g for g in data["domain_knowledge"]["terminology_glossary"]
        if g["term"] in glossary_terms
    ]
    if glossary:
        lines.append("")
        lines.append("## Glossary")
        for g in glossary:
            lines.append(f"- **{g['term']}:** {g['definition']}")

    return "\n".join(lines)


def _build_risk_management(data: dict) -> str:
    lines = [
        "# Risk Management in Global Trade",
        "",
        "## Risk Categories",
        "",
        "### Country Risk",
        "Political instability, foreign exchange controls, expropriation, or war in the buyer's country.",
        "Mitigated by: export credit insurance (e.g. UKEF, Euler Hermes), confirmed LCs, advance payment.",
        "",
        "### Counterparty Risk",
        "Buyer insolvency or deliberate non-payment. Higher under Open Account terms.",
        "Mitigated by: Letter of Credit (bank substitutes buyer's credit), trade credit insurance, factoring.",
        "",
        "### Currency Risk (FX Risk)",
        "Adverse exchange rate movements between contract date and payment date.",
        "Mitigated by: FX forward contracts, currency options, LC in currency of cost base.",
        "",
        "### Political Risk",
        "Sanctions, trade embargoes, import/export bans imposed after contract signing.",
        "Mitigated by: Force majeure clauses, political risk insurance, advance payment.",
        "",
        "### Logistics / Cargo Risk",
        "Goods damaged, lost, or delayed in transit.",
        "Mitigated by: Marine cargo insurance (Institute Cargo Clauses A = all risk), proper packing, carrier vetting.",
        "",
        "## Payment Method Risk Spectrum",
        "",
        "| Payment Method | Exporter Risk | Importer Risk |",
        "|----------------|---------------|---------------|",
        "| Advance Payment | Lowest | Highest |",
        "| Letter of Credit | Low (if compliant docs) | Low-Medium |",
        "| Documentary Collection (D/P) | Medium | Low |",
        "| Documentary Collection (D/A) | High | Low |",
        "| Open Account | Highest | Lowest |",
        "",
        "## Key Business Rules",
        "",
    ]
    risk_rules = [
        r for r in data["domain_knowledge"]["business_rules"]
        if any(kw in r.lower() for kw in ["risk", "insurance", "credit", "sanction"])
    ]
    for r in risk_rules:
        lines.append(f"- {r}")

    # Full glossary for risk section
    lines.append("")
    lines.append("## Key Terminology")
    for g in data["domain_knowledge"]["terminology_glossary"]:
        lines.append(f"- **{g['term']}:** {g['definition']}")

    return "\n".join(lines)


# ── Topic registry ─────────────────────────────────────────────────────────────

def _build_topic_registry(data: dict) -> list[dict]:
    """Return ordered list of {topic_label, role, tasks, content} dicts."""
    actor_role = data["structural_role"]["actor_persona"]

    return [
        {
            "topic_label": "incoterms",
            "role": actor_role,
            "tasks": [
                "Determine the applicable Incoterm for a described shipment scenario",
                "Identify which party (buyer or seller) bears cost and risk at each stage",
                "Explain the precise point at which risk transfers between parties",
                "Flag any non-standard or ambiguous trade terms that require legal review",
                "Recommend the most appropriate Incoterm based on the mode of transport",
            ],
            "content": _build_incoterms(data),
        },
        {
            "topic_label": "trade_finance",
            "role": actor_role,
            "tasks": [
                "Identify the appropriate trade finance instrument for a given transaction risk profile",
                "Explain the obligations of each bank in an LC transaction under UCP 600",
                "List document requirements and common discrepancies for Letter of Credit presentations",
                "Compare risk and cost trade-offs across LC, documentary collection, and open account",
                "Advise on mitigants (credit insurance, factoring) for open account transactions",
            ],
            "content": _build_trade_finance(data),
        },
        {
            "topic_label": "hs_codes",
            "role": actor_role,
            "tasks": [
                "Classify a described good into the correct HS chapter and heading",
                "Identify the 6-digit HS subheading for internationally harmonised classification",
                "Advise on national tariff line extensions (7–10 digit) for key import countries",
                "Flag mis-classification risks and recommend seeking an Advance Ruling",
                "Explain the difference between HS classification for duty and for export control purposes",
            ],
            "content": _build_hs_codes(data),
        },
        {
            "topic_label": "trade_agreements",
            "role": actor_role,
            "tasks": [
                "Determine whether a shipment qualifies for preferential tariff under a named FTA",
                "Explain the applicable Rules of Origin (RVC or TCC) for a given trade corridor",
                "Identify the correct certificate of origin format required by the importing country",
                "Advise on GSP eligibility and cumulation rules for developing country exporters",
                "Compare tariff outcomes under MFN vs FTA rates for a specific HS code",
            ],
            "content": _build_trade_agreements(data),
        },
        {
            "topic_label": "logistics",
            "role": actor_role,
            "tasks": [
                "Recommend the appropriate cargo mode (FCL, LCL, air, multimodal) for a shipment",
                "Explain demurrage and detention rules and how to minimise exposure",
                "Advise on bonded warehousing and deferred duty options",
                "Clarify the role and liability of freight forwarders vs. carriers",
                "Guide on ATA Carnet and TIR for temporary import and transit shipments",
            ],
            "content": _build_logistics(data),
        },
        {
            "topic_label": "sanctions",
            "role": actor_role,
            "tasks": [
                "Screen all transaction parties against OFAC SDN, BIS Entity List, and EU sanctions lists",
                "Determine if a good requires an export licence under BIS EAR or EU Dual-Use Regulation",
                "Advise on catch-all controls and red flags for WMD-related end uses",
                "Explain the sanctions risk in a described trade corridor or counterparty relationship",
                "Recommend a sanctions compliance workflow for a trade finance team",
            ],
            "content": _build_sanctions(data),
        },
        {
            "topic_label": "documentation",
            "role": actor_role,
            "tasks": [
                "List all required export documents for a described shipment and destination",
                "Identify discrepancies between trade documents that would cause an LC rejection",
                "Advise on certificate of origin requirements for preferential tariff claims",
                "Explain the difference between a negotiable and straight Bill of Lading",
                "Guide on customs entry filing requirements for a specific import country",
            ],
            "content": _build_documentation(data),
        },
        {
            "topic_label": "risk_management",
            "role": actor_role,
            "tasks": [
                "Assess the overall risk profile of a described cross-border transaction",
                "Recommend the appropriate payment method based on buyer/country risk",
                "Advise on FX hedging strategies for a multi-currency trade transaction",
                "Explain marine cargo insurance options (ICC A/B/C) and when each applies",
                "Identify political risk mitigation options for high-risk destination countries",
            ],
            "content": _build_risk_management(data),
        },
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold blue]contextengg — Global Trade Seed[/bold blue]")
    console.print(f"[dim]Loading seed file: {_SEED_FILE}[/dim]\n")

    seed_data = json.loads(_SEED_FILE.read_text(encoding="utf-8"))

    embeddings = build_embeddings(settings)
    store = ContextStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.context_collection_name,
        lc_embeddings=embeddings,
    )

    topics = _build_topic_registry(seed_data)

    table = Table(title="Seed Results", show_lines=True)
    table.add_column("Topic", style="cyan", no_wrap=True)
    table.add_column("Version", style="green")
    table.add_column("Tasks", justify="right")
    table.add_column("Content (chars)", justify="right")
    table.add_column("Status", style="bold")

    for topic in topics:
        entry = store.seed(
            domain=DOMAIN,
            topic_label=topic["topic_label"],
            content=topic["content"],
            role=topic["role"],
            tasks=topic["tasks"],
        )
        already_existed = store.get_latest_version(DOMAIN, topic["topic_label"]) != "1.0"
        status = "[yellow]already existed[/yellow]" if already_existed and entry.version != "1.0" else "[green]OK seeded[/green]"
        table.add_row(
            entry.topic_label,
            f"v{entry.version}",
            str(len(entry.tasks)),
            str(len(entry.content)),
            status,
        )

    console.print(table)
    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"Seeded [cyan]{len(topics)}[/cyan] topics under domain "
        f"[cyan]'{DOMAIN}'[/cyan] into ChromaDB at [dim]{settings.chroma_persist_dir}[/dim]"
    )
    console.print(
        "\n[dim]Run a workflow:[/dim]\n"
        "  python examples/run_sync_hitl.py\n"
        "[dim]or via CLI:[/dim]\n"
        "  contextengg run \"What Incoterm should I use for containerised cargo?\" "
        "--domain global_trade --topic incoterms\n"
    )
