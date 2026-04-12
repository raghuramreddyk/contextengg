"""context_engine/merger.py — LLM-based delta extraction and context merging."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from context_engine.schemas import ContextDelta, ContextEntry
from core.logger import get_logger

logger = get_logger(__name__)

_DELTA_EXTRACTION_SYSTEM = """You are a knowledge engineer. Given an original domain context document and a newly approved expert response, extract ONLY the genuinely new facts that are NOT already present in the current context.

Return your output in exactly this format:
FACTS:
- <fact 1>
- <fact 2>
...

SUMMARY:
<A concise paragraph (3-5 sentences) combining the existing context with the new facts, suitable for replacing the context document>
"""


class ContextMerger:
    """Extracts new knowledge from approved responses and merges with existing context."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._parser = StrOutputParser()

    def extract_delta(
        self,
        existing_entry: ContextEntry | None,
        approved_response: str,
        query: str,
        domain: str,
        topic_label: str,
        approved_by: str,
        confidence: float,
    ) -> ContextDelta:
        """Ask the LLM to extract new facts from the approved response."""
        existing_content = existing_entry.content if existing_entry else "(No prior context)"

        human_msg = (
            f"DOMAIN: {domain}\n"
            f"TOPIC: {topic_label}\n\n"
            f"--- EXISTING CONTEXT ---\n{existing_content}\n\n"
            f"--- APPROVED RESPONSE ---\nQuery: {query}\nResponse: {approved_response}\n\n"
            "Extract only the NEW facts not present in the existing context."
        )

        raw = self._llm.invoke(
            [SystemMessage(content=_DELTA_EXTRACTION_SYSTEM), HumanMessage(content=human_msg)]
        )
        text = self._parser.invoke(raw)

        facts, summary = self._parse_extraction(text)
        logger.info(f"Extracted {len(facts)} new facts for '{topic_label}' [{domain}]")

        return ContextDelta(
            original_query=query,
            approved_response=approved_response,
            extracted_facts=facts,
            summary=summary,
            domain=domain,
            topic_label=topic_label,
            confidence=confidence,
            approved_by=approved_by,
        )

    def build_merged_content(
        self,
        existing_entry: ContextEntry | None,
        delta: ContextDelta,
    ) -> str:
        """Build the full merged content string to store as the new context version."""
        if not delta.extracted_facts:
            # Nothing new — keep existing
            return existing_entry.content if existing_entry else delta.summary

        # Use the LLM-generated summary as the new canonical content,
        # appending an explicit "Recent additions" section for traceability.
        recent = "\n".join(f"- {f}" for f in delta.extracted_facts)
        return (
            f"{delta.summary}\n\n"
            f"### Recent Additions (approved by {delta.approved_by})\n"
            f"{recent}"
        )

    @staticmethod
    def _parse_extraction(text: str) -> tuple[list[str], str]:
        """Parse the structured LLM output into (facts_list, summary_string)."""
        facts: list[str] = []
        summary = ""
        in_facts = False
        in_summary = False
        summary_lines: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FACTS:"):
                in_facts = True
                in_summary = False
                continue
            if stripped.upper().startswith("SUMMARY:"):
                in_facts = False
                in_summary = True
                continue
            if in_facts and stripped.startswith("-"):
                facts.append(stripped.lstrip("- ").strip())
            elif in_summary and stripped:
                summary_lines.append(stripped)

        summary = " ".join(summary_lines)
        return facts, summary
