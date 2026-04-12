"""context_engine/retriever.py — Semantic retrieval of domain context chunks."""
from __future__ import annotations

import json

from langchain_core.embeddings import Embeddings as LCEmbeddings

from context_engine.schemas import ContextEntry
from context_engine.store import ContextStore
from core.logger import get_logger

logger = get_logger(__name__)


class ContextRetriever:
    """Retrieves the most relevant context entries for a given query.

    Strategy:
    - Always return the **latest approved version** for the resolved domain entries.
    - Performs semantic similarity search across domain-filtered entries.
    - Returns a formatted markdown string suitable for injection into the Actor prompt.
    """

    def __init__(
        self,
        store: ContextStore,
        lc_embeddings: LCEmbeddings,
        top_k: int = 5,
    ) -> None:
        self._store = store
        self._embeddings = lc_embeddings
        self._top_k = top_k

    def retrieve(self, query: str, domain: str) -> tuple[list[ContextEntry], str]:
        """Retrieve top-k relevant context chunks for *query* within *domain*.

        Returns:
            (entries, formatted_context_string)
        """
        # Embed the query
        query_embedding = self._embeddings.embed_query(query)

        # Query ChromaDB with domain filter
        results = self._store._collection.query(
            query_embeddings=[query_embedding],
            n_results=self._top_k,
            where={"domain": domain},
            include=["documents", "metadatas", "distances"],
        )

        entries: list[ContextEntry] = []
        if results["ids"] and results["ids"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                # Restore None for optional datetime fields (ChromaDB stores as "").
                for nullable_field in ("approved_at", "approved_by"):
                    if meta.get(nullable_field) == "":
                        meta[nullable_field] = None
                # Deserialise tasks from JSON string back to list[str].
                if isinstance(meta.get("tasks"), str):
                    try:
                        meta["tasks"] = json.loads(meta["tasks"])
                    except (json.JSONDecodeError, TypeError):
                        meta["tasks"] = []
                entries.append(ContextEntry(content=doc, **meta))

        if not entries:
            logger.warning(f"No context found for domain='{domain}', query='{query[:60]}'")
            return [], ""

        formatted = self._format_context(entries, domain)
        logger.info(
            f"Retrieved {len(entries)} context entries for domain='{domain}' "
            f"(versions: {[e.version for e in entries]})"
        )
        return entries, formatted

    @staticmethod
    def _format_context(entries: list[ContextEntry], domain: str) -> str:
        """Format entries into a rich markdown context block for the Actor system prompt.

        Structure per entry:
          ## Domain Context — <domain>
          ### Role
          ### Tasks
          ### Knowledge
        """
        lines = [
            f"## Domain Context — {domain}",
            "",
            "Use the following approved knowledge, role definition, and task list to "
            "ground and constrain your response.\n",
        ]

        for i, entry in enumerate(entries, start=1):
            lines.append(f"### [{i}] {entry.topic_label} (v{entry.version})")

            # ── Role ────────────────────────────────────────────────────────
            if entry.role:
                lines.append(f"**Role:** {entry.role}")
                lines.append("")

            # ── Tasks ───────────────────────────────────────────────────────
            if entry.tasks:
                lines.append("**Tasks you are expected to perform:**")
                for j, task in enumerate(entry.tasks, start=1):
                    lines.append(f"{j}. {task}")
                lines.append("")

            # ── Knowledge content ────────────────────────────────────────────
            lines.append("**Domain Knowledge:**")
            lines.append(entry.content)
            lines.append("")

        return "\n".join(lines)
