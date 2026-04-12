"""context_engine/store.py — ChromaDB-backed Context Store with full CRUD + versioning."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Embeddings
from langchain_core.embeddings import Embeddings as LCEmbeddings

from context_engine.schemas import ContextEntry
from context_engine.versioning import bump_version, initial_version
from core.logger import get_logger

logger = get_logger(__name__)


# ── ChromaDB embedding function adapter ──────────────────────────────────────

class _LangChainEmbeddingAdapter(EmbeddingFunction):
    """Wraps a LangChain Embeddings instance for ChromaDB."""

    def __init__(self, lc_embeddings: LCEmbeddings) -> None:
        self._lc = lc_embeddings

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002
        return self._lc.embed_documents(input)


# ── ContextStore ─────────────────────────────────────────────────────────────

class ContextStore:
    """Versioned, ChromaDB-backed store for domain context entries.

    Each entry is identified by (domain, topic_hash, version).
    Retrieval always uses the latest approved version unless overridden.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        lc_embeddings: LCEmbeddings,
    ) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed_fn = _LangChainEmbeddingAdapter(lc_embeddings)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )
        logger.info(f"ContextStore initialised — collection: '{collection_name}'")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def topic_hash(domain: str, topic_label: str) -> str:
        raw = f"{domain.lower().strip()}::{topic_label.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _entry_to_chroma(self, entry: ContextEntry) -> dict[str, Any]:
        meta = entry.model_dump(exclude={"content"})
        # ChromaDB metadata values must be str | int | float | bool.
        # Convert complex types before storing.
        for k, v in meta.items():
            if isinstance(v, datetime):
                meta[k] = v.isoformat()
            elif v is None:
                meta[k] = ""
            elif isinstance(v, list):
                # tasks (list[str]) → JSON string for ChromaDB compatibility
                meta[k] = json.dumps(v)
        return {
            "id": entry.entry_id,
            "document": entry.content,
            "metadata": meta,
        }

    # ── Latest version lookup ────────────────────────────────────────────────

    def get_latest_version(self, domain: str, topic_label: str) -> str | None:
        """Return the highest version string for this topic, or None if not seeded."""
        th = self.topic_hash(domain, topic_label)
        result = self._collection.get(
            where={"topic_hash": th},
            include=["metadatas"],
        )
        if not result["ids"]:
            return None
        versions = [m["version"] for m in result["metadatas"]]
        return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))

    # ── Write ────────────────────────────────────────────────────────────────

    def seed(
        self,
        domain: str,
        topic_label: str,
        content: str,
        role: str = "",
        tasks: list[str] | None = None,
    ) -> ContextEntry:
        """Insert an initial seed entry at version 1.0.

        Args:
            role:  Agent persona for this domain e.g. 'Senior Trade Compliance Expert'
            tasks: Ordered task list the agent should perform in this domain.
        """
        th = self.topic_hash(domain, topic_label)
        existing = self.get_latest_version(domain, topic_label)
        if existing:
            logger.warning(
                f"Seed skipped — '{topic_label}' in domain '{domain}' "
                f"already at v{existing}"
            )
            return self.get_by_version(domain, topic_label, existing)  # type: ignore[return-value]

        entry = ContextEntry(
            topic_hash=th,
            topic_label=topic_label,
            domain=domain,
            content=content,
            version=initial_version(),
            source="seed",
            role=role,
            tasks=tasks or [],
        )
        doc = self._entry_to_chroma(entry)
        self._collection.add(
            ids=[doc["id"]],
            documents=[doc["document"]],
            metadatas=[doc["metadata"]],
        )
        logger.info(f"Seeded context '{topic_label}' [{domain}] at v{entry.version}")
        return entry

    def save_update(
        self,
        domain: str,
        topic_label: str,
        new_content: str,
        approved_by: str,
        confidence: float,
        bump: str = "minor",
        role: str | None = None,
        tasks: list[str] | None = None,
    ) -> ContextEntry:
        """Save a new incremental version, bumped from the current latest.

        role and tasks are carried forward from the previous version when not supplied.
        """
        th = self.topic_hash(domain, topic_label)
        current_version = self.get_latest_version(domain, topic_label)
        next_ver = bump_version(current_version or "1.0", bump=bump)  # type: ignore[arg-type]

        # Carry forward role / tasks from the existing entry if not explicitly provided
        existing = self.retrieve_latest(domain, topic_label)
        inherited_role = role if role is not None else (existing.role if existing else "")
        inherited_tasks = tasks if tasks is not None else (existing.tasks if existing else [])

        entry = ContextEntry(
            topic_hash=th,
            topic_label=topic_label,
            domain=domain,
            content=new_content,
            version=next_ver,
            source="hitl_approved",
            confidence_at_creation=confidence,
            approved_by=approved_by,
            approved_at=datetime.utcnow(),
            role=inherited_role,
            tasks=inherited_tasks,
        )
        doc = self._entry_to_chroma(entry)
        self._collection.add(
            ids=[doc["id"]],
            documents=[doc["document"]],
            metadatas=[doc["metadata"]],
        )
        logger.info(
            f"Context updated '{topic_label}' [{domain}]: "
            f"v{current_version} → v{next_ver} (approved by {approved_by})"
        )
        return entry

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_by_version(
        self, domain: str, topic_label: str, version: str
    ) -> ContextEntry | None:
        th = self.topic_hash(domain, topic_label)
        result = self._collection.get(
            where={"$and": [{"topic_hash": th}, {"version": version}]},
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        # Restore None for optional fields (ChromaDB stores None as "").
        for nullable_field in ("approved_at", "approved_by"):
            if meta.get(nullable_field) == "":
                meta[nullable_field] = None
        # Deserialise tasks from JSON string back to list[str].
        if isinstance(meta.get("tasks"), str):
            try:
                meta["tasks"] = json.loads(meta["tasks"])
            except (json.JSONDecodeError, TypeError):
                meta["tasks"] = []
        return ContextEntry(content=result["documents"][0], **meta)

    def retrieve_latest(
        self, domain: str, topic_label: str
    ) -> ContextEntry | None:
        version = self.get_latest_version(domain, topic_label)
        if not version:
            return None
        return self.get_by_version(domain, topic_label, version)

    def list_versions(self, domain: str, topic_label: str) -> list[str]:
        """Return all stored versions for a topic, sorted ascending."""
        th = self.topic_hash(domain, topic_label)
        result = self._collection.get(
            where={"topic_hash": th},
            include=["metadatas"],
        )
        versions = [m["version"] for m in result["metadatas"]]
        return sorted(versions, key=lambda v: tuple(int(x) for x in v.split(".")))

    def rollback(
        self, domain: str, topic_label: str, target_version: str
    ) -> ContextEntry | None:
        """Return the entry at target_version (does not delete later versions)."""
        entry = self.get_by_version(domain, topic_label, target_version)
        if entry:
            logger.info(f"Rollback: using v{target_version} for '{topic_label}'")
        return entry
