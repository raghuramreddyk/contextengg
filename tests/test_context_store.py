"""tests/test_context_store.py — Integration tests for ContextStore using a temp ChromaDB."""
from __future__ import annotations

import shutil

import pytest

from context_engine.store import ContextStore
from chromadb import EmbeddingFunction, Embeddings as ChromaEmbeddings
from langchain_core.embeddings import Embeddings as LCEmbeddings


# ── Stub embedding function (avoids Ollama in CI) ─────────────────────────────

class _FixedEmbeddings(LCEmbeddings):
    """Returns deterministic 128-dim vectors for any input."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 128


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path):
    embeddings = _FixedEmbeddings()
    s = ContextStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_context",
        lc_embeddings=embeddings,
    )
    yield s
    # Windows-safe cleanup
    shutil.rmtree(str(tmp_path), ignore_errors=True)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_seed_creates_v1_0(store):
    entry = store.seed("trade", "incoterms", "FOB means seller loads vessel.")
    assert entry.version == "1.0"
    assert entry.source == "seed"
    assert entry.domain == "trade"
    assert entry.role == ""
    assert entry.tasks == []


def test_seed_with_role_and_tasks(store):
    entry = store.seed(
        domain="trade",
        topic_label="incoterms",
        content="FOB content.",
        role="Senior Trade Expert",
        tasks=["Identify Incoterms", "Determine risk transfer point"],
    )
    assert entry.role == "Senior Trade Expert"
    assert entry.tasks == ["Identify Incoterms", "Determine risk transfer point"]


def test_role_and_tasks_round_trip(store):
    """role and tasks must survive the ChromaDB serialise/deserialise cycle."""
    store.seed(
        domain="trade",
        topic_label="incoterms",
        content="Content.",
        role="Expert",
        tasks=["Task A", "Task B", "Task C"],
    )
    retrieved = store.retrieve_latest("trade", "incoterms")
    assert retrieved is not None
    assert retrieved.role == "Expert"
    assert retrieved.tasks == ["Task A", "Task B", "Task C"]


def test_seed_idempotent(store):
    store.seed("trade", "incoterms", "Initial content.")
    # Second seed with different content should be skipped
    entry2 = store.seed("trade", "incoterms", "Different content.")
    assert entry2.version == "1.0"  # Still v1.0 — no duplicate


def test_save_update_bumps_version(store):
    store.seed("trade", "incoterms", "Initial.")
    updated = store.save_update(
        domain="trade",
        topic_label="incoterms",
        new_content="Initial + new fact.",
        approved_by="reviewer_1",
        confidence=0.9,
    )
    assert updated.version == "1.1"
    assert updated.approved_by == "reviewer_1"
    assert updated.source == "hitl_approved"


def test_role_and_tasks_inherited_on_update(store):
    """role and tasks are automatically carried forward on version bump."""
    store.seed(
        "trade", "incoterms", "v1.0",
        role="Expert",
        tasks=["Task A", "Task B"],
    )
    updated = store.save_update("trade", "incoterms", "v1.1", "r1", 0.9)
    assert updated.role == "Expert"
    assert updated.tasks == ["Task A", "Task B"]


def test_role_and_tasks_can_be_overridden_on_update(store):
    """Caller can override role/tasks on a specific version bump."""
    store.seed("trade", "incoterms", "v1.0", role="Junior", tasks=["Old task"])
    updated = store.save_update(
        "trade", "incoterms", "v1.1", "r1", 0.9,
        role="Senior Expert",
        tasks=["New task 1", "New task 2"],
    )
    assert updated.role == "Senior Expert"
    assert updated.tasks == ["New task 1", "New task 2"]


def test_multiple_updates_increment(store):
    store.seed("trade", "incoterms", "v1.0")
    store.save_update("trade", "incoterms", "v1.1", "r1", 0.9)
    store.save_update("trade", "incoterms", "v1.2", "r1", 0.88)
    latest = store.get_latest_version("trade", "incoterms")
    assert latest == "1.2"


def test_major_bump(store):
    store.seed("trade", "incoterms", "v1.0")
    updated = store.save_update(
        "trade", "incoterms", "major rework", "architect", 1.0, bump="major"
    )
    assert updated.version == "2.0"


def test_list_versions(store):
    store.seed("trade", "incoterms", "v1.0")
    store.save_update("trade", "incoterms", "v1.1", "r1", 0.9)
    versions = store.list_versions("trade", "incoterms")
    assert versions == ["1.0", "1.1"]


def test_rollback(store):
    store.seed("trade", "incoterms", "v1.0 content")
    store.save_update("trade", "incoterms", "v1.1 content", "r1", 0.9)
    entry = store.rollback("trade", "incoterms", "1.0")
    assert entry is not None
    assert entry.version == "1.0"
    assert "v1.0" in entry.content


def test_get_latest_none_when_empty(store):
    assert store.get_latest_version("empty_domain", "no_topic") is None
