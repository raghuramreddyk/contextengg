"""core/config.py — Application-wide settings loaded from .env"""
from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai", "azure"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    ollama_embedding_model: str = "nomic-embed-text"

    # ── Context Engine ────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_data"
    context_collection_name: str = "domain_context"
    top_k_context: int = Field(default=5, ge=1, le=20)

    # ── Actor-Critic ──────────────────────────────────────────────────────────
    confidence_high: float = Field(default=0.85, ge=0.0, le=1.0)
    confidence_medium: float = Field(default=0.60, ge=0.0, le=1.0)

    # ── HITL ──────────────────────────────────────────────────────────────────
    hitl_mode: Literal["sync", "async"] = "sync"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Singleton — import this everywhere
settings = AppConfig()
