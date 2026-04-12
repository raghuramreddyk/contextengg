"""core/llm_factory.py — Build LLM and embedding instances from config."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from core.config import AppConfig


def build_llm(config: AppConfig) -> BaseChatModel:
    """Return a configured chat model based on the provider setting."""
    if config.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
        )
    if config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
        )
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def build_embeddings(config: AppConfig) -> Embeddings:
    """Return a configured embedding model based on the provider setting."""
    if config.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=config.ollama_embedding_model,
            base_url=config.ollama_base_url,
        )
    if config.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=config.openai_api_key)
    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
