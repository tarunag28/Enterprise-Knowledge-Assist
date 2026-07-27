"""
Configuration Loader
Loads YAML configuration files and provides typed access to all
configurable parameters (embedding models, chunking, retrieval, etc.).
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


# Project root is one level up from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Load environment variables from the workspace .env file.
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(filename: str) -> dict:
    """Load a YAML file from the config directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def _resolve_env_placeholders(value):
    """Expand environment placeholders such as ${VAR} in config values."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Prefer .env values for Ollama base URLs when present."""
    if "llm" in config and isinstance(config["llm"], dict):
        config["llm"]["base_url"] = os.getenv("LLM_BASE_URL", _resolve_env_placeholders(config["llm"].get("base_url")))
    if "embedding" in config and isinstance(config["embedding"], dict):
        config["embedding"]["base_url"] = os.getenv("EMBEDDING_BASE_URL", _resolve_env_placeholders(config["embedding"].get("base_url")))
    return config


def get_embedding_config() -> dict:
    """Load embedding and model configuration."""
    return _apply_env_overrides(_load_yaml("embedding_config.yaml"))


def get_retrieval_config() -> dict:
    """Load retrieval and chunking configuration."""
    return _load_yaml("retrieval_config.yaml")


# --- Convenience accessors ---

def get_llm_settings() -> dict:
    """Get LLM-specific settings (model, base_url, temperature)."""
    config = get_embedding_config()
    return config.get("llm", {})


def get_embedding_settings() -> dict:
    """Get embedding model settings."""
    config = get_embedding_config()
    return config.get("embedding", {})


def get_vectorstore_settings() -> dict:
    """Get vector store settings (type, persist directory, collection)."""
    config = get_embedding_config()
    settings = config.get("vectorstore", {})
    # Resolve persist_directory relative to project root
    persist_dir = settings.get("persist_directory", "./chroma_db")
    if not os.path.isabs(persist_dir):
        settings["persist_directory"] = str(PROJECT_ROOT / persist_dir)
    return settings


def get_chunking_settings() -> dict:
    """Get document chunking settings (size, overlap, separators)."""
    config = get_retrieval_config()
    return config.get("chunking", {})


def get_retrieval_settings() -> dict:
    """Get retrieval settings (top_k, threshold, search type)."""
    config = get_retrieval_config()
    return config.get("retrieval", {})
