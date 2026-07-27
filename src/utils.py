"""
Utility functions shared across the Enterprise Knowledge Assistant.
"""

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts/ directory.
    
    Args:
        prompt_name: Name of the prompt file (e.g., 'retrieval_prompt.txt')
    
    Returns:
        The prompt template string.
    """
    prompt_path = PROJECT_ROOT / "prompts" / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text().strip()


def load_document_metadata() -> dict:
    """
    Load document metadata from metadata.json.
    
    Returns:
        Dictionary mapping filenames to metadata.
    """
    metadata_path = PROJECT_ROOT / "metadata.json"
    if not metadata_path.exists():
        return {}
    
    with open(metadata_path, "r") as f:
        data = json.load(f)
    
    # Convert to a lookup dict by filename
    metadata_map = {}
    for doc in data.get("documents", []):
        filename = doc.get("filename", "")
        metadata_map[filename] = {
            "department": doc.get("department", "Unknown"),
            "owner": doc.get("owner", "Unknown"),
            "access_level": doc.get("access_level", "all_employees"),
            "description": doc.get("description", ""),
        }
    return metadata_map


def get_documents_dir() -> Path:
    """Get the path to the documents directory."""
    return PROJECT_ROOT / "documents"


def get_logs_dir() -> Path:
    """Get the path to the logs directory, creating it if needed."""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def format_docs_for_context(docs: list) -> str:
    """
    Format a list of LangChain Document objects into a context string
    for the reasoning prompt.
    
    Args:
        docs: List of Document objects with page_content and metadata.
    
    Returns:
        Formatted string with document content and source info.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        source_name = Path(source).name if source else "Unknown"
        section = doc.metadata.get("section", "")
        
        header = f"[Document {i}: {source_name}"
        if section:
            header += f", Section: {section}"
        header += "]"
        
        formatted.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)


def format_sources_for_citation(docs: list) -> str:
    """
    Format source documents for the citation prompt.
    
    Args:
        docs: List of Document objects with metadata.
    
    Returns:
        Formatted string listing source documents with metadata.
    """
    metadata_map = load_document_metadata()
    sources = []
    
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        source_name = Path(source).name if source else "Unknown"
        
        if source_name in seen:
            continue
        seen.add(source_name)
        
        meta = metadata_map.get(source_name, {})
        department = meta.get("department", "Unknown")
        description = meta.get("description", "")
        
        entry = f"- Document: {source_name}, Department: {department}"
        if description:
            entry += f", Description: {description}"
        sources.append(entry)
    
    return "\n".join(sources) if sources else "No sources available."
