"""
Citation Agent
Maps response claims to source documents and formats citations.
Ensures every answer has traceable, verifiable source references.
"""

from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from src.config_loader import get_llm_settings
from src.utils import load_prompt, format_sources_for_citation, load_document_metadata


def add_citations(answer: str, documents: list[Document]) -> tuple[str, list[dict]]:
    """Add source citations to the generated answer using LLM."""
    llm_settings = get_llm_settings()
    llm = ChatOllama(
        model=llm_settings.get("model", "llama3.2"),
        base_url=llm_settings.get("base_url"),
        temperature=0,
    )
    sources_text = format_sources_for_citation(documents)
    prompt_template = load_prompt("citation_prompt.txt")
    prompt = prompt_template.format(answer=answer, sources=sources_text)
    response = llm.invoke(prompt)
    cited_answer = response.content.strip()
    citations = build_citations_list(documents)
    return cited_answer, citations


def build_citations_list(documents: list[Document]) -> list[dict]:
    """Build a structured list of citations from the source documents."""
    metadata_map = load_document_metadata()
    citations = []
    seen_sources = set()
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        source_name = Path(source).name if source else "Unknown"
        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)
        meta = metadata_map.get(source_name, {})
        citation = {
            "document": source_name,
            "department": meta.get("department", doc.metadata.get("department", "Unknown")),
            "owner": meta.get("owner", doc.metadata.get("owner", "Unknown")),
            "access_level": meta.get("access_level", "all_employees"),
            "description": meta.get("description", ""),
            "chunk_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
        }
        citations.append(citation)
    return citations
