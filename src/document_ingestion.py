"""
Document Ingestion Agent
Loads enterprise documents (PDF, DOCX, TXT), splits them into chunks,
and attaches metadata for downstream embedding and retrieval.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config_loader import get_chunking_settings
from src.utils import get_documents_dir, load_document_metadata


# Map file extensions to their LangChain loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def get_loader_for_file(filepath: str):
    """
    Select the appropriate document loader based on file extension.
    
    Args:
        filepath: Path to the document file.
    
    Returns:
        An instance of the appropriate LangChain document loader.
    
    Raises:
        ValueError: If the file type is not supported.
    """
    ext = Path(filepath).suffix.lower()
    loader_class = LOADER_MAP.get(ext)
    if loader_class is None:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported types: {', '.join(LOADER_MAP.keys())}"
        )
    return loader_class(filepath)


def load_single_document(filepath: str) -> list[Document]:
    """
    Load a single document file and return raw Document objects.
    
    Args:
        filepath: Path to the document file.
    
    Returns:
        List of Document objects (one per page for PDFs, one for TXT/DOCX).
    """
    loader = get_loader_for_file(filepath)
    docs = loader.load()
    
    # Enrich metadata with filename
    filename = Path(filepath).name
    metadata_map = load_document_metadata()
    file_meta = metadata_map.get(filename, {})
    
    for doc in docs:
        doc.metadata["source"] = filepath
        doc.metadata["filename"] = filename
        doc.metadata.update(file_meta)
    
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks for embedding.
    Uses config-driven chunk size and overlap.
    
    Args:
        documents: List of Document objects to chunk.
    
    Returns:
        List of chunked Document objects with preserved metadata.
    """
    chunking_config = get_chunking_settings()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_config.get("chunk_size", 1000),
        chunk_overlap=chunking_config.get("chunk_overlap", 200),
        separators=chunking_config.get("separators", ["\n\n", "\n", ". ", " "]),
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    return chunks


def ingest_directory(
    directory: Optional[str] = None,
    file_types: Optional[list[str]] = None,
) -> list[Document]:
    """
    Ingest all supported documents from a directory.
    
    Args:
        directory: Path to the documents directory. Defaults to documents/.
        file_types: List of file extensions to process (e.g., ['.pdf', '.txt']).
                    Defaults to all supported types.
    
    Returns:
        List of chunked Document objects ready for embedding.
    """
    if directory is None:
        directory = str(get_documents_dir())
    
    if file_types is None:
        file_types = list(LOADER_MAP.keys())
    
    all_docs = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {directory}")
    
    # Collect all matching files
    files = []
    for ext in file_types:
        files.extend(dir_path.glob(f"*{ext}"))
    
    if not files:
        print(f"⚠️  No supported documents found in {directory}")
        return []
    
    print(f"📁 Found {len(files)} document(s) to ingest:")
    
    for filepath in sorted(files):
        print(f"   📄 Loading: {filepath.name}")
        try:
            docs = load_single_document(str(filepath))
            all_docs.extend(docs)
            print(f"      ✅ Loaded {len(docs)} page(s)")
        except Exception as e:
            print(f"      ❌ Error loading {filepath.name}: {e}")
    
    # Chunk all documents
    print(f"\n✂️  Chunking {len(all_docs)} page(s)...")
    chunks = chunk_documents(all_docs)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    return chunks


if __name__ == "__main__":
    # Quick test
    chunks = ingest_directory()
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk.metadata.get('filename', 'unknown')}")
        print(f"Content: {chunk.page_content[:200]}...")
