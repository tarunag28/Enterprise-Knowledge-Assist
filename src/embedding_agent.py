"""
Embedding Agent
Converts document chunks into vector representations and stores them
in a persistent ChromaDB vector store using Ollama embeddings.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config_loader import get_embedding_settings, get_vectorstore_settings


def get_embeddings() -> OllamaEmbeddings:
    """
    Initialize the Ollama embedding model from config.
    
    Returns:
        Configured OllamaEmbeddings instance.
    """
    settings = get_embedding_settings()
    return OllamaEmbeddings(
        model=settings.get("model", "nomic-embed-text"),
        base_url=settings.get("base_url"),
    )


def get_vectorstore(embeddings: OllamaEmbeddings = None) -> Chroma:
    """
    Get or create the ChromaDB vector store.
    
    Args:
        embeddings: Optional pre-initialized embeddings. Creates new if None.
    
    Returns:
        Chroma vector store instance.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    
    vs_settings = get_vectorstore_settings()
    
    return Chroma(
        collection_name=vs_settings.get("collection_name", "enterprise_docs"),
        embedding_function=embeddings,
        persist_directory=vs_settings.get("persist_directory", "./chroma_db"),
    )


def index_documents(chunks: list[Document], batch_size: int = 50) -> Chroma:
    """
    Embed and index document chunks into the vector store.
    
    Processes in batches to handle large document sets without
    overwhelming the embedding model.
    
    Args:
        chunks: List of Document objects to embed and store.
        batch_size: Number of documents to process per batch.
    
    Returns:
        The populated Chroma vector store.
    """
    embeddings = get_embeddings()
    vs_settings = get_vectorstore_settings()
    
    print(f"\n🧮 Embedding {len(chunks)} chunks using '{get_embedding_settings()['model']}'...")
    print(f"   Vector store: {vs_settings['persist_directory']}")
    print(f"   Collection: {vs_settings['collection_name']}")
    
    # Process in batches
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        if vectorstore is None:
            # First batch — create the vector store
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=vs_settings.get("collection_name", "enterprise_docs"),
                persist_directory=vs_settings.get("persist_directory", "./chroma_db"),
            )
        else:
            # Subsequent batches — add to existing store
            vectorstore.add_documents(batch)
    
    if vectorstore is None:
        # No chunks to index, return empty store
        vectorstore = get_vectorstore(embeddings)
    
    doc_count = vectorstore._collection.count()
    print(f"   ✅ Vector store now contains {doc_count} documents")
    
    return vectorstore


def clear_vectorstore():
    """
    Clear all documents from the vector store.
    Useful for re-indexing from scratch.
    """
    vs_settings = get_vectorstore_settings()
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        collection_name=vs_settings.get("collection_name", "enterprise_docs"),
        embedding_function=embeddings,
        persist_directory=vs_settings.get("persist_directory", "./chroma_db"),
    )
    
    # Delete all documents
    collection = vectorstore._collection
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)
        print(f"🗑️  Cleared {len(all_ids)} documents from vector store")
    else:
        print("ℹ️  Vector store is already empty")
