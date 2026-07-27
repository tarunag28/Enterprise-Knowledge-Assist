#!/usr/bin/env python3
"""
Document Ingestion CLI
Ingests all documents from the documents/ directory into the ChromaDB vector store.

Usage:
    python ingest.py              # Ingest all documents
    python ingest.py --clear      # Clear vector store and re-ingest
"""

import sys
import time
import argparse

from src.document_ingestion import ingest_directory
from src.embedding_agent import index_documents, clear_vectorstore
from src.audit_logger import audit_logger


def main():
    parser = argparse.ArgumentParser(
        description="Ingest enterprise documents into the vector store"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store before ingesting",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Custom documents directory (defaults to ./documents/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Enterprise Knowledge Assistant — Document Ingestion")
    print("=" * 60)

    start_time = time.time()

    # Optionally clear existing data
    if args.clear:
        print("\n🗑️  Clearing existing vector store...")
        clear_vectorstore()

    # Step 1: Load and chunk documents
    print("\n📥 Step 1: Loading and chunking documents...\n")
    chunks = ingest_directory(directory=args.directory)

    if not chunks:
        print("\n❌ No documents to ingest. Exiting.")
        sys.exit(1)

    # Step 2: Embed and index
    print("\n📥 Step 2: Embedding and indexing chunks...\n")
    vectorstore = index_documents(chunks)

    duration = time.time() - start_time

    # Log the ingestion event
    audit_logger.log_ingestion(
        num_files=len(set(c.metadata.get("filename", "") for c in chunks)),
        num_chunks=len(chunks),
        duration_seconds=duration,
    )

    print("\n" + "=" * 60)
    print(f"  ✅ Ingestion complete!")
    print(f"  📄 Documents processed: {len(set(c.metadata.get('filename', '') for c in chunks))}")
    print(f"  🧩 Total chunks created: {len(chunks)}")
    print(f"  ⏱️  Duration: {duration:.1f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
