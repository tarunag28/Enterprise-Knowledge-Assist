"""
Audit Logger
Tracks user queries, retrieved sources, generated responses, and citations.
Writes structured JSON logs for compliance and debugging.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from src.utils import get_logs_dir


class AuditLogger:
    """Structured audit logger that writes JSONL entries."""

    def __init__(self):
        self.log_file = get_logs_dir() / "audit.jsonl"

    def log_query(
        self,
        query: str,
        retrieved_docs: list,
        raw_answer: str,
        cited_answer: str,
        citations: list,
        relevance_score: float,
        duration_seconds: float,
        status: str = "success",
    ):
        """Log a complete query-response cycle."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "query": query,
            "relevance_score": round(relevance_score, 3),
            "num_docs_retrieved": len(retrieved_docs),
            "sources": [
                {
                    "filename": doc.metadata.get("filename", "unknown"),
                    "department": doc.metadata.get("department", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                }
                for doc in retrieved_docs
            ],
            "raw_answer_length": len(raw_answer),
            "cited_answer_length": len(cited_answer),
            "num_citations": len(citations),
            "duration_seconds": round(duration_seconds, 2),
        }
        self._write_entry(entry)

    def log_error(self, query: str, error: str, duration_seconds: float):
        """Log a failed query."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "query": query,
            "error": str(error),
            "duration_seconds": round(duration_seconds, 2),
        }
        self._write_entry(entry)

    def log_ingestion(self, num_files: int, num_chunks: int, duration_seconds: float):
        """Log a document ingestion event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "ingestion",
            "num_files": num_files,
            "num_chunks": num_chunks,
            "duration_seconds": round(duration_seconds, 2),
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict):
        """Append a JSON entry to the log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent_logs(self, n: int = 20) -> list[dict]:
        """Read the most recent n log entries."""
        if not self.log_file.exists():
            return []
        with open(self.log_file, "r") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries


# Singleton instance
audit_logger = AuditLogger()
