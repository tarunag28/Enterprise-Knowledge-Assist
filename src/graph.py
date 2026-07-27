"""
LangGraph Multi-Agent Workflow
Orchestrates the RAG pipeline: Retrieve → Grade → Reason → Cite.
Uses a StateGraph with conditional edges for intelligent routing.
"""

import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.retrieval_agent import retrieve_documents, grade_document_relevance
from src.reasoning_agent import generate_answer, generate_no_context_response
from src.citation_agent import add_citations
from src.audit_logger import audit_logger


# --- State Schema ---

class GraphState(TypedDict):
    """Shared state across all nodes in the workflow."""
    query: str
    retrieved_docs: list
    relevant_docs: list
    relevance_score: float
    raw_answer: str
    cited_answer: str
    citations: list
    audit_trail: list
    start_time: float
    status: str


# --- Node Functions ---

def retrieve_node(state: GraphState) -> dict:
    """Retrieve relevant documents from the vector store."""
    query = state["query"]
    audit = state.get("audit_trail", [])
    audit.append({"step": "retrieve", "status": "started"})

    docs = retrieve_documents(query)

    audit.append({
        "step": "retrieve",
        "status": "completed",
        "num_docs": len(docs),
    })

    return {
        "retrieved_docs": docs,
        "audit_trail": audit,
    }


def grade_node(state: GraphState) -> dict:
    """Grade relevance of retrieved documents using a hybrid approach.
    
    Documents with high similarity scores are auto-accepted.
    Borderline documents are graded by the LLM.
    This prevents false negatives from small local LLMs being too strict.
    """
    query = state["query"]
    docs = state["retrieved_docs"]
    audit = state.get("audit_trail", [])
    audit.append({"step": "grade", "status": "started"})

    # Similarity score threshold: docs above this are auto-accepted
    # ChromaDB similarity_score in metadata was computed as (1 - distance)
    AUTO_ACCEPT_THRESHOLD = 0.25
    LLM_GRADE_THRESHOLD = 0.10  # Below this, skip entirely

    relevant = []
    for doc in docs:
        sim_score = doc.metadata.get("similarity_score", 0.5)

        if sim_score >= AUTO_ACCEPT_THRESHOLD:
            # High similarity — auto-accept without LLM call
            relevant.append(doc)
        elif sim_score >= LLM_GRADE_THRESHOLD:
            # Borderline — use LLM to decide
            if grade_document_relevance(query, doc):
                relevant.append(doc)
        # Below LLM_GRADE_THRESHOLD — skip (too dissimilar)

    # If no docs passed grading but we have retrieved docs, 
    # fall back to using top 2 retrieved docs (trust the vector search)
    if not relevant and docs:
        relevant = docs[:2]

    score = len(relevant) / len(docs) if docs else 0.0

    audit.append({
        "step": "grade",
        "status": "completed",
        "relevant": len(relevant),
        "total": len(docs),
        "score": round(score, 3),
    })

    return {
        "relevant_docs": relevant,
        "relevance_score": score,
        "audit_trail": audit,
    }


def reason_node(state: GraphState) -> dict:
    """Generate an answer from relevant documents."""
    query = state["query"]
    docs = state["relevant_docs"]
    audit = state.get("audit_trail", [])
    audit.append({"step": "reason", "status": "started"})

    answer = generate_answer(query, docs)

    audit.append({
        "step": "reason",
        "status": "completed",
        "answer_length": len(answer),
    })

    return {
        "raw_answer": answer,
        "audit_trail": audit,
    }


def cite_node(state: GraphState) -> dict:
    """Add citations to the generated answer."""
    answer = state["raw_answer"]
    docs = state["relevant_docs"]
    audit = state.get("audit_trail", [])
    audit.append({"step": "cite", "status": "started"})

    cited_answer, citations = add_citations(answer, docs)
    duration = time.time() - state.get("start_time", time.time())

    audit.append({
        "step": "cite",
        "status": "completed",
        "num_citations": len(citations),
    })

    # Log the complete query
    audit_logger.log_query(
        query=state["query"],
        retrieved_docs=docs,
        raw_answer=answer,
        cited_answer=cited_answer,
        citations=citations,
        relevance_score=state.get("relevance_score", 0.0),
        duration_seconds=duration,
    )

    return {
        "cited_answer": cited_answer,
        "citations": citations,
        "audit_trail": audit,
        "status": "success",
    }


def no_context_node(state: GraphState) -> dict:
    """Handle case where no relevant documents were found."""
    query = state["query"]
    audit = state.get("audit_trail", [])
    duration = time.time() - state.get("start_time", time.time())

    answer = generate_no_context_response(query)

    audit.append({"step": "no_context", "status": "completed"})

    audit_logger.log_query(
        query=query,
        retrieved_docs=[],
        raw_answer=answer,
        cited_answer=answer,
        citations=[],
        relevance_score=0.0,
        duration_seconds=duration,
        status="no_context",
    )

    return {
        "raw_answer": answer,
        "cited_answer": answer,
        "citations": [],
        "audit_trail": audit,
        "status": "no_context",
    }


# --- Conditional Edge ---

def should_continue(state: GraphState) -> str:
    """Decide whether to proceed to reasoning or return no-context."""
    relevant_docs = state.get("relevant_docs", [])
    if relevant_docs:
        return "reason"
    return "no_context"


# --- Build the Graph ---

def build_graph() -> StateGraph:
    """Build and compile the LangGraph multi-agent workflow."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("reason", reason_node)
    workflow.add_node("cite", cite_node)
    workflow.add_node("no_context", no_context_node)

    # Define edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        should_continue,
        {"reason": "reason", "no_context": "no_context"},
    )
    workflow.add_edge("reason", "cite")
    workflow.add_edge("cite", END)
    workflow.add_edge("no_context", END)

    return workflow.compile()


# Compile the graph once
app_graph = build_graph()


def run_query(query: str) -> dict:
    """
    Execute the full RAG pipeline for a user query.

    Args:
        query: The user's natural language question.

    Returns:
        Dictionary with cited_answer, citations, status, and audit_trail.
    """
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "relevant_docs": [],
        "relevance_score": 0.0,
        "raw_answer": "",
        "cited_answer": "",
        "citations": [],
        "audit_trail": [],
        "start_time": time.time(),
        "status": "pending",
    }

    try:
        result = app_graph.invoke(initial_state)
        return {
            "query": query,
            "answer": result.get("cited_answer", ""),
            "citations": result.get("citations", []),
            "relevance_score": result.get("relevance_score", 0.0),
            "status": result.get("status", "unknown"),
            "audit_trail": result.get("audit_trail", []),
        }
    except Exception as e:
        duration = time.time() - initial_state["start_time"]
        audit_logger.log_error(query, str(e), duration)
        return {
            "query": query,
            "answer": f"An error occurred while processing your query: {str(e)}",
            "citations": [],
            "relevance_score": 0.0,
            "status": "error",
            "audit_trail": [],
        }
