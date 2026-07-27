"""
Retrieval Agent
Fetches relevant document chunks from the vector store using similarity search,
then grades their relevance using similarity scores and LLM verification.
"""

from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from src.config_loader import get_retrieval_settings, get_llm_settings
from src.embedding_agent import get_vectorstore
from src.utils import load_prompt


def get_retriever():
    """
    Create a retriever from the vector store with config-driven settings.
    
    Returns:
        A LangChain retriever instance.
    """
    retrieval_config = get_retrieval_settings()
    vectorstore = get_vectorstore()
    
    return vectorstore.as_retriever(
        search_type=retrieval_config.get("search_type", "similarity"),
        search_kwargs={
            "k": retrieval_config.get("top_k", 4),
        },
    )


def retrieve_documents(query: str) -> list[Document]:
    """
    Retrieve relevant document chunks for a given query
    using similarity search with scores for filtering.
    
    Args:
        query: The user's natural language question.
    
    Returns:
        List of relevant Document objects with metadata.
    """
    retrieval_config = get_retrieval_settings()
    vectorstore = get_vectorstore()
    top_k = retrieval_config.get("top_k", 4)
    
    # Use similarity_search_with_score for score-based filtering
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    docs = []
    for doc, score in results:
        # ChromaDB returns distance (lower = better); store it in metadata
        doc.metadata["similarity_score"] = round(1.0 - score, 4) if score <= 1.0 else round(score, 4)
        docs.append(doc)
    
    return docs


def grade_document_relevance(query: str, document: Document) -> bool:
    """
    Use the LLM to grade whether a document chunk is relevant to the query.
    Falls back to True if the LLM response is ambiguous (to avoid
    false negatives with smaller local models).
    
    Args:
        query: The user's question.
        document: A Document object to evaluate.
    
    Returns:
        True if the document is relevant, False otherwise.
    """
    llm_settings = get_llm_settings()
    llm = ChatOllama(
        model=llm_settings.get("model", "llama3.2"),
        base_url=llm_settings.get("base_url"),
        temperature=0,
    )
    
    prompt_template = load_prompt("retrieval_prompt.txt")
    prompt = prompt_template.format(
        query=query,
        document=document.page_content,
    )
    
    try:
        response = llm.invoke(prompt)
        result = response.content.strip().lower()
        # Be lenient — only reject if the LLM explicitly says "no"
        return "no" not in result.split()[0] if result else True
    except Exception:
        # If grading fails, assume relevant (don't drop good results)
        return True


def retrieve_and_grade(query: str) -> tuple[list[Document], float]:
    """
    Retrieve documents and grade their relevance.
    Returns only relevant documents and a relevance score.
    
    Args:
        query: The user's question.
    
    Returns:
        Tuple of (relevant_documents, relevance_score).
        relevance_score is the fraction of retrieved docs deemed relevant.
    """
    docs = retrieve_documents(query)
    
    if not docs:
        return [], 0.0
    
    relevant_docs = []
    for doc in docs:
        if grade_document_relevance(query, doc):
            relevant_docs.append(doc)
    
    relevance_score = len(relevant_docs) / len(docs) if docs else 0.0
    
    return relevant_docs, relevance_score
