"""
Reasoning Agent
Synthesizes answers from retrieved context using the LLM.
Answers are strictly grounded in the provided documents — no hallucination.
"""

from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from src.config_loader import get_llm_settings
from src.utils import load_prompt, format_docs_for_context


def generate_answer(query: str, documents: list[Document]) -> str:
    """
    Generate an answer to the user's query using retrieved document context.
    
    The answer is grounded strictly in the provided documents. If the context
    is insufficient, the LLM will explicitly state so.
    
    Args:
        query: The user's natural language question.
        documents: List of relevant Document objects providing context.
    
    Returns:
        The generated answer string.
    """
    llm_settings = get_llm_settings()
    llm = ChatOllama(
        model=llm_settings.get("model", "llama3.2"),
        base_url=llm_settings.get("base_url"),
        temperature=llm_settings.get("temperature", 0),
        num_predict=llm_settings.get("num_predict", 1024),
    )
    
    # Format context from documents
    context = format_docs_for_context(documents)
    
    # Load and populate prompt template
    prompt_template = load_prompt("reasoning_prompt.txt")
    prompt = prompt_template.format(
        query=query,
        context=context,
    )
    
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_no_context_response(query: str) -> str:
    """
    Generate a response when no relevant documents are found.
    
    Args:
        query: The user's question.
    
    Returns:
        A helpful "no information available" response.
    """
    return (
        f"I apologize, but I could not find any relevant information in the enterprise "
        f"documents to answer your question: \"{query}\"\n\n"
        f"This could mean:\n"
        f"• The topic is not covered in the currently indexed documents\n"
        f"• The question might need to be rephrased for better matching\n"
        f"• Additional documents may need to be uploaded to cover this topic\n\n"
        f"Please try rephrasing your question or contact the relevant department directly."
    )
