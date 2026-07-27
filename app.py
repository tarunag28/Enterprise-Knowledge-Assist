"""
University Knowledge Hub — Streamlit UI
A polished chat interface for querying university documents with citations.
"""

import os
import sys
import time
import streamlit as st
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import run_query
from src.document_ingestion import ingest_directory
from src.embedding_agent import index_documents, clear_vectorstore, get_vectorstore
from src.audit_logger import audit_logger
from src.config_loader import get_llm_settings, get_embedding_settings

# --- Page Configuration ---
st.set_page_config(
    page_title="University Knowledge Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS — Light Mode Corporate Blue ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main App Background — Light Mode */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 50%, #f8fafc 100%);
        color: #1e293b;
    }

    /* Main Header — Light Glassmorphism */
    .main-header {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 4px 24px -4px rgba(30, 58, 95, 0.10), 0 1px 3px rgba(30, 58, 95, 0.06);
    }
    .main-header h1 {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #64748b;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.90) !important;
        border: 1px solid rgba(59, 130, 246, 0.08) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 8px -2px rgba(30, 58, 95, 0.08), 0 1px 2px rgba(30, 58, 95, 0.04) !important;
    }

    /* Chat Input */
    .stChatInputContainer {
        border-radius: 20px !important;
        border: 1px solid rgba(59, 130, 246, 0.25) !important;
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Citation Cards */
    .citation-card {
        background: rgba(248, 250, 252, 0.95);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
    }
    .citation-card:hover {
        background: rgba(241, 245, 249, 1);
        border-color: #3b82f6;
        transform: translateX(4px);
        box-shadow: 0 4px 12px -4px rgba(59, 130, 246, 0.15);
    }
    .citation-doc {
        color: #1e293b;
        font-weight: 600;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .citation-dept {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    .citation-preview {
        color: #475569;
        font-size: 0.9rem;
        margin-top: 0.75rem;
        line-height: 1.5;
        font-style: italic;
        padding-left: 1rem;
        border-left: 2px solid rgba(59, 130, 246, 0.25);
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .status-success {
        background: rgba(16, 185, 129, 0.08);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.20);
    }
    .status-no-context {
        background: rgba(245, 158, 11, 0.08);
        color: #d97706;
        border: 1px solid rgba(245, 158, 11, 0.20);
    }
    .status-error {
        background: rgba(239, 68, 68, 0.08);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.20);
    }

    /* Sidebar — Light Mode */
    [data-testid="stSidebar"] {
        background-color: rgba(241, 245, 249, 0.98) !important;
        border-right: 1px solid rgba(59, 130, 246, 0.10);
    }
    .sidebar-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.10);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px -2px rgba(30, 58, 95, 0.06);
    }
    .sidebar-section h3 {
        color: #1e3a5f;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(59, 130, 246, 0.20) !important;
        border-radius: 12px !important;
        color: #1e3a5f !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: rgba(59, 130, 246, 0.08) !important;
        border-color: #3b82f6 !important;
        color: #1e3a5f !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }

    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px dashed rgba(59, 130, 246, 0.30) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3b82f6 !important;
        background: rgba(59, 130, 246, 0.04) !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stHeader"] button {
        visibility: visible !important;
        color: #1e3a5f !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h3>🎓 University Knowledge Hub</h3>
        <p style="color: #64748b; font-size: 0.85rem;">
            Multi-Agent RAG System powered by local LLM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Info
    llm_settings = get_llm_settings()
    emb_settings = get_embedding_settings()
    
    st.markdown("### 🔧 System Configuration")
    st.markdown(f"**LLM Model:** `{llm_settings.get('model', 'N/A')}`")
    st.markdown(f"**Embeddings:** `{emb_settings.get('model', 'N/A')}`")
    st.markdown(f"**Server:** `{llm_settings.get('base_url', 'N/A')}`")
    
    st.divider()
    
    # Vector Store Stats
    st.markdown("### 📈 Vector Store")
    try:
        vs = get_vectorstore()
        doc_count = vs._collection.count()
        st.metric("Indexed Documents", doc_count)
    except Exception:
        doc_count = 0
        st.metric("Indexed Documents", 0)
    
    st.divider()
    
    # Document Upload
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload university documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
    )
    
    if uploaded_files:
        if st.button("⬇️ Ingest Uploaded Documents", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                # Save uploaded files to documents directory
                docs_dir = PROJECT_ROOT / "documents"
                docs_dir.mkdir(exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    filepath = docs_dir / uploaded_file.name
                    with open(filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"✅ Saved: {uploaded_file.name}")
                
                # Run ingestion pipeline
                try:
                    start = time.time()
                    clear_vectorstore()
                    chunks = ingest_directory()
                    if chunks:
                        index_documents(chunks)
                        duration = time.time() - start
                        st.success(f"✅ Ingested {len(chunks)} chunks in {duration:.1f}s")
                        st.rerun()
                    else:
                        st.warning("No documents found to ingest.")
                except Exception as e:
                    st.error(f"❌ Ingestion error: {str(e)}")
    
    # Re-index button
    if doc_count > 0:
        st.divider()
        if st.button("♻️ Re-index All Documents", use_container_width=True):
            with st.spinner("Re-indexing..."):
                try:
                    start = time.time()
                    clear_vectorstore()
                    chunks = ingest_directory()
                    if chunks:
                        index_documents(chunks)
                        duration = time.time() - start
                        st.success(f"✅ Re-indexed {len(chunks)} chunks in {duration:.1f}s")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Audit Log Viewer
    st.markdown("### 🕐 Recent Queries")
    logs = audit_logger.get_recent_logs(10)
    if logs:
        for log in reversed(logs):
            if log.get("event") == "ingestion":
                continue
            status = log.get("status", "unknown")
            query = log.get("query", "N/A")
            if len(query) > 50:
                query = query[:50] + "..."
            icon = "✅" if status == "success" else "⚠️" if status == "no_context" else "❌"
            st.markdown(f"{icon} {query}")
    else:
        st.markdown("*No queries yet.*")


# --- Main Content ---
st.markdown("""
<div class="main-header">
    <h1>🎓 University Knowledge Hub</h1>
    <p>Ask questions about academics, campus IT, student services, and more. 
    All answers are grounded in university documents with full source citations.</p>
</div>
""", unsafe_allow_html=True)

# Pipeline status indicator
if doc_count == 0:
    st.warning(
        "⚠️ **No documents indexed yet.** Upload documents in the sidebar "
        "or run `python ingest.py` to index the sample documents."
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show citations for assistant messages
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander(f"🔗 Sources ({len(message['citations'])} documents)", expanded=False):
                for citation in message["citations"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-doc">📑 {citation.get('document', 'Unknown')}</div>
                        <div class="citation-dept">🏛️ {citation.get('department', 'Unknown')} · 
                        ✍️ {citation.get('owner', 'Unknown')}</div>
                        <div class="citation-preview">{citation.get('chunk_preview', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show status badge
            status = message.get("status", "unknown")
            score = message.get("relevance_score", 0)
            if status == "success":
                st.markdown(f"""
                <span class="status-badge status-success">✓ Grounded · Relevance: {score:.0%}</span>
                """, unsafe_allow_html=True)
            elif status == "no_context":
                st.markdown("""
                <span class="status-badge status-no-context">⚠ No relevant documents found</span>
                """, unsafe_allow_html=True)


# Chat input
prompt = st.chat_input("Ask a question about university policies...", key="chat_input")

# --- Sample Questions (shown when chat is empty) ---
if not st.session_state.messages and not prompt:
    st.markdown("### ✨ Try asking:")
    
    cols = st.columns(2)
    sample_questions = [
        "What is the university's grading scale?",
        "How do I connect to campus WiFi?",
        "What are the scholarship requirements?",
        "What happens during orientation week?",
        "How do I report a campus IT issue?",
        "What are the graduation requirements?",
    ]
    
    for i, question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"▶️ {question}", key=f"sample_{i}", use_container_width=True):
                prompt = question

# Process the prompt (whether from chat_input or a sample button)
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("⏳ Searching documents..."):
            result = run_query(prompt)
        
        answer = result.get("answer", "Sorry, I encountered an error.")
        citations = result.get("citations", [])
        status = result.get("status", "unknown")
        score = result.get("relevance_score", 0)
        
        st.markdown(answer)
        
        # Show citations
        if citations:
            with st.expander(f"🔗 Sources ({len(citations)} documents)", expanded=True):
                for citation in citations:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-doc">📑 {citation.get('document', 'Unknown')}</div>
                        <div class="citation-dept">🏛️ {citation.get('department', 'Unknown')} · 
                        ✍️ {citation.get('owner', 'Unknown')}</div>
                        <div class="citation-preview">{citation.get('chunk_preview', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Status badge
        if status == "success":
            st.markdown(f"""
            <span class="status-badge status-success">✓ Grounded · Relevance: {score:.0%}</span>
            """, unsafe_allow_html=True)
        elif status == "no_context":
            st.markdown("""
            <span class="status-badge status-no-context">⚠ No relevant documents found</span>
            """, unsafe_allow_html=True)
        elif status == "error":
            st.markdown("""
            <span class="status-badge status-error">✗ Error occurred</span>
            """, unsafe_allow_html=True)
    
    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "status": status,
        "relevance_score": score,
    })
    
    # Force a rerun to update the UI cleanly (hides sample buttons)
    st.rerun()
