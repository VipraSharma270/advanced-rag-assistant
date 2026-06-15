import streamlit as st

from config import DATA_DIR, LLM_MODEL, RETRIEVAL_K, TOP_K_CONTEXT, VECTOR_DB_DIR
from rag_pipeline import ask

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">Advanced RAG Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Semantic search · Cross-encoder reranking · Local LLM generation</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        f"""
        **1. Retrieve** — top-{RETRIEVAL_K} chunks via BGE embeddings  
        **2. Rerank** — cross-encoder precision scoring  
        **3. Generate** — `{LLM_MODEL}` via Ollama
        """
    )
    st.divider()
    st.caption(f"📁 Documents: `{DATA_DIR.name}/`")
    st.caption(f"🗄️ Vector store: `{VECTOR_DB_DIR.name}/`")

query = st.text_input("Ask a question about your documents", placeholder="What is Dense Passage Retrieval?")

if st.button("Search", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving, reranking, and generating..."):
            result = ask(query)

        col_answer, col_sources = st.columns([3, 2])

        with col_answer:
            st.subheader("Answer")
            st.markdown(result["answer"])

        with col_sources:
            st.subheader("Sources")
            for i, doc in enumerate(result["sources"], start=1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "—")
                preview = doc.page_content[:280].strip()
                if len(doc.page_content) > 280:
                    preview += "…"

                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>#{i}</strong> &nbsp; {source}<br>
                        <small>Page {page}</small><br><br>
                        <span style="color:#475569">{preview}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
