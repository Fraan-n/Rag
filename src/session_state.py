"""Gestión del estado de sesión de Streamlit."""

import streamlit as st
import uuid


def init_session_state(cfg) -> None:
    """Inicializa las claves del estado de sesión si aún no existen."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False
    if "collection" not in st.session_state:
        st.session_state.collection = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
    if "document_embeddings" not in st.session_state:
        st.session_state.document_embeddings = []