import os

import streamlit as st
from dotenv import load_dotenv

from src.config import load_config
from src.document_loader import load_uploaded_file
from src.chunking import create_chunks
from src.embeddings import configure, get_embeddings
from src.vectorstore import (
    get_or_create_collection,
    add_documents_to_collection,
    search_similar,
    delete_collection,
    get_unique_filenames,
    remove_documents_by_filename,
)
from src.rag_chain import generate_answer
from src.session_state import init_session_state


def get_api_key() -> str | None:
    """Obtiene la API key de variable de entorno o de Streamlit Secrets."""
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="RAG con Gemini", page_icon="📄", layout="wide")
    load_dotenv()
    cfg = load_config()
    init_session_state(cfg)

    api_key = get_api_key()
    if not api_key:
        st.error(
            "Falta la API key de Google Gemini. "
            "Configura GOOGLE_API_KEY en un archivo .env (local) o en los Secrets de Streamlit Cloud."
        )
        st.stop()

    configure(api_key)

    st.title("📄 Asistente RAG con Gemini")
    st.caption(
        "Sube PDF, DOCX o TXT y haz preguntas sobre su contenido. "
        "Tus documentos son privados de esta sesión de navegador."
    )

    # ------------------- Sidebar: subida y gestión de documentos -------------------
    with st.sidebar:
        st.header("📁 Documentos")

        uploaded_files = st.file_uploader(
            "Seleccionar archivos",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Máximo 10 MB por archivo. Formatos: PDF, DOCX, TXT.",
        )

        if uploaded_files:
            oversized = [
                f
                for f in uploaded_files
                if (getattr(f, "size", 0) or 0)
                > cfg.max_file_size_mb * 1024 * 1024
            ]
            if oversized:
                for f in oversized:
                    st.error(
                        f"{f.name} supera los {cfg.max_file_size_mb} MB permitidos."
                    )
                uploaded_files = [f for f in uploaded_files if f not in oversized]

        # Determinar qué archivos ya fueron procesados para no repetir trabajo
        already_processed = {
            (pf["name"], pf.get("size", 0)) for pf in st.session_state.processed_files
        }
        new_files = [
            f
            for f in uploaded_files
            if (f.name, getattr(f, "size", 0) or 0) not in already_processed
        ]

        if st.button(
            "Procesar archivos seleccionados",
            disabled=not new_files,
            type="primary",
            help="Solo se procesarán los archivos que aún no hayan sido indexados.",
        ):
            _process_files(new_files, cfg)

        st.divider()
        st.subheader("📄 Documentos cargados en esta sesión")

        filenames_in_collection = []
        if st.session_state.vector_store_ready and st.session_state.collection is not None:
            filenames_in_collection = get_unique_filenames(
                st.session_state.collection
            )

        if not filenames_in_collection:
            st.info("No hay documentos cargados. Sube uno o más archivos y procesalos.")
        else:
            for filename in filenames_in_collection:
                col1, col2 = st.columns([4, 1])
                col1.write(f"📄 {filename}")
                if col2.button("Eliminar", key=f"del_{filename}"):
                    _remove_document(filename, cfg)

        if st.button("Limpiar sesión", type="secondary"):
            _clear_session(cfg)

    st.divider()

    # ------------------- Historial del chat -------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("🔗 Fuentes"):
                    for source in message["sources"]:
                        st.write(
                            f"- {source.get('filename')} (página {source.get('page')})"
                        )

    # ------------------- Entrada del usuario -------------------
    if prompt := st.chat_input("Escribe tu pregunta sobre los documentos..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        if (
            not st.session_state.vector_store_ready
            or st.session_state.collection is None
        ):
            warning = (
                "Primero sube y procesa al menos un documento. "
                "Si acabas de recargar la página, vuelve a seleccionar y procesar tus archivos."
            )
            with st.chat_message("assistant"):
                st.warning(warning)
            st.session_state.messages.append(
                {"role": "assistant", "content": warning, "sources": []}
            )
        else:
            with st.chat_message("assistant"):
                with st.spinner("Buscando fragmentos y generando respuesta..."):
                    query_embedding = get_embeddings([prompt], cfg.embedding_model)[0]
                    results = search_similar(
                        st.session_state.collection,
                        query_embedding,
                        top_k=cfg.top_k,
                    )

                    if not results:
                        answer = (
                            "No encontré información suficiente en los documentos "
                            "subidos para responder a tu pregunta."
                        )
                        sources = []
                    else:
                        contexts = [r["document"] for r in results]
                        sources = [r["metadata"] for r in results]
                        try:
                            answer = generate_answer(
                                prompt,
                                contexts,
                                sources,
                                model_name=cfg.generation_model,
                                api_key=api_key,
                            )
                        except Exception as exc:  # noqa: BLE001
                            answer = (
                                "No se pudo generar la respuesta. "
                                f"Error del servicio de IA: {exc}"
                            )
                            sources = []

                    st.markdown(answer)
                    if sources:
                        with st.expander("🔗 Fuentes"):
                            for source in sources:
                                st.write(
                                    f"- {source.get('filename')} "
                                    f"(página {source.get('page')})"
                                )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )


def _process_files(uploaded_files, cfg) -> None:
    """Procesa e indexa solo los archivos nuevos, sin borrar los ya cargados."""
    if not uploaded_files:
        return

    collection = get_or_create_collection(
        st.session_state.session_id, cfg.persist_directory
    )

    all_new_chunks = []
    new_docs_info = []

    for uf in uploaded_files:
        try:
            documents = load_uploaded_file(uf)
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo leer {uf.name}: {exc}")
            continue

        if not documents:
            st.warning(f"{uf.name} no tiene texto extraíble.")
            continue

        chunks = create_chunks(documents, cfg.chunk_size, cfg.chunk_overlap)
        all_new_chunks.extend(chunks)
        new_docs_info.append(
            {
                "name": uf.name,
                "size": getattr(uf, "size", 0) or 0,
            }
        )

    if not all_new_chunks:
        st.error("No se pudo extraer texto de ninguno de los archivos seleccionados.")
        return

    texts = [chunk["page_content"] for chunk in all_new_chunks]
    embeddings = get_embeddings(texts, cfg.embedding_model)

    try:
        add_documents_to_collection(collection, all_new_chunks, embeddings)
        st.session_state.collection = collection
        st.session_state.vector_store_ready = True

        # Conservar en session_state para mostrar entre reruns sin reprocesar
        st.session_state.processed_files.extend(new_docs_info)
        st.session_state.document_chunks.extend(all_new_chunks)
        st.session_state.document_embeddings.extend(embeddings)

        st.success(
            f"Procesados {len(uploaded_files)} archivo(s) "
            f"en {len(all_new_chunks)} fragmento(s)."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error al indexar en la base vectorial: {exc}")


def _remove_document(filename: str, cfg) -> None:
    """Elimina un documento específico de la colección y del estado de sesión."""
    removed = 0
    if st.session_state.vector_store_ready and st.session_state.collection is not None:
        removed = remove_documents_by_filename(
            st.session_state.collection, filename
        )

    # Eliminar de session_state, manteniendo alineamiento entre chunks y embeddings
    filtered_chunks = []
    filtered_embeddings = []
    for chunk, emb in zip(
        st.session_state.document_chunks, st.session_state.document_embeddings
    ):
        if chunk["metadata"].get("filename") != filename:
            filtered_chunks.append(chunk)
            filtered_embeddings.append(emb)

    st.session_state.processed_files = [
        pf for pf in st.session_state.processed_files if pf["name"] != filename
    ]
    st.session_state.document_chunks = filtered_chunks
    st.session_state.document_embeddings = filtered_embeddings

    if not st.session_state.processed_files:
        st.session_state.vector_store_ready = False
        st.session_state.collection = None

    if removed > 0:
        st.success(f"Se eliminó '{filename}' ({removed} fragmento(s)).")
    else:
        st.warning(f"No se encontró '{filename}' en la colección.")


def _clear_session(cfg) -> None:
    """Borra todos los datos de la sesión actual."""
    delete_collection(st.session_state.session_id, cfg.persist_directory)
    st.session_state.uploaded_docs = []
    st.session_state.processed_files = []
    st.session_state.document_chunks = []
    st.session_state.document_embeddings = []
    st.session_state.messages = []
    st.session_state.vector_store_ready = False
    st.session_state.collection = None
    st.success("Sesión limpiada.")
    st.rerun()


if __name__ == "__main__":
    main()