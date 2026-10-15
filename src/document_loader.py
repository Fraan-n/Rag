"""Extracción de texto desde archivos subidos (PDF, DOCX, TXT).

Cada documento extraído se devuelve como un diccionario con:
- page_content: texto extraído.
- metadata: nombre de archivo, página (cuando aplica), tipo y tamaño.
"""

import io
from typing import Any, Dict, List

import docx
import pypdf


def load_uploaded_file(uploaded_file) -> List[Dict[str, Any]]:
    """Extrae texto de un archivo subido por Streamlit.

    Args:
        uploaded_file: objeto de st.file_uploader (atributos: name, type, size, read()).

    Returns:
        Lista de diccionarios con page_content y metadata.
    """
    name = uploaded_file.name
    file_type = (uploaded_file.type or "").lower()
    size = getattr(uploaded_file, "size", 0) or 0

    data = uploaded_file.read()
    uploaded_file.seek(0)

    documents: List[Dict[str, Any]] = []

    lower_name = name.lower()

    if lower_name.endswith(".pdf") or "pdf" in file_type:
        reader = pypdf.PdfReader(io.BytesIO(data))
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            documents.append(
                {
                    "page_content": text,
                    "metadata": {
                        "filename": name,
                        "page": index + 1,
                        "file_type": "pdf",
                        "size": size,
                    },
                }
            )

    elif lower_name.endswith(".docx") or "word" in file_type or "document" in file_type:
        document = docx.Document(io.BytesIO(data))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        for index, paragraph in enumerate(paragraphs):
            documents.append(
                {
                    "page_content": paragraph,
                    "metadata": {
                        "filename": name,
                        "page": index + 1,
                        "file_type": "docx",
                        "size": size,
                    },
                }
            )

    elif lower_name.endswith(".txt") or "text" in file_type:
        text = data.decode("utf-8", errors="ignore")
        lines = [line for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            documents.append(
                {
                    "page_content": line,
                    "metadata": {
                        "filename": name,
                        "page": index + 1,
                        "file_type": "txt",
                        "size": size,
                    },
                }
            )

    return documents