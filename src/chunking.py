"""División de texto en fragmentos (chunks) con solapamiento."""

from typing import Any, Dict, List


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Divide un texto en fragmentos de tamaño máximo chunk_size con solapamiento."""
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def create_chunks(
    documents: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> List[Dict[str, Any]]:
    """Crea fragmentos a partir de una lista de documentos preservando metadata."""
    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        content = doc["page_content"]
        for piece in split_text(content, chunk_size, chunk_overlap):
            chunks.append({"page_content": piece, "metadata": dict(doc["metadata"])})
    return chunks