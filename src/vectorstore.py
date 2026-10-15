"""Base vectorial ligera con NumPy y persistencia en disco.

Cada sesión de usuario guarda sus propios vectores y metadatos en archivos
independientes dentro de ``persist_directory``. Esto brinda aislamiento entre
sesiones sin requerir un servidor externo ni compiladores de C++.

Los archivos generados por sesión son:
- ``<session_id>.npz`` : matriz de embeddings.
- ``<session_id>.json`` : fragmentos y metadatos asociados.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np


class _SessionStore:
    """Almacén de vectores para una sesión específica."""

    def __init__(self, session_id: str, persist_directory: str) -> None:
        self.session_id = session_id
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.embeddings_path = os.path.join(persist_directory, f"{session_id}.npz")
        self.meta_path = os.path.join(persist_directory, f"{session_id}.json")
        self.embeddings: np.ndarray | None = None
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.embeddings_path) and os.path.exists(self.meta_path):
            data = np.load(self.embeddings_path)
            self.embeddings = data["embeddings"]
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.documents = meta.get("documents", [])
            self.metadatas = meta.get("metadatas", [])
        else:
            self.embeddings = None
            self.documents = []
            self.metadatas = []

    def _save(self) -> None:
        np.savez(self.embeddings_path, embeddings=self.embeddings)
        payload = {
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def add(
        self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        embs = np.array(embeddings, dtype="float32")
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])
        self.documents.extend([chunk["page_content"] for chunk in chunks])
        self.metadatas.extend([chunk["metadata"] for chunk in chunks])
        self._save()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Busca por similitud de coseno y devuelve los mejores fragmentos."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        q = np.array(query_embedding, dtype="float32")
        emb_norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = float(np.linalg.norm(q))

        if q_norm == 0.0 or np.all(emb_norms == 0):
            return []

        scores = (self.embeddings @ q) / (emb_norms * q_norm)
        top_indices = np.argsort(-scores)[: int(top_k)]

        results: List[Dict[str, Any]] = []
        for i in top_indices:
            score = float(scores[i])
            results.append(
                {
                    "document": self.documents[i],
                    "metadata": self.metadatas[i] or {},
                    "distance": float(1.0 - score),
                    "score": score,
                }
            )

        return results

    def delete(self) -> None:
        for path in (self.embeddings_path, self.meta_path):
            if os.path.exists(path):
                os.remove(path)
        self.embeddings = None
        self.documents = []
        self.metadatas = []

    def get_unique_filenames(self) -> List[str]:
        """Devuelve la lista de nombres de archivo únicos en la colección."""
        seen = set()
        result: List[str] = []
        for meta in self.metadatas:
            name = meta.get("filename")
            if name and name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def get_documents_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        """Devuelve todos los fragmentos (documento + metadata) de un archivo."""
        result: List[Dict[str, Any]] = []
        for doc, meta in zip(self.documents, self.metadatas):
            if meta.get("filename") == filename:
                result.append({"page_content": doc, "metadata": dict(meta)})
        return result

    def remove_by_filename(self, filename: str) -> int:
        """Elimina todos los fragmentos de un archivo determinado.

        Returns:
            Cantidad de fragmentos eliminados.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return 0

        keep_indices = [
            i for i, meta in enumerate(self.metadatas)
            if meta.get("filename") != filename
        ]
        removed_count = len(self.documents) - len(keep_indices)

        if removed_count == 0:
            return 0

        if keep_indices:
            self.embeddings = self.embeddings[keep_indices]
            self.documents = [self.documents[i] for i in keep_indices]
            self.metadatas = [self.metadatas[i] for i in keep_indices]
        else:
            self.embeddings = None
            self.documents = []
            self.metadatas = []

        self._save()
        return removed_count


# Caché en memoria para evitar recargar desde disco en cada llamada dentro de
# una misma ejecución de la aplicación.
_stores_cache: Dict[tuple, _SessionStore] = {}


def get_or_create_collection(
    session_id: str, persist_directory: str = "./vector_store"
) -> _SessionStore:
    """Obtiene o crea el almacén de vectores para una sesión."""
    key = (session_id, persist_directory)
    if key not in _stores_cache:
        _stores_cache[key] = _SessionStore(session_id, persist_directory)
    return _stores_cache[key]


def add_documents_to_collection(
    collection: _SessionStore,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> None:
    """Agrega fragmentos y sus embeddings al almacén."""
    collection.add(chunks, embeddings)


def search_similar(
    collection: _SessionStore,
    query_embedding: List[float],
    top_k: int = 5,
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Busca fragmentos similares al vector de consulta."""
    return collection.search(query_embedding, top_k=top_k, threshold=threshold)


def delete_collection(session_id: str, persist_directory: str = "./vector_store") -> None:
    """Elimina los vectores de una sesión (aislamiento entre usuarios)."""
    key = (session_id, persist_directory)
    if key in _stores_cache:
        _stores_cache[key].delete()
        del _stores_cache[key]
    # También asegurar borrado de archivos aunque no exista caché.
    _SessionStore(session_id, persist_directory).delete()


def get_unique_filenames(collection) -> List[str]:
    """Devuelve la lista de nombres de archivo únicos en la colección."""
    return collection.get_unique_filenames()


def remove_documents_by_filename(
    collection, filename: str
) -> int:
    """Elimina todos los fragmentos de un archivo de la colección.

    Returns:
        Cantidad de fragmentos eliminados.
    """
    return collection.remove_by_filename(filename)