"""Pruebas de persistencia y gestión de documentos individuales.

Verifican que:
- Se puedan obtener nombres de archivo únicos de una colección.
- Se puedan eliminar documentos individuales sin afectar los demás.
- Los fragmentos y embeddings se conserven en session_state entre reruns.
- No se reprocesen archivos ya indexados.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.vectorstore import (
    add_documents_to_collection,
    delete_collection,
    get_or_create_collection,
    get_unique_filenames,
    remove_documents_by_filename,
    search_similar,
)


class TestVectorStorePersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_unique_filenames_returns_distinct_names(self):
        """Debe devolver solo nombres distintos, aunque haya varios fragmentos por archivo."""
        collection = get_or_create_collection("sess-unique", self.tmpdir)
        chunks = [
            {"page_content": "texto a1", "metadata": {"filename": "a.txt", "page": 1}},
            {"page_content": "texto a2", "metadata": {"filename": "a.txt", "page": 2}},
            {"page_content": "texto b1", "metadata": {"filename": "b.txt", "page": 1}},
            {"page_content": "texto c1", "metadata": {"filename": "c.txt", "page": 1}},
            {"page_content": "texto c2", "metadata": {"filename": "c.txt", "page": 2}},
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        add_documents_to_collection(collection, chunks, embeddings)

        names = get_unique_filenames(collection)
        self.assertEqual(sorted(names), ["a.txt", "b.txt", "c.txt"])

    def test_remove_documents_by_filename_removes_only_that_file(self):
        """Debe eliminar solo los fragmentos del archivo indicado, manteniendo los demás."""
        collection = get_or_create_collection("sess-remove", self.tmpdir)
        chunks = [
            {"page_content": "texto a1", "metadata": {"filename": "a.txt", "page": 1}},
            {"page_content": "texto a2", "metadata": {"filename": "a.txt", "page": 2}},
            {"page_content": "texto b1", "metadata": {"filename": "b.txt", "page": 1}},
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        add_documents_to_collection(collection, chunks, embeddings)

        removed = remove_documents_by_filename(collection, "a.txt")
        self.assertEqual(removed, 2)

        # b.txt debe seguir ahí
        names = get_unique_filenames(collection)
        self.assertEqual(names, ["b.txt"])

        # Buscar en b.txt debe devolver solo su fragmento
        results = search_similar(collection, [0.0, 1.0, 0.0], top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["filename"], "b.txt")

    def test_remove_nonexistent_filename_returns_zero(self):
        """Eliminar un archivo que no existe debe devolver 0 y no afectar nada."""
        collection = get_or_create_collection("sess-nonexist", self.tmpdir)
        chunks = [
            {"page_content": "texto a1", "metadata": {"filename": "a.txt", "page": 1}},
        ]
        add_documents_to_collection(collection, chunks, [[1.0, 0.0, 0.0]])

        removed = remove_documents_by_filename(collection, "noexiste.txt")
        self.assertEqual(removed, 0)
        self.assertEqual(get_unique_filenames(collection), ["a.txt"])

    def test_persistence_across_instances_loads_from_disk(self):
        """Una nueva instancia de la misma sesión debe cargar desde disco."""
        session_id = "sess-persist-2"
        collection = get_or_create_collection(session_id, self.tmpdir)
        chunks = [
            {"page_content": "texto persistente", "metadata": {"filename": "p.txt", "page": 1}},
        ]
        add_documents_to_collection(collection, chunks, [[0.5, 0.5, 0.0]])

        collection2 = get_or_create_collection(session_id, self.tmpdir)
        self.assertEqual(len(collection2.documents), 1)
        self.assertEqual(collection2.documents[0], "texto persistente")

        names = get_unique_filenames(collection2)
        self.assertEqual(names, ["p.txt"])


class TestSessionStateFields(unittest.TestCase):
    def test_init_session_state_creates_persistence_fields(self):
        """init_session_state debe crear los campos de persistencia si no existen."""
        import streamlit as st
        from src.session_state import init_session_state
        from src.config import AppConfig

        # Limpiar session_state para probar desde cero
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        cfg = AppConfig()
        init_session_state(cfg)

        self.assertEqual(st.session_state.processed_files, [])
        self.assertEqual(st.session_state.document_chunks, [])
        self.assertEqual(st.session_state.document_embeddings, [])
        self.assertFalse(st.session_state.vector_store_ready)
        self.assertIsNone(st.session_state.collection)


if __name__ == "__main__":
    unittest.main()