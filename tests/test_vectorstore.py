"""Pruebas de la base vectorial ligera (NumPy).

No requieren API key ni servidor externo.
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
    search_similar,
)


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _sample_chunks(self):
        return [
            {
                "page_content": "El cielo es azul",
                "metadata": {"filename": "a.txt", "page": 1},
            },
            {
                "page_content": "El pasto es verde",
                "metadata": {"filename": "a.txt", "page": 2},
            },
            {
                "page_content": "El mar es profundo",
                "metadata": {"filename": "b.txt", "page": 1},
            },
        ]

    def test_add_and_search_returns_most_similar(self):
        collection = get_or_create_collection("sess-1", self.tmpdir)
        chunks = self._sample_chunks()
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        add_documents_to_collection(collection, chunks, embeddings)

        results = search_similar(collection, [1.0, 0.0, 0.0], top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["document"], "El cielo es azul")
        self.assertEqual(results[0]["metadata"]["filename"], "a.txt")
        self.assertEqual(results[0]["metadata"]["page"], 1)
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])

    def test_empty_collection_returns_empty(self):
        collection = get_or_create_collection("sess-empty", self.tmpdir)
        results = search_similar(collection, [1.0, 0.0, 0.0], top_k=5)
        self.assertEqual(results, [])

    def test_delete_clears_session(self):
        collection = get_or_create_collection("sess-del", self.tmpdir)
        chunks = self._sample_chunks()[:1]
        add_documents_to_collection(collection, chunks, [[1.0, 0.0, 0.0]])
        self.assertEqual(len(collection.documents), 1)

        delete_collection("sess-del", self.tmpdir)
        collection2 = get_or_create_collection("sess-del", self.tmpdir)
        self.assertEqual(len(collection2.documents), 0)
        self.assertEqual(search_similar(collection2, [1.0, 0.0, 0.0]), [])

    def test_persistence_across_instances(self):
        session_id = "sess-persist"
        collection = get_or_create_collection(session_id, self.tmpdir)
        chunks = self._sample_chunks()[:1]
        add_documents_to_collection(collection, chunks, [[0.5, 0.5, 0.0]])

        # Nueva instancia misma sesión: debe cargar desde disco.
        collection2 = get_or_create_collection(session_id, self.tmpdir)
        self.assertEqual(len(collection2.documents), 1)
        self.assertEqual(collection2.documents[0], "El cielo es azul")

    def test_files_created(self):
        session_id = "sess-files"
        collection = get_or_create_collection(session_id, self.tmpdir)
        chunks = self._sample_chunks()[:1]
        add_documents_to_collection(collection, chunks, [[1.0, 0.0, 0.0]])

        npz_path = os.path.join(self.tmpdir, f"{session_id}.npz")
        json_path = os.path.join(self.tmpdir, f"{session_id}.json")
        self.assertTrue(os.path.exists(npz_path))
        self.assertTrue(os.path.exists(json_path))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["documents"][0], "El cielo es azul")


if __name__ == "__main__":
    unittest.main()