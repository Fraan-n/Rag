"""Pruebas básicas del proyecto RAG.

No requieren API key ni base vectorial.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking import create_chunks, split_text


class TestChunking(unittest.TestCase):
    def test_split_text_long_text(self):
        text = "A" * 2500
        chunks = split_text(text, chunk_size=1000, overlap=100)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))

    def test_split_text_empty(self):
        self.assertEqual(split_text("", 1000, 100), [])

    def test_split_text_short(self):
        chunks = split_text("Hola mundo", 1000, 100)
        self.assertEqual(chunks, ["Hola mundo"])

    def test_create_chunks_preserves_metadata(self):
        docs = [
            {
                "page_content": "Hola mundo",
                "metadata": {"filename": "test.txt", "page": 1, "file_type": "txt"},
            }
        ]
        chunks = create_chunks(docs, chunk_size=10, chunk_overlap=2)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["filename"], "test.txt")
        self.assertEqual(chunks[0]["metadata"]["page"], 1)


if __name__ == "__main__":
    unittest.main()