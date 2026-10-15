"""Pruebas de configuración y embeddings de Google Gemini (google.genai).

No realizan llamadas reales a la API.
"""

import os
import sys
import unittest
from unittest.mock import patch

from google import genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import embeddings
from src.config import AppConfig
from src import rag_chain


class TestConfigureAndEmbeddings(unittest.TestCase):
    def setUp(self):
        # Resetear el estado del módulo antes de cada prueba
        embeddings.configure.api_key = None

    def test_configure_sets_api_key_without_name_error(self):
        """configure() no debe lanzar NameError y debe guardar la clave."""
        embeddings.configure("test-api-key-123")
        self.assertEqual(embeddings.configure.api_key, "test-api-key-123")

    def test_configure_rejects_empty_key(self):
        """configure() debe rechazar claves vacías con ValueError."""
        with self.assertRaises(ValueError):
            embeddings.configure("")

    def test_configure_rejects_none_key(self):
        """configure() debe rechazar None con ValueError."""
        with self.assertRaises(ValueError):
            embeddings.configure(None)

    def test_get_client_raises_without_configuration(self):
        """get_client() debe fallar si configure() no se llamó previamente."""
        with self.assertRaises(ValueError):
            embeddings.get_client()

    def test_get_client_returns_client_after_configure(self):
        """get_client() debe devolver un cliente tras configure(), sin llamar a la API."""
        embeddings.configure("test-api-key-123")
        client = embeddings.get_client()
        self.assertIsInstance(client, genai.Client)

    @patch("src.embeddings.get_client")
    def test_get_embeddings_uses_configured_client(self, mock_get_client):
        """get_embeddings() debe usar el cliente configurado sin llamadas reales a la API."""
        mock_client = mock_get_client.return_value
        mock_embedding_item = type("Item", (), {"values": [0.1, 0.2, 0.3]})()
        mock_response = type("Response", (), {"embeddings": [mock_embedding_item]})()
        mock_client.models.embed_content.return_value = mock_response

        embeddings.configure("test-api-key-123")
        vectors = embeddings.get_embeddings(["texto de prueba"])

        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])
        mock_client.models.embed_content.assert_called_once_with(
            model="gemini-embedding-001",
            contents=["texto de prueba"],
        )

    @patch("src.embeddings.get_client")
    def test_get_embeddings_empty_list(self, mock_get_client):
        """Lista vacía debe retornar lista vacía sin llamar a la API."""
        embeddings.configure("test-api-key-123")
        vectors = embeddings.get_embeddings([])
        self.assertEqual(vectors, [])
        mock_get_client.return_value.models.embed_content.assert_not_called()

    @patch("src.embeddings.get_client")
    def test_get_embeddings_batch_size_limit(self, mock_get_client):
        """Ningún lote debe superar 50 elementos, incluso con más de 100 textos."""
        mock_client = mock_get_client.return_value

        def make_response(*args, **kwargs):
            contents = kwargs.get("contents", [])
            items = [
                type("Item", (), {"values": [float(i)]})()
                for i in range(len(contents))
            ]
            return type("Response", (), {"embeddings": items})()

        mock_client.models.embed_content.side_effect = make_response

        embeddings.configure("test-api-key-123")
        textos = [f"texto {i}" for i in range(150)]
        vectors = embeddings.get_embeddings(textos)

        # 150 textos en lotes de 50 => 3 llamadas
        self.assertEqual(mock_client.models.embed_content.call_count, 3)

        # Verificar que ningún lote supera 50
        for call_args in mock_client.models.embed_content.call_args_list:
            contents = call_args.kwargs["contents"]
            self.assertLessEqual(len(contents), 50)

        # Verificar que se procesaron todos los textos
        total_procesados = sum(
            len(call_args.kwargs["contents"])
            for call_args in mock_client.models.embed_content.call_args_list
        )
        self.assertEqual(total_procesados, 150)

        # Verificar que se devolvieron 150 vectores
        self.assertEqual(len(vectors), 150)

    @patch("src.embeddings.get_client")
    def test_get_embeddings_preserves_order(self, mock_get_client):
        """Los vectores deben mantener el orden original de los textos."""
        mock_client = mock_get_client.return_value

        def make_response(*args, **kwargs):
            contents = kwargs.get("contents", [])
            items = []
            for texto in contents:
                try:
                    num = float(texto.split()[-1])
                except (ValueError, IndexError):
                    num = 0.0
                items.append(type("Item", (), {"values": [num]})())
            return type("Response", (), {"embeddings": items})()

        mock_client.models.embed_content.side_effect = make_response

        embeddings.configure("test-api-key-123")
        textos = [f"texto {i}" for i in range(120)]
        vectors = embeddings.get_embeddings(textos)

        # Cada vector debe tener el valor correspondiente a su índice original
        for i, vector in enumerate(vectors):
            self.assertEqual(vector, [float(i)])

    @patch("src.embeddings.get_client")
    def test_get_embeddings_batch_error_propagates_cause(self, mock_get_client):
        """Un error en un lote debe propagarse con la causa original."""
        mock_client = mock_get_client.return_value
        mock_client.models.embed_content.side_effect = Exception("API rate limit exceeded")

        embeddings.configure("test-api-key-123")
        textos = [f"texto {i}" for i in range(60)]

        with self.assertRaises(RuntimeError) as ctx:
            embeddings.get_embeddings(textos)

        self.assertIn("lote", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, Exception)


class TestDefaultModel(unittest.TestCase):
    def test_default_embedding_model_is_gemini_embedding_001(self):
        """El modelo de embeddings predeterminado debe ser gemini-embedding-001."""
        import inspect

        sig = inspect.signature(embeddings.get_embeddings)
        default_model = sig.parameters["model"].default
        self.assertEqual(default_model, "gemini-embedding-001")

    def test_app_config_default_embedding_model(self):
        """AppConfig debe usar gemini-embedding-001 como modelo predeterminado."""
        cfg = AppConfig()
        # No se establece EMBEDDING_MODEL en el entorno de pruebas, por lo que
        # se usa el valor por defecto.
        self.assertEqual(cfg.embedding_model, "gemini-embedding-001")

    @patch("src.embeddings.get_client")
    def test_get_embeddings_uses_default_model(self, mock_get_client):
        """get_embeddings() sin argumento model debe usar gemini-embedding-001."""
        mock_client = mock_get_client.return_value

        def make_response(*args, **kwargs):
            contents = kwargs.get("contents", [])
            items = [
                type("Item", (), {"values": [0.1, 0.2, 0.3]})()
                for _ in contents
            ]
            return type("Response", (), {"embeddings": items})()

        mock_client.models.embed_content.side_effect = make_response

        embeddings.configure("test-api-key-123")
        embeddings.get_embeddings(["texto de prueba"])

        # Verificar que se llamó con el modelo predeterminado correcto
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-embedding-001")


class TestGenerationModel(unittest.TestCase):
    def test_default_generation_model_is_gemini_3_8_flash(self):
        """El modelo de generación predeterminado debe ser gemini-3.8-flash."""
        import inspect

        sig = inspect.signature(rag_chain.generate_answer)
        default_model = sig.parameters["model_name"].default
        self.assertEqual(default_model, "gemini-3.8-flash")

    def test_app_config_default_generation_model(self):
        """AppConfig debe usar gemini-3.8-flash como modelo predeterminado."""
        cfg = AppConfig()
        self.assertEqual(cfg.generation_model, "gemini-3.8-flash")

    @patch("src.rag_chain.genai.Client")
    def test_generate_answer_uses_default_model(self, mock_client_class):
        """generate_answer() sin argumento model_name debe usar gemini-3.8-flash."""
        mock_client = mock_client_class.return_value
        mock_response = type("Response", (), {"text": "Respuesta de prueba"})()
        mock_client.models.generate_content.return_value = mock_response

        rag_chain.generate_answer(
            query="¿Qué dice el documento?",
            contexts=["Contexto de prueba"],
            sources=[{"filename": "doc.txt", "page": 1}],
            api_key="test-api-key-123",
        )

        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gemini-3.8-flash")


if __name__ == "__main__":
    unittest.main()