"""Pruebas de reintentos y manejo de errores en rag_chain.

Todas las pruebas usan mocks y no realizan llamadas reales a la API de Gemini.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rag_chain


class _MockAPIError(Exception):
    """Excepción simulada de la API de Gemini con código de estado."""

    def __init__(self, message: str, code: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class TestRAGChainRetries(unittest.TestCase):
    def setUp(self):
        self.query = "¿Qué dice el documento?"
        self.contexts = ["Contexto de prueba"]
        self.sources = [{"filename": "doc.txt", "page": 1}]

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_success_first_attempt_no_retries(self, mock_sleep, mock_client_class):
        """Éxito en el primer intento: no hay reintentos ni sleeps."""
        mock_client = mock_client_class.return_value
        mock_response = type("Response", (), {"text": "Respuesta exitosa"})()
        mock_client.models.generate_content.return_value = mock_response

        result = rag_chain.generate_answer(
            query=self.query,
            contexts=self.contexts,
            sources=self.sources,
            api_key="test-api-key-123",
        )

        self.assertEqual(result, "Respuesta exitosa")
        mock_client.models.generate_content.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_retry_on_503_then_success(self, mock_sleep, mock_client_class):
        """503 en los primeros 2 intentos, éxito en el tercero."""
        mock_client = mock_client_class.return_value
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise _MockAPIError("503 UNAVAILABLE", code=503)
            return type("Response", (), {"text": "Respuesta tras reintento"})()

        mock_client.models.generate_content.side_effect = side_effect

        result = rag_chain.generate_answer(
            query=self.query,
            contexts=self.contexts,
            sources=self.sources,
            api_key="test-api-key-123",
        )

        self.assertEqual(result, "Respuesta tras reintento")
        self.assertEqual(call_count, 3)
        # 2 sleeps entre los 3 intentos
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_retry_on_500_then_success(self, mock_sleep, mock_client_class):
        """500 en el primer intento, éxito en el segundo."""
        mock_client = mock_client_class.return_value
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _MockAPIError("500 Internal Server Error", code=500)
            return type("Response", (), {"text": "OK"})()

        mock_client.models.generate_content.side_effect = side_effect

        result = rag_chain.generate_answer(
            query=self.query,
            contexts=self.contexts,
            sources=self.sources,
            api_key="test-api-key-123",
        )

        self.assertEqual(result, "OK")
        self.assertEqual(call_count, 2)
        mock_sleep.assert_called_once()

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_503_exhausts_all_attempts(self, mock_sleep, mock_client_class):
        """503 persistente: agota todos los intentos y lanza RuntimeError claro."""
        mock_client = mock_client_class.return_value
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _MockAPIError("503 UNAVAILABLE", code=503)

        mock_client.models.generate_content.side_effect = side_effect

        with self.assertRaises(RuntimeError) as ctx:
            rag_chain.generate_answer(
                query=self.query,
                contexts=self.contexts,
                sources=self.sources,
                api_key="test-api-key-123",
            )

        self.assertEqual(call_count, rag_chain.MAX_ATTEMPTS)
        self.assertEqual(mock_sleep.call_count, rag_chain.MAX_ATTEMPTS - 1)
        msg = str(ctx.exception)
        self.assertIn("no está disponible temporalmente", msg)
        self.assertIn("503", msg)

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_429_quota_exhausted_no_retry(self, mock_sleep, mock_client_class):
        """429 con cuota agotada: no reintenta, mensaje claro."""
        mock_client = mock_client_class.return_value
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _MockAPIError(
                "429 RESOURCE_EXHAUSTED: daily quota exhausted",
                code=429,
            )

        mock_client.models.generate_content.side_effect = side_effect

        with self.assertRaises(RuntimeError) as ctx:
            rag_chain.generate_answer(
                query=self.query,
                contexts=self.contexts,
                sources=self.sources,
                api_key="test-api-key-123",
            )

        self.assertEqual(call_count, 1)
        mock_sleep.assert_not_called()
        msg = str(ctx.exception)
        self.assertIn("cuota", msg.lower())

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_429_temporary_limit_retries(self, mock_sleep, mock_client_class):
        """429 sin indicio de cuota agotada: se reintenta como error temporal."""
        mock_client = mock_client_class.return_value
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _MockAPIError("429 Too Many Requests", code=429, retry_after=1)
            return type("Response", (), {"text": "OK tras espera"})()

        mock_client.models.generate_content.side_effect = side_effect

        result = rag_chain.generate_answer(
            query=self.query,
            contexts=self.contexts,
            sources=self.sources,
            api_key="test-api-key-123",
        )

        self.assertEqual(result, "OK tras espera")
        self.assertEqual(call_count, 2)
        mock_sleep.assert_called_once_with(1)  # respeta retry_after

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_404_permanent_no_retry(self, mock_sleep, mock_client_class):
        """404 es permanente: no reintenta, mensaje claro."""
        mock_client = mock_client_class.return_value
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise _MockAPIError("404 Not Found", code=404)

        mock_client.models.generate_content.side_effect = side_effect

        with self.assertRaises(RuntimeError) as ctx:
            rag_chain.generate_answer(
                query=self.query,
                contexts=self.contexts,
                sources=self.sources,
                api_key="test-api-key-123",
            )

        self.assertEqual(call_count, 1)
        mock_sleep.assert_not_called()
        msg = str(ctx.exception)
        self.assertIn("no fue encontrado", msg)

    @patch("src.rag_chain.genai.Client")
    @patch("src.rag_chain.time.sleep")
    def test_unknown_error_no_retry(self, mock_sleep, mock_client_class):
        """Error sin código reconocible: no reintenta, mensaje genérico."""
        mock_client = mock_client_class.return_value
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Error desconocido de red")

        mock_client.models.generate_content.side_effect = side_effect

        with self.assertRaises(RuntimeError) as ctx:
            rag_chain.generate_answer(
                query=self.query,
                contexts=self.contexts,
                sources=self.sources,
                api_key="test-api-key-123",
            )

        self.assertEqual(call_count, 1)
        mock_sleep.assert_not_called()
        msg = str(ctx.exception)
        self.assertIn("Error del servicio de IA", msg)


if __name__ == "__main__":
    unittest.main()