"""Generación de respuestas fundamentadas con Google Gemini (google.genai)."""

import time
from typing import Any, Dict, List

from google import genai

# Códigos de error HTTP del servidor que merecen reintento automático.
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
# Máximo de intentos totales (1 intento inicial + hasta 3 reintentos).
MAX_ATTEMPTS = 4
# Tiempo base de espera en segundos para backoff exponencial.
BASE_WAIT_SECONDS = 2


def _extract_status_code(exc: Exception) -> int | None:
    """Intenta extraer el código de estado HTTP de una excepción de la API."""
    code = getattr(exc, "code", None)
    if code is not None:
        try:
            return int(code)
        except (ValueError, TypeError):
            pass

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
        if status is not None:
            try:
                return int(status)
            except (ValueError, TypeError):
                pass

    return None


def _is_quota_exhausted(exc: Exception) -> bool:
    """Determina si un error 429 indica cuota diaria agotada (no reintentar)."""
    msg = str(exc).lower()
    quota_keywords = [
        "quota",
        "cuota",
        "daily",
        "día",
        "exhausted",
        "agotado",
        "limit reached",
    ]
    return any(kw in msg for kw in quota_keywords)


def _is_retryable_error(exc: Exception) -> bool:
    """Determina si un error merece reintento automático."""
    status = _extract_status_code(exc)
    if status is None:
        return False
    if status == 429:
        # 429 puede ser límite temporal (reintentar) o cuota agotada (no reintentar).
        return not _is_quota_exhausted(exc)
    return status in RETRYABLE_STATUS_CODES


def _get_retry_wait_seconds(exc: Exception, attempt: int) -> float:
    """Calcula segundos de espera antes del siguiente reintento.

    Respeta ``retry_after`` si la API lo indica; de lo contrario usa
    backoff exponencial limitado.
    """
    retry_after = getattr(exc, "retry_after", None) or getattr(
        exc, "retry_after_seconds", None
    )
    if retry_after is not None:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass
    return BASE_WAIT_SECONDS * (2 ** attempt)


def build_prompt(
    query: str,
    contexts: List[str],
    sources: List[Dict[str, Any]],
) -> str:
    """Construye el prompt con instrucciones, contexto y fuentes."""
    source_lines = "\n".join(
        f"- {s.get('filename', 'desconocido')} (página {s.get('page', '?')})"
        for s in sources
    )
    context_text = "\n\n---\n\n".join(contexts)

    return f"""Eres un asistente que responde preguntas basándote ÚNICAMENTE en los fragmentos de documentos proporcionados.

Instrucciones:
- Responde solo con la información contenida en los fragmentos.
- Si la respuesta no se puede deducir del contexto, di claramente que no hay información suficiente.
- No inventes datos ni cites documentos que no estén en la lista de fuentes.
- Al final, indica las fuentes utilizadas.

Fuentes disponibles:
{source_lines}

Contexto:
{context_text}

Pregunta: {query}

Respuesta:"""


def generate_answer(
    query: str,
    contexts: List[str],
    sources: List[Dict[str, Any]],
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
) -> str:
    """Genera una respuesta usando Google Gemini con los fragmentos recuperados.

    Incluye reintentos automáticos limitados para errores temporales del
    servidor (500, 502, 503, 504) y límites de velocidad temporales (429).
    No reintenta errores permanentes como 404 ni cuota diaria agotada (429).
    """
    if api_key:
        from google import genai as _genai

        client = _genai.Client(api_key=api_key)
    else:
        client = genai.Client()  # se espera GOOGLE_API_KEY en el entorno

    prompt = build_prompt(query, contexts, sources)

    last_exception: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text or "No se pudo generar una respuesta."
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            if attempt == MAX_ATTEMPTS - 1:
                break
            if not _is_retryable_error(exc):
                break
            wait = _get_retry_wait_seconds(exc, attempt)
            time.sleep(wait)

    # Si llegamos aquí, falló después de reintentos o fue error permanente.
    if last_exception is not None:
        status = _extract_status_code(last_exception)
        if status == 429:
            if _is_quota_exhausted(last_exception):
                raise RuntimeError(
                    "Se agotó la cuota diaria de Google Gemini. "
                    "Inténtalo nuevamente más tarde o verifica tu plan de API."
                ) from last_exception
            raise RuntimeError(
                "El servicio de Google Gemini está temporalmente sobrecargado "
                "(límite de velocidad). Inténtalo nuevamente en unos minutos."
            ) from last_exception
        elif status in RETRYABLE_STATUS_CODES:
            raise RuntimeError(
                f"El servicio de Google Gemini no está disponible temporalmente "
                f"(código {status}). Inténtalo nuevamente más tarde."
            ) from last_exception
        elif status == 404:
            raise RuntimeError(
                f"El modelo de Gemini '{model_name}' no fue encontrado "
                f"o no es compatible con generate_content."
            ) from last_exception
        else:
            raise RuntimeError(
                f"Error del servicio de IA: {last_exception}"
            ) from last_exception

    return "No se pudo generar una respuesta."