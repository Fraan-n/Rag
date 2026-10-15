"""Generación de embeddings con el SDK oficial vigente de Google Gemini (google.genai)."""

from typing import List

from google import genai


def configure(api_key: str) -> None:
    """Guarda la API key para ser usada por el cliente de Google Gemini.

    El cliente de ``google.genai`` se instancia con la clave en cada operación,
    por lo que aquí solo validamos que exista.
    """
    if not api_key:
        raise ValueError("Falta la API key de Google Gemini.")
    configure.api_key = api_key


configure.api_key: str | None = None


def get_client() -> genai.Client:
    """Devuelve un cliente de Google Gemini configurado."""
    api_key = getattr(configure, "api_key", None)
    if not api_key:
        raise ValueError("Falta la API key de Google Gemini.")
    return genai.Client(api_key=api_key)


# Límite de la API de Gemini: máximo 100 requests por lote. Usamos 50 para
# dejar margen de seguridad y evitar errores de "too many requests in one batch".
EMBEDDING_BATCH_SIZE = 50


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    """Divide una lista en sublistas (lotes) de tamaño máximo `size`."""
    if size <= 0:
        raise ValueError("El tamaño de lote debe ser mayor a cero.")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_vectors(response) -> List[List[float]]:
    """Extrae los vectores de una respuesta de embeddings de Gemini."""
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        return []
    vectors: List[List[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if values is None:
            continue
        vectors.append(list(values))
    return vectors


def get_embeddings(texts: List[str], model: str = "gemini-embedding-001") -> List[List[float]]:
    """Genera embeddings para una lista de textos.

    Los textos se dividen en lotes de máximo 50 elementos para respetar el
    límite de la API de Gemini (100 requests por lote). Los resultados se
    combinan preservando el orden original.

    Returns:
        Lista de vectores (uno por cada texto de entrada).
    """
    if not texts:
        return []

    client = get_client()
    batches = _chunk_list(texts, EMBEDDING_BATCH_SIZE)

    all_vectors: List[List[float]] = []
    for batch_index, batch in enumerate(batches):
        try:
            response = client.models.embed_content(
                model=model,
                contents=batch,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Error al generar embeddings del lote {batch_index + 1} "
                f"({len(batch)} texto(s)): {exc}"
            ) from exc

        batch_vectors = _extract_vectors(response)
        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                f"El lote {batch_index + 1} devolvió {len(batch_vectors)} "
                f"embedding(s) pero se enviaron {len(batch)} texto(s)."
            )
        all_vectors.extend(batch_vectors)

    return all_vectors