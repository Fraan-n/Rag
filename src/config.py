import os
from dotenv import load_dotenv


class AppConfig:
    """Configuración de la aplicación cargada desde variables de entorno."""

    def __init__(self) -> None:
        self.google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.generation_model: str = os.getenv("GENERATION_MODEL", "gemini-3.8-flash")
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.top_k: int = int(os.getenv("TOP_K", "5"))
        self.similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
        self.max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
        self.persist_directory: str = os.getenv("VECTOR_STORE_DIR", "./vector_store")


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig()