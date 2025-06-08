from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # Gemini Configuration
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "models/embedding-001"

    # ChromaDB Configuration
    CHROMA_PATH: str = "./chroma_db"
    CHROMA_COLLECTION: str = "multimodal_docs"

    # Allowed File Types
    allowed_file_types: List[str] = [".pdf", ".docx"]

    # Processing Limits
    MAX_FILE_SIZE: int = 10_000_000  # 10MB
    MAX_IMAGE_DIMENSION: int = 2048

    # API Server Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    DEBUG_MODE: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
