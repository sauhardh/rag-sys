from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = Field(validation_alias="DATABASE_URL")

    # OpenAI / Ollama
    openai_api_key: str = "ollama"
    openai_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "mistral:latest"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 86400  # 24h

    # Upload
    upload_dir: Path = Path(__file__).parent.parent.parent / "dump"
    allowed_extensions: tuple[str, ...] = (".txt", ".pdf")
    upload_chunk_size: int = 2048


settings = Settings()  # type: ignore[call-arg]
