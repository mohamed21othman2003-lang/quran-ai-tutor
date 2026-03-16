"""Centralised settings loaded from .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    chroma_persist_dir: str = "data/chroma_db"
    knowledge_dir: str = "data/knowledge"
    top_k: int = 3
    log_level: str = "INFO"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours
    db_path: str = "data/quran_tutor.db"

    # Admin
    admin_api_key: str = "change-me-admin-key"

    # Tafsir
    tafsir_db_dir: str = "data/tafsir"


settings = Settings()
