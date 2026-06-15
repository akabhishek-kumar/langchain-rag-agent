from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: str = "gsk_..."
    groq_model: str = "llama-3.1-8b-instant"

    retriever_k: int = 4
    grade_threshold: int = 6
    max_retries: int = 2
    recursion_limit: int = 20


settings = Settings()
