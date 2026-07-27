from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    NVIDIA_API_KEY: str
    TAVILY_API_KEY: str | None = None
    GITHUB_TOKEN: str | None = None

    QDRANT_URL: str
    QDRANT_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()