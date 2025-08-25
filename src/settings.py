from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_model: str = Field(..., alias="LLM_MODEL", description="Language model to use for generation")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY", description="OpenAI API key for authentication")
    embedding_model: str = Field(..., alias="EMBEDDING_MODEL", description="Model name for text embeddings")
    milvus_uri: str = Field(..., alias="MILVUS_URI", description="Milvus database URI")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN", description="Milvus database token")
    cross_encoder_model: str = Field(
        ...,
        alias="CROSS_ENCODER_MODEL",
        description="Cross-encoder model for reranking",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",
        extra="ignore",
    )
