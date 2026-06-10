from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- LM mode ---
    # "direct"  → connect straight to Ollama (Option A in the DSPy skeleton)
    # "gateway" → route through the AI Gateway (Option B) — all calls go through
    #             auth, logging, rate limiting, and the full pipeline
    lm_mode: str = "direct"

    # --- Option A: direct Ollama ---
    ollama_api_base: str = "http://localhost:11434"

    # --- Option B: AI Gateway ---
    gateway_url: str = "http://localhost:8000/v1"
    gateway_api_key: str = ""

    # --- Model used for ticket classification ---
    # Use the LiteLLM provider prefix: ollama/..., gemini/..., etc.
    # When lm_mode="gateway", this must also be in the gateway's allowed_models list.
    classifier_model: str = "ollama/llama3.1"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
