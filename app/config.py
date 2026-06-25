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

    # --- RAG retrieval (M21) ---
    embed_model: str = "nomic-embed-text"  # model name as Ollama knows it (no provider prefix)
    knowledge_base_path: str = "data/knowledge_base.json"  # relative to project root

    # --- Context moderation (M29) ---
    # Model that inspects retrieved KB content for instruction injection before it
    # is passed to any DSPy predictor. Should be a dedicated guard model where
    # possible (e.g. Llama Guard); defaults to the classifier model.
    # Must be specified WITHOUT the provider prefix — it is called via httpx
    # directly (not through DSPy/LiteLLM) because forward() runs in a thread.
    # Leave empty to auto-derive from classifier_model (strips "ollama/" prefix).
    moderation_model: str = ""
    moderation_timeout_seconds: float = 30.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
