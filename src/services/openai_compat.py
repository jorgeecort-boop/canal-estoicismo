"""Long-form stories via OpenAI-compatible APIs (OpenRouter, NVIDIA NIM, FreeLLMAPI).

All expose POST {base_url}/chat/completions with OpenAI schema.
Raises loudly on failure so callers never silently fall back to short
local templates when an explicit --llm provider was requested.
"""

import json
from typing import Optional

import requests

from src.narrative import Story

from src.services.google_ai import (
    GoogleAIService,
    StoicStoryData,
    create_story_from_gemini,
    story_within_budget,
    story_word_count,
)
from src.services.local_llm import _extract_json, maybe_trim_scenes, validate_story

PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        # Free/cheap strong Spanish narrator; override with OPENROUTER_MODEL.
        "default_model": "qwen/qwen-2.5-72b-instruct",
        "fallback_models": ["qwen/qwen-2.5-7b-instruct"],
        "key_url": "https://openrouter.ai/keys",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_API_KEY",
        # Kimi K2: strong Spanish prose. Override with NVIDIA_MODEL.
        # llama-3.1-* were retired from the hosted catalog (HTTP 410).
        "default_model": "moonshotai/kimi-k2-instruct",
        "fallback_models": ["deepseek-ai/deepseek-v4-flash", "meta/llama-3.1-70b-instruct"],
        "key_url": "https://build.nvidia.com",
    },
    "freellmapi": {
        # Self-hosted router (default http://localhost:3001/v1).
        # Only reachable where the router runs (your PC, NOT Colab).
        # Override with FREELLMAPI_URL + FREELLMAPI_MODEL.
        "base_url": "http://localhost:3001/v1",
        "key_env": "FREELLMAPI_API_KEY",
        # "auto" lets the router pick across its 34 free providers.
        "default_model": "auto",
        "key_url": "http://localhost:3001 (Keys page header)",
    },
}

TIMEOUT = 180


class OpenAICompatStoryService:
    """Generate validated stoic stories through any OpenAI-compatible API."""

    def __init__(
        self,
        provider: str = "openrouter",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        import os

        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        self.provider = provider
        self.config = PROVIDERS[provider]
        self.api_key = (
            api_key
            or os.getenv(self.config["key_env"])
            or os.getenv(f"{provider.upper()}_MODEL_KEY")
        )
        env_model = os.getenv(f"{provider.upper()}_MODEL", "")
        self.model = model or env_model or self.config["default_model"]
        self.base_url = (
            base_url or os.getenv(f"{provider.upper()}_URL", "") or self.config["base_url"]
        ).rstrip("/")

    def _chat(self, prompt: str, model: Optional[str] = None) -> str:
        model = model or self.model
        if not self.api_key:
            raise RuntimeError(
                f"Missing {self.config['key_env']}. "
                f"Get one at {self.config['key_url']}"
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/canal-estoicismo"
            headers["X-Title"] = "canal-estoicismo"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 8192,
                "response_format": {"type": "json_object"},
            },
            timeout=TIMEOUT,
        )
        if response.status_code in (401, 403):
            raise RuntimeError(f"{self.provider}: invalid API key (HTTP {response.status_code})")
        if response.status_code == 429:
            raise RuntimeError(f"{self.provider}: quota/rate limit (HTTP 429)")
        if response.status_code in (404, 410):
            raise RuntimeError(f"{self.provider}: model {model} gone (HTTP {response.status_code})")
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{self.provider}: unexpected response shape: {exc}") from exc
        if not content:
            raise ValueError(f"{self.provider}: empty response")
        return content

    def generate_story(
        self,
        theme: str,
        num_scenes: int = 10,
        target_duration_min: float = 6.0,
    ) -> Story:
        """Generate a validated story (models x attempts, loud failure)."""
        prompt = GoogleAIService._build_story_prompt(
            None, theme, num_scenes, target_duration_min
        )
        models = [self.model] + [
            m for m in self.config.get("fallback_models", []) if m != self.model
        ]
        last_error: Exception | None = None
        for model in models:
            for attempt in range(1, 3):
                try:
                    raw = self._chat(prompt, model)
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        data = _extract_json(raw)
                    story: Story = create_story_from_gemini(StoicStoryData(**data))
                    story = maybe_trim_scenes(story, num_scenes)
                    errors = validate_story(story, num_scenes, target_duration_min)
                    if errors:
                        raise ValueError("; ".join(errors))
                    print(f"   [OK] {self.provider} story: {model} "
                          f"({story_word_count(story)} words)")
                    return story
                except Exception as exc:
                    last_error = exc
                    message = str(exc)
                    if "gone (HTTP" in message or "NOT_FOUND" in message or " 404" in message:
                        print(f"   [WARN] {self.provider} model {model} retired, trying next...")
                        break
                    print(f"   [WARN] {self.provider} {model} attempt {attempt}: {exc}")
        raise RuntimeError(
            f"{self.provider} failed (models tried: {models}). "
            f"Last error: {last_error}"
        ) from last_error
