"""Local story generation via Ollama (qwen2.5:7b primary, dolphin-llama3 fallback).

No quotas, no API keys. Raises loudly on failure so callers never
silently fall back to short local templates when --llm ollama was requested.
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

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
PRIMARY_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "dolphin-llama3"
TIMEOUT = 180


def ollama_available(url: str = OLLAMA_CHAT_URL) -> bool:
    """Check if an Ollama server answers (used for --llm auto)."""
    base = url.rsplit("/api/", 1)[0]
    try:
        response = requests.get(f"{base}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def _extract_json(text: str) -> dict:
    """Extract the first {...} JSON block from model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def validate_story(
    story: Story, num_scenes: int, target_duration_min: float
) -> list[str]:
    """Strict validation: scene count, required fields, word budget."""
    errors: list[str] = []
    if len(story.scenes) != num_scenes:
        errors.append(
            f"Expected {num_scenes} scenes, got {len(story.scenes)}"
        )
    for scene in story.scenes:
        for field in ("text_overlay", "voiceover_text", "image_prompt"):
            if not getattr(scene, field, ""):
                errors.append(f"Scene {scene.scene_number} missing {field}")
    if not story_within_budget(story, target_duration_min):
        errors.append(
            f"Word count {story_word_count(story)} outside ±10% budget "
            f"for {target_duration_min} min"
        )
    return errors


class OllamaStoryService:
    """Generate long-form stoic stories with local Ollama models."""

    def __init__(
        self,
        url: str = OLLAMA_CHAT_URL,
        models: Optional[list[str]] = None,
    ):
        self.url = url
        self.models = models or [PRIMARY_MODEL, FALLBACK_MODEL]

    def _chat(self, model: str, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "keep_alive": "15m",
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 8192,
                },
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError(f"Empty response from {model}")
        return content

    def generate_story(
        self,
        theme: str,
        num_scenes: int = 10,
        target_duration_min: float = 6.0,
    ) -> Story:
        """Try models in order; raise if all fail (never silent)."""
        prompt = GoogleAIService._build_story_prompt(
            None, theme, num_scenes, target_duration_min
        )
        last_error: Exception | None = None
        for model in self.models:
            try:
                raw = self._chat(model, prompt)
                data = _extract_json(raw)
                story = create_story_from_gemini(StoicStoryData(**data))
                errors = validate_story(story, num_scenes, target_duration_min)
                if errors:
                    raise ValueError("; ".join(errors))
                print(f"   [OK] Ollama story: {model} "
                      f"({story_word_count(story)} words)")
                return story
            except Exception as exc:
                last_error = exc
                print(f"   [WARN] Ollama {model} failed: {exc}")
        raise RuntimeError(
            f"All Ollama models failed ({self.models}). "
            f"Last error: {last_error}"
        ) from last_error
