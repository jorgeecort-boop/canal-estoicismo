"""Tests for OpenAI-compatible story providers (mocked, no network)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.services.openai_compat import OpenAICompatStoryService


def _payload(num_scenes=2, words_per_scene=70):
    scenes = [
        {
            "scene_number": i,
            "text_overlay": f"Overlay {i}",
            "voiceover_text": " ".join(["word"] * words_per_scene),
            "image_prompt": f"Prompt {i}",
            "estimated_duration": 30.0,
        }
        for i in range(1, num_scenes + 1)
    ]
    return {
        "title": "T",
        "philosopher": "Epicteto",
        "theme": "control_dichotomy",
        "scenes": scenes,
        "total_estimated_duration": 60.0,
    }


def _ok_response(payload):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    return response


class TestOpenAICompat:
    def test_success(self):
        service = OpenAICompatStoryService("openrouter", api_key="k")
        with patch("src.services.openai_compat.requests.post") as mock_post:
            mock_post.return_value = _ok_response(_payload())
            story = service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert len(story.scenes) == 2
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "qwen/qwen-2.5-72b-instruct"
        assert body["response_format"] == {"type": "json_object"}

    def test_env_model_override(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
        service = OpenAICompatStoryService("nvidia", api_key="k")
        assert service.model == "meta/llama-3.1-70b-instruct"

    def test_missing_key_is_loud(self):
        service = OpenAICompatStoryService("openrouter", api_key=None)
        service.api_key = ""
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            service.generate_story("control_dichotomy", num_scenes=1, target_duration_min=1.0)

    def test_401_is_loud(self):
        service = OpenAICompatStoryService("nvidia", api_key="bad")
        response = MagicMock()
        response.status_code = 401
        with patch("src.services.openai_compat.requests.post", return_value=response):
            with pytest.raises(RuntimeError, match="invalid API key"):
                service.generate_story("control_dichotomy", num_scenes=1, target_duration_min=1.0)

    def test_out_of_budget_retries_then_raises(self):
        service = OpenAICompatStoryService("openrouter", api_key="k")
        with patch("src.services.openai_compat.requests.post") as mock_post:
            mock_post.return_value = _ok_response(_payload(words_per_scene=10))
            with pytest.raises(RuntimeError, match="failed after 2 attempts"):
                service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert mock_post.call_count == 2

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            OpenAICompatStoryService("anthropic", api_key="k")
