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
            with pytest.raises(RuntimeError, match="models tried"):
                service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert mock_post.call_count == 4  # 2 attempts x (default + fallback)

    def test_410_advances_to_fallback_model(self):
        service = OpenAICompatStoryService("nvidia", api_key="k")
        gone = MagicMock()
        gone.status_code = 410
        with patch("src.services.openai_compat.requests.post") as mock_post:
            mock_post.side_effect = [gone, _ok_response(_payload())]
            story = service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert len(story.scenes) == 2
        models_called = [c[1]["json"]["model"] for c in mock_post.call_args_list]
        assert models_called[0] == "moonshotai/kimi-k2-instruct"
        assert models_called[1] == "deepseek-ai/deepseek-v4-flash"

    def test_extra_scenes_trimmed_not_rejected(self):
        service = OpenAICompatStoryService("openrouter", api_key="k")
        with patch("src.services.openai_compat.requests.post") as mock_post:
            mock_post.return_value = _ok_response(_payload(num_scenes=5))
            story = service.generate_story("control_dichotomy", num_scenes=3, target_duration_min=1.5)
        assert len(story.scenes) == 3
        assert mock_post.call_count == 1

    def test_nvidia_default_is_current(self):
        service = OpenAICompatStoryService("nvidia", api_key="k")
        assert service.model == "moonshotai/kimi-k2-instruct"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            OpenAICompatStoryService("anthropic", api_key="k")


class TestLLMChain:
    def test_auto_order_gemini_then_nvidia_then_openrouter(self):
        from main import _llm_chain
        assert _llm_chain("auto") == ["ollama", "gemini", "nvidia", "openrouter", "freellmapi"]

    def test_explicit_single(self):
        from main import _llm_chain
        assert _llm_chain("nvidia") == ["nvidia"]
        assert _llm_chain(None) == []

    def test_freellmapi_defaults(self, monkeypatch):
        monkeypatch.delenv("FREELLMAPI_MODEL", raising=False)
        monkeypatch.delenv("FREELLMAPI_URL", raising=False)
        service = OpenAICompatStoryService("freellmapi", api_key="freellmapi-test")
        assert service.base_url == "http://localhost:3001/v1"
        assert service.model == "auto"

    def test_freellmapi_env_overrides(self, monkeypatch):
        monkeypatch.setenv("FREELLMAPI_URL", "http://192.168.1.10:3001/v1")
        monkeypatch.setenv("FREELLMAPI_MODEL", "qwen-test")
        service = OpenAICompatStoryService("freellmapi", api_key="k")
        assert service.base_url == "http://192.168.1.10:3001/v1"
        assert service.model == "qwen-test"

    def test_freellmapi_success(self):
        service = OpenAICompatStoryService("freellmapi", api_key="k")
        with patch("src.services.openai_compat.requests.post") as mock_post:
            mock_post.return_value = _ok_response(_payload())
            story = service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert len(story.scenes) == 2
        called_url = mock_post.call_args[0][0]
        assert called_url == "http://localhost:3001/v1/chat/completions"
