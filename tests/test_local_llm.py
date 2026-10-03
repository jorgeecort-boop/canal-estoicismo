"""Tests for Ollama local story provider (mocked, no server needed)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.narrative import Scene, Story
from src.services.local_llm import (
    OllamaStoryService,
    ollama_available,
    validate_story,
)


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
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": json.dumps(payload)}}
    return response


class TestOllamaAvailable:
    def test_true(self):
        with patch("src.services.local_llm.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            assert ollama_available() is True

    def test_false_on_error(self):
        with patch("src.services.local_llm.requests.get", side_effect=ConnectionError):
            assert ollama_available() is False


class TestOllamaStoryService:
    def test_primary_success(self):
        service = OllamaStoryService()
        with patch("src.services.local_llm.requests.post") as mock_post:
            mock_post.return_value = _ok_response(_payload())
            story = service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert len(story.scenes) == 2
        assert story.title == "T"

    def test_fallback_to_second_model(self):
        service = OllamaStoryService(models=["qwen2.5:7b", "dolphin-llama3"])
        with patch("src.services.local_llm.requests.post") as mock_post:
            mock_post.side_effect = [TimeoutError("t"), _ok_response(_payload())]
            story = service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert len(story.scenes) == 2
        assert mock_post.call_count == 2

    def test_all_fail_raises_loudly(self):
        service = OllamaStoryService()
        with patch("src.services.local_llm.requests.post", side_effect=TimeoutError("t")):
            with pytest.raises(RuntimeError, match="All Ollama models failed"):
                service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)

    def test_invalid_json_falls_to_second_model(self):
        service = OllamaStoryService()
        bad = MagicMock()
        bad.raise_for_status.return_value = None
        bad.json.return_value = {"message": {"content": "not json at all"}}
        with patch("src.services.local_llm.requests.post") as mock_post:
            mock_post.side_effect = [bad, _ok_response(_payload())]
            story = service.generate_story("control_dichotomy", num_scenes=2, target_duration_min=1.0)
        assert len(story.scenes) == 2


class TestValidateStory:
    def _story(self, num_scenes=2, words=70):
        scenes = [Scene(i, f"O{i}", " ".join(["w"] * words), f"P{i}") for i in range(1, num_scenes + 1)]
        return Story("control_dichotomy", "T", "Epicteto", scenes)

    def test_valid(self):
        assert validate_story(self._story(), 2, 1.0) == []

    def test_wrong_scene_count(self):
        errors = validate_story(self._story(3), 2, 1.0)
        assert any("scenes" in e for e in errors)

    def test_missing_field(self):
        story = self._story()
        story.scenes[0].image_prompt = ""
        errors = validate_story(story, 2, 1.0)
        assert any("image_prompt" in e for e in errors)

    def test_out_of_budget(self):
        story = self._story(words=10)
        errors = validate_story(story, 2, 1.0)
        assert any("budget" in e for e in errors)
