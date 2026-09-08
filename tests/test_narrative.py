"""Tests for narrative/story_generator module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.narrative import StoryGenerator, Scene, Story
from src.narrative.templates import STOIC_TEMPLATES, get_template, list_themes


class TestTemplates:
    def test_all_themes_present(self):
        expected = [
            "control_dichotomy", "impermanence", "virtue_ethics", "amor_fati",
            "memento_mori", "inner_fortress", "adversity_growth", "present_moment",
            "ego_death", "cosmic_perspective",
        ]
        for theme in expected:
            assert theme in STOIC_TEMPLATES

    def test_template_structure(self):
        for theme, template in STOIC_TEMPLATES.items():
            assert template.theme == theme
            assert template.title
            assert template.philosopher
            assert template.hook
            assert template.core_teaching
            assert template.practical_application
            assert template.closing_reflection
            assert len(template.image_keywords) > 0

    def test_get_template(self):
        template = get_template("control_dichotomy")
        assert template.theme == "control_dichotomy"
        assert template.philosopher == "Epicteto"

    def test_get_template_invalid(self):
        with pytest.raises(ValueError):
            get_template("invalid_theme")

    def test_list_themes(self):
        themes = list_themes()
        assert len(themes) == 10
        assert "control_dichotomy" in themes


class TestScene:
    def test_scene_creation(self):
        scene = Scene(
            scene_number=1,
            text_overlay="Test overlay",
            voiceover_text="Test voiceover text for narration",
            image_prompt="Test image prompt",
            estimated_duration=10.0,
        )
        assert scene.scene_number == 1
        assert scene.text_overlay == "Test overlay"
        assert scene.estimated_duration == 10.0

    def test_scene_to_dict(self):
        scene = Scene(
            scene_number=1,
            text_overlay="Test",
            voiceover_text="Voiceover",
            image_prompt="Prompt",
        )
        d = scene.to_dict()
        assert d["scene_number"] == 1
        assert d["text_overlay"] == "Test"

    def test_scene_from_dict(self):
        data = {
            "scene_number": 2,
            "text_overlay": "Test",
            "voiceover_text": "Voiceover",
            "image_prompt": "Prompt",
            "estimated_duration": 5.0,
        }
        scene = Scene.from_dict(data)
        assert scene.scene_number == 2
        assert scene.estimated_duration == 5.0


class TestStory:
    def test_story_creation(self):
        scenes = [
            Scene(1, "Overlay 1", "Voiceover 1", "Prompt 1", 10.0),
            Scene(2, "Overlay 2", "Voiceover 2", "Prompt 2", 15.0),
        ]
        story = Story(
            theme="control_dichotomy",
            title="Test Title",
            philosopher="Epicteto",
            scenes=scenes,
            total_estimated_duration=25.0,
        )
        assert story.theme == "control_dichotomy"
        assert len(story.scenes) == 2
        assert story.total_estimated_duration == 25.0

    def test_story_to_dict(self):
        scenes = [Scene(1, "O", "V", "P", 10.0)]
        story = Story("theme", "Title", "Philosopher", scenes)
        d = story.to_dict()
        assert d["theme"] == "theme"
        assert len(d["scenes"]) == 1

    def test_story_from_dict(self):
        data = {
            "theme": "test",
            "title": "Title",
            "philosopher": "Phil",
            "scenes": [
                {"scene_number": 1, "text_overlay": "O", "voiceover_text": "V", "image_prompt": "P", "estimated_duration": 10.0}
            ],
            "total_estimated_duration": 10.0,
        }
        story = Story.from_dict(data)
        assert story.theme == "test"
        assert len(story.scenes) == 1
        assert story.scenes[0].estimated_duration == 10.0

    def test_story_json_serialization(self):
        scenes = [Scene(1, "O", "V", "P", 10.0)]
        story = Story("theme", "Title", "Phil", scenes)

        json_str = story.to_json()
        assert "theme" in json_str
        assert "Title" in json_str

        loaded = Story.from_json(json_str)
        assert loaded.theme == "theme"
        assert loaded.title == "Title"

    def test_story_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "story.json"
            scenes = [Scene(1, "O", "V", "P", 10.0)]
            story = Story("theme", "Title", "Phil", scenes)
            story.save(path)

            loaded = Story.load(path)
            assert loaded.theme == "theme"
            assert len(loaded.scenes) == 1


class TestStoryGenerator:
    def test_generate_default(self):
        gen = StoryGenerator()
        gen.set_seed(42)  # Reproducible
        story = gen.generate()

        assert story.theme in STOIC_TEMPLATES
        assert story.title
        assert story.philosopher
        assert len(story.scenes) >= 5
        assert len(story.scenes) <= 8
        assert story.total_estimated_duration > 0

    def test_generate_with_theme(self):
        gen = StoryGenerator()
        gen.set_seed(42)
        story = gen.generate(theme="impermanence")

        assert story.theme == "impermanence"
        assert story.title == "La Impermanencia de Todas las Cosas"
        assert story.philosopher == "Marco Aurelio"

    def test_generate_with_duration(self):
        gen = StoryGenerator()
        gen.set_seed(42)
        story = gen.generate(target_duration=4.0)

        # Duration is estimated from word count, so just check it's reasonable
        # (not too short, not excessively long)
        assert story.total_estimated_duration > 30.0  # At least 30 seconds
        assert story.total_estimated_duration < 600.0  # Not more than 10 minutes

    def test_generate_with_scenes(self):
        gen = StoryGenerator()
        gen.set_seed(42)
        story = gen.generate(num_scenes=6)

        assert len(story.scenes) == 6

    def test_generate_scene_structure(self):
        gen = StoryGenerator()
        gen.set_seed(42)
        story = gen.generate()

        for i, scene in enumerate(story.scenes):
            assert scene.scene_number == i + 1
            assert scene.text_overlay
            assert scene.voiceover_text
            assert scene.image_prompt
            assert scene.estimated_duration > 0
            # Text overlay may slightly exceed max due to "..."
            assert len(scene.text_overlay) <= 85  # max chars + ellipsis

    def test_generate_multiple(self):
        gen = StoryGenerator()
        gen.set_seed(42)
        stories = gen.generate_multiple(3)

        assert len(stories) == 3
        for story in stories:
            assert story.theme in STOIC_TEMPLATES
            assert len(story.scenes) > 0

    def test_generate_multiple_with_themes(self):
        gen = StoryGenerator()
        gen.set_seed(42)
        stories = gen.generate_multiple(2, themes=["control_dichotomy", "virtue_ethics"])

        assert len(stories) == 2
        # Themes are randomly chosen from the provided list
        for story in stories:
            assert story.theme in ["control_dichotomy", "virtue_ethics"]

    def test_reproducibility(self):
        gen1 = StoryGenerator()
        gen1.set_seed(123)
        story1 = gen1.generate(theme="control_dichotomy")

        gen2 = StoryGenerator()
        gen2.set_seed(123)
        story2 = gen2.generate(theme="control_dichotomy")

        assert story1.theme == story2.theme
        assert len(story1.scenes) == len(story2.scenes)
        for s1, s2 in zip(story1.scenes, story2.scenes):
            assert s1.text_overlay == s2.text_overlay
            assert s1.voiceover_text == s2.voiceover_text


class TestCreateStoryFromOutline:
    def test_custom_story(self):
        from src.narrative.story_generator import create_story_from_outline

        custom_scenes = [
            {"text_overlay": "Custom 1", "voiceover_text": "Voice 1", "image_prompt": "Prompt 1", "estimated_duration": 10.0},
            {"text_overlay": "Custom 2", "voiceover_text": "Voice 2", "image_prompt": "Prompt 2", "estimated_duration": 15.0},
        ]

        story = create_story_from_outline("control_dichotomy", custom_scenes)

        assert story.theme == "control_dichotomy"
        assert len(story.scenes) == 2
        assert story.scenes[0].text_overlay == "Custom 1"
        assert story.total_estimated_duration == 25.0