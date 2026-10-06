"""Tests for config module."""

import tempfile
from pathlib import Path

import pytest

from config import Settings, get_settings, reset_settings
from config.settings import (
    VideoSettings,
    FontSettings,
    KenBurnsSettings,
    TTSSettings,
    ImageSettings,
    NarrativeSettings,
    VisualEngineSettings,
    PathSettings,
)


class TestVideoSettings:
    def test_default_values(self):
        s = VideoSettings()
        assert s.width == 1920
        assert s.height == 1080
        assert s.fps == 30
        assert s.codec == "libx264"

    def test_resolution_property(self):
        s = VideoSettings(width=1280, height=720)
        assert s.resolution == (1280, 720)

    def test_aspect_ratio(self):
        s = VideoSettings()
        assert s.aspect_ratio == "16:9"


class TestFontSettings:
    def test_default_values(self):
        s = FontSettings()
        assert s.family == "Georgia"
        assert s.size == 72
        assert s.color == "#F5F0E1"
        assert s.position == "bottom"


class TestKenBurnsSettings:
    def test_default_values(self):
        s = KenBurnsSettings()
        assert s.enabled is True
        assert s.zoom_factor == 1.24
        assert s.direction == "random"


class TestTTSSettings:
    def test_default_values(self):
        s = TTSSettings()
        assert s.provider == "edge-tts"
        assert s.voice == "es-MX-JorgeNeural"
        assert s.rate == "-15%"
        assert s.voice_gravitas_semitones == 2.0


class TestImageSettings:
    def test_default_values(self):
        s = ImageSettings()
        assert s.provider == "local_assets"
        assert "marble statue" in s.style_prompt

    def test_sdxl_opt_in_defaults(self):
        s = ImageSettings()
        assert s.sdxl_width == 1024
        assert s.sdxl_height == 576
        assert s.upscale_enabled is False


class TestNarrativeSettings:
    def test_default_values(self):
        s = NarrativeSettings()
        assert s.target_duration_min == 5.0
        assert s.target_duration_max == 8.0
        assert s.scenes_min == 5
        assert s.scenes_max == 8
        assert "control_dichotomy" in s.themes
        assert "Marcus Aurelius" in s.philosophers


class TestVisualEngineSettings:
    def test_disabled_by_default(self):
        s = VisualEngineSettings()
        assert s.enabled is False
        assert s.particle_count == 120
        assert s.smoke_alpha == 0.075
        assert s.lamp_flicker == 0.05


class TestPathSettings:
    def test_from_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = PathSettings.from_root(root)
            assert paths.project_root == root
            assert paths.output_dir == root / "output"
            assert paths.temp_dir == root / "temp"

    def test_ensure_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = PathSettings.from_root(root)
            paths.ensure_dirs()
            assert paths.output_dir.exists()
            assert paths.temp_dir.exists()
            assert paths.assets_dir.exists()
            assert paths.logs_dir.exists()


class TestSettings:
    def test_load_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings.load(root)
            assert isinstance(settings.video, VideoSettings)
            assert isinstance(settings.font, FontSettings)
            assert isinstance(settings.paths, PathSettings)
            assert settings.paths.project_root == root

    def test_load_with_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings.load(root, debug=True, draft_mode=True)
            assert settings.debug is True
            assert settings.draft_mode is True


class TestGetSettings:
    def test_singleton_behavior(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s1 = get_settings(root)
            s2 = get_settings()
            assert s1 is s2

    def test_force_reload(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s1 = get_settings(root)
            s2 = get_settings(root, force_reload=True)
            # New instance created
            assert s1 is not s2

    def test_reset_settings(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            get_settings(root)
            reset_settings()
            # Should be able to get new settings
            s = get_settings(root)
            assert s is not None
