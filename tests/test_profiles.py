"""Tests for hardware profiles (config/profiles.py) and the fixes that came
with them: the reel-aware upscale target and the profile-aware SDXL cache key.

These are the tests that protect the Colab A100 path. If one of them goes red,
a notebook would burn compute units on a misconfigured run.
"""

import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from config import get_settings, reset_settings
from config.profiles import (
    A100_MIN_VRAM_GB,
    FORMAT_CANVAS,
    PROFILE_CHOICES,
    SDXL_FULL_A100,
    apply_profile,
    assert_a100_ready,
    describe_profile,
    detect_profile,
)
from src.media import ImageManager
from src.video import VideoComposer


@pytest.fixture
def settings():
    reset_settings()
    s = get_settings()
    yield s
    reset_settings()


# ── apply_profile ────────────────────────────────────────────────────

class TestApplyProfile:
    def test_none_is_a_noop(self, settings):
        assert apply_profile(settings, "none") is settings

    def test_unknown_profile_raises(self, settings):
        with pytest.raises(ValueError, match="Unknown profile"):
            apply_profile(settings, "quantum")

    def test_reel_canvas(self, settings):
        s = apply_profile(settings, "t4", fmt="reel")
        assert (s.video.width, s.video.height) == (1080, 1920)
        assert (s.image.width, s.image.height) == (1080, 1920)
        assert (s.image.sdxl_width, s.image.sdxl_height) == (832, 1216)
        assert s.font.size == 54
        assert s.font.margin_bottom == 440
        assert s.font.max_chars_per_line == 26
        assert s.video.aspect_ratio == "9:16"

    def test_landscape_canvas(self, settings):
        s = apply_profile(settings, "t4", fmt="landscape")
        assert (s.video.width, s.video.height) == (1920, 1080)
        assert (s.image.sdxl_width, s.image.sdxl_height) == (1216, 832)
        assert s.video.aspect_ratio == "16:9"

    def test_t4_keeps_the_turbo_model(self, settings):
        """The T4 profile must not silently upgrade the model."""
        s = apply_profile(settings, "t4", fmt="reel")
        assert s.image.sdxl_model == "stabilityai/sdxl-turbo"
        assert s.image.sdxl_steps == 2
        assert s.image.sdxl_refiner == ""
        assert s.image.sdxl_hires is False
        assert s.image.sdxl_cpu_offload is True
        assert s.video.threads == 4
        assert s.video.preset == "slow"

    def test_a100_upgrades_to_sdxl_full(self, settings):
        s = apply_profile(settings, "a100", fmt="reel")
        assert s.image.sdxl_model == "stabilityai/stable-diffusion-xl-base-1.0"
        assert s.image.sdxl_refiner == "stabilityai/stable-diffusion-xl-refiner-1.0"
        assert s.image.sdxl_steps == 30
        assert s.image.sdxl_guidance == 6.5
        assert s.image.sdxl_hires is True
        assert s.image.sdxl_cpu_offload is False
        # A100 has no NVENC: encode on CPU with every vCPU available.
        assert s.video.nvenc is False
        assert s.video.threads == 0
        assert s.video.preset == "medium"

    def test_a100_keeps_the_reel_canvas(self, settings):
        s = apply_profile(settings, "a100", fmt="reel")
        assert (s.video.width, s.video.height) == (1080, 1920)
        assert (s.image.sdxl_width, s.image.sdxl_height) == (832, 1216)

    def test_a100_never_uses_esrgan(self, settings):
        """Real-ESRGAN/basicsr is the classic Colab breakage; the hires pass
        replaces it, so the profile must leave upscale_enabled off."""
        s = apply_profile(settings, "a100", fmt="reel")
        assert s.image.upscale_enabled is False

    def test_explicit_flag_still_wins_over_profile(self, settings):
        """main() applies the profile first, so a later CLI override survives."""
        s = apply_profile(settings, "a100", fmt="reel")
        forced = replace(s, image=replace(s.image, upscale_enabled=True))
        assert forced.image.upscale_enabled is True

    def test_auto_falls_back_to_detected(self, settings):
        s = apply_profile(settings, "auto", fmt="reel")
        assert (s.video.width, s.video.height) == (1080, 1920)

    def test_colab_is_an_alias_for_auto(self, settings):
        assert apply_profile(settings, "colab", fmt="reel").video.width == 1080

    def test_original_settings_untouched(self, settings):
        """Settings are frozen dataclasses: apply_profile must not mutate."""
        before = settings.image.sdxl_model
        apply_profile(settings, "a100", fmt="reel")
        assert settings.image.sdxl_model == before


# ── detect_profile ───────────────────────────────────────────────────

class TestDetectProfile:
    def test_returns_a_known_profile(self):
        assert detect_profile() in ("cpu", "t4", "a100")

    def test_threshold_is_sane(self):
        assert 20 <= A100_MIN_VRAM_GB <= 80

    def test_every_format_has_a_canvas(self):
        for fmt, canvas in FORMAT_CANVAS.items():
            assert len(canvas["video"]) == 2
            assert len(canvas["sdxl"]) == 2
            # SDXL was trained at ~1 MP; staying near it is what keeps
            # compositions from collapsing into duplicated limbs.
            sw, sh = canvas["sdxl"]
            megapixels = (sw * sh) / 1_000_000
            assert 0.9 <= megapixels <= 1.1, f"{fmt}: {sw}x{sh} is {megapixels:.2f} MP"


# ── describe / assert ────────────────────────────────────────────────

class TestDescribeAndAssert:
    def test_describe_reports_the_channel_voice(self, settings):
        described = describe_profile(apply_profile(settings, "a100", fmt="reel"))
        assert described["tts"] == "edge-tts:es-MX-JorgeNeural"

    def test_describe_has_the_contract_keys(self, settings):
        described = describe_profile(settings)
        for key in (
            "video", "encoder", "threads", "engine", "image_provider",
            "sdxl_model", "sdxl_refiner", "sdxl_hires", "tts",
        ):
            assert key in described

    def test_a100_profile_is_ready(self, settings):
        assert assert_a100_ready(apply_profile(settings, "a100", fmt="reel")) == []

    def test_a100_check_catches_each_misconfiguration(self, settings):
        base = apply_profile(settings, "a100", fmt="reel")
        broken_cases = [
            replace(base, video=replace(base.video, threads=4)),
            replace(base, video=replace(base.video, nvenc=True)),
            replace(base, image=replace(base.image, sdxl_model="stabilityai/sdxl-turbo")),
            replace(base, image=replace(base.image, sdxl_refiner="")),
            replace(base, image=replace(base.image, sdxl_steps=2)),
            replace(base, image=replace(base.image, sdxl_hires=False)),
            replace(base, image=replace(base.image, sdxl_cpu_offload=True)),
            replace(base, image=replace(base.image, upscale_enabled=True)),
        ]
        for broken in broken_cases:
            assert assert_a100_ready(broken), "a known-bad config passed the gate"

    def test_a100_dict_is_complete(self):
        for key in ("sdxl_model", "sdxl_refiner", "sdxl_steps", "sdxl_hires",
                    "sdxl_cpu_offload", "upscale_enabled"):
            assert key in SDXL_FULL_A100

    def test_profile_choices_are_documented(self):
        assert "auto" in PROFILE_CHOICES and "none" in PROFILE_CHOICES


# ── upscale target fix ───────────────────────────────────────────────

class TestUpscaleTarget:
    def _manager(self, width, height):
        s = replace(
            self.settings,
            image=replace(
                self.settings.image, width=width, height=height, upscale_enabled=True
            ),
        )
        return ImageManager(s)

    @pytest.fixture(autouse=True)
    def _bind(self, settings):
        self.settings = settings

    def test_reel_upscale_keeps_the_vertical_canvas(self):
        """Regression: a hardcoded 1920x1080 turned every reel landscape."""
        manager = self._manager(1080, 1920)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.jpg"
            Image.new("RGB", (320, 480), (20, 20, 40)).save(path, "JPEG")
            out = manager._maybe_upscale(path)
            with Image.open(out) as img:
                assert img.size == (1080, 1920)

    def test_landscape_upscale_keeps_the_horizontal_canvas(self):
        manager = self._manager(1920, 1080)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.jpg"
            Image.new("RGB", (320, 180), (20, 20, 40)).save(path, "JPEG")
            out = manager._maybe_upscale(path)
            with Image.open(out) as img:
                assert img.size == (1920, 1080)

    def test_no_upscale_when_disabled(self, settings):
        s = replace(settings, image=replace(settings.image, upscale_enabled=False))
        manager = ImageManager(s)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.jpg"
            Image.new("RGB", (320, 480), (20, 20, 40)).save(path, "JPEG")
            assert manager._maybe_upscale(path) == path


# ── profile-aware SDXL cache ─────────────────────────────────────────

class TestSdxlCacheKey:
    def test_switching_profile_changes_the_cache_path(self, settings):
        """Otherwise an old sdxl-turbo image would be served for an A100 run."""
        reel = apply_profile(settings, "t4", fmt="reel")
        a100 = apply_profile(settings, "a100", fmt="reel")
        turbo_manager = ImageManager(reel)
        a100_manager = ImageManager(a100)

        with patch.object(ImageManager, "_verify_image", return_value=True):
            turbo_asset = turbo_manager._generate_sdxl_local("a stoic bust", 1)
            a100_asset = a100_manager._generate_sdxl_local("a stoic bust", 1)

        assert turbo_asset.path != a100_asset.path

    def test_same_profile_is_reproducible(self, settings):
        reel = apply_profile(settings, "t4", fmt="reel")
        with patch.object(ImageManager, "_verify_image", return_value=True):
            first = ImageManager(reel)._generate_sdxl_local("a stoic bust", 1)
            second = ImageManager(reel)._generate_sdxl_local("a stoic bust", 1)
        assert first.path == second.path


# ── composer threads ─────────────────────────────────────────────────

class TestComposerThreads:
    def test_default_threads(self, settings):
        composer = VideoComposer(settings)
        assert composer._encoder_threads() == 4

    def test_zero_means_let_ffmpeg_decide(self, settings):
        s = replace(settings, video=replace(settings.video, threads=0))
        assert VideoComposer(s)._encoder_threads() is None

    def test_a100_profile_uses_auto_threads(self, settings):
        a100 = apply_profile(settings, "a100", fmt="reel")
        assert VideoComposer(a100)._encoder_threads() is None
