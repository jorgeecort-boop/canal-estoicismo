"""Tests for media/image_manager module — updated for MediaManager."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from io import BytesIO

import pytest
from PIL import Image

from src.media import ImageManager, ImageAsset, get_image_manager
from config import get_settings, reset_settings


def _make_valid_jpeg_bytes(width=4, height=4) -> bytes:
    """Return minimal valid JPEG bytes using PIL."""
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(30, 30, 60)).save(buf, format="JPEG")
    return buf.getvalue()


class TestImageAsset:
    def test_asset_creation(self):
        asset = ImageAsset(
            path=Path("/tmp/img.jpg"),
            prompt="Test prompt",
            source="local",
            width=1920,
            height=1080,
        )
        assert asset.path == Path("/tmp/img.jpg")
        assert asset.source == "local"

    def test_asset_to_dict(self):
        asset = ImageAsset(
            path=Path("/tmp/img.jpg"),
            prompt="Test",
            source="cache",
            width=1920,
            height=1080,
        )
        d = asset.to_dict()
        assert "/tmp/img.jpg" in d["path"] or "\\tmp\\img.jpg" in d["path"]
        assert d["source"] == "cache"


class TestMediaManager:
    @pytest.fixture
    def settings(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = get_settings(root)
            assets_dir = settings.paths.assets_dir / "images"
            assets_dir.mkdir(parents=True, exist_ok=True)
            yield settings
        reset_settings()

    @pytest.fixture
    def manager(self, settings):
        return ImageManager(settings)

    def test_manager_initialization(self, manager):
        assert manager._assets_dir.exists()

    def test_get_cache_key_deterministic(self, manager):
        k1 = manager._get_cache_key("Prompt one")
        k2 = manager._get_cache_key("Prompt one")
        k3 = manager._get_cache_key("Prompt two")
        assert k1 == k2
        assert k1 != k3
        assert len(k1) == 16

    def test_get_cached_path(self, manager):
        path = manager._get_cached_path("abc123")
        assert path.name == "abc123.jpg"
        assert path.parent == manager._assets_dir

    # ── _verify_image ────────────────────────────────────────────────
    def test_verify_image_missing(self, manager):
        assert manager._verify_image(Path("/nonexistent/path.jpg")) is False

    def test_verify_image_zero_bytes(self, manager, tmp_path):
        zero = tmp_path / "zero.jpg"
        zero.write_bytes(b"")
        assert manager._verify_image(zero) is False

    def test_verify_image_corrupt(self, manager, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image at all")
        assert manager._verify_image(bad) is False

    def test_verify_image_valid(self, manager, tmp_path):
        good = tmp_path / "good.jpg"
        good.write_bytes(_make_valid_jpeg_bytes())
        assert manager._verify_image(good) is True

    # ── clean_legacy_caches ──────────────────────────────────────────
    def test_clean_legacy_caches_removes_zero_byte(self, manager):
        bad = manager._assets_dir / "corrupt.jpg"
        bad.write_bytes(b"")
        manager.clean_legacy_caches()
        assert not bad.exists()

    def test_clean_legacy_caches_keeps_valid(self, manager):
        good = manager._assets_dir / "good.jpg"
        good.write_bytes(_make_valid_jpeg_bytes())
        manager.clean_legacy_caches()
        assert good.exists()

    # ── Pollinations mock: success ────────────────────────────────────
    @patch("src.media.image_manager.requests.get")
    def test_generate_pollinations_success(self, mock_get, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [_make_valid_jpeg_bytes()]
        mock_response.content = _make_valid_jpeg_bytes()
        mock_get.return_value = mock_response

        asset = manager._generate_pollinations("stoic philosopher", scene_id=1)
        assert asset is not None
        assert asset.source == "pollinations"
        assert asset.path.exists()

    # ── Pollinations mock: timeout / network failure ──────────────────
    @patch("src.media.image_manager.requests.get")
    def test_generate_pollinations_timeout_returns_none(self, mock_get, manager):
        """Pollinations timeout should return None (not raise), triggering fallback."""
        import requests as req
        mock_get.side_effect = req.exceptions.Timeout("timed out")

        asset = manager._generate_pollinations("any prompt", scene_id=1)
        assert asset is None

    # ── Placeholder fallback ─────────────────────────────────────────
    def test_create_placeholder_creates_valid_image(self, manager):
        asset = manager._create_placeholder("Test scene", scene_id=3)
        assert asset.source == "placeholder"
        assert asset.path.exists()
        assert asset.path.stat().st_size > 0
        assert manager._verify_image(asset.path)

    # ── get_image_for_prompt: uses valid cache, skips Pollinations ────
    @patch("src.media.image_manager.requests.get")
    def test_get_image_uses_valid_cache(self, mock_get, manager):
        cache_key = manager._get_cache_key("Cached prompt")
        cached_path = manager._get_cached_path(cache_key)
        cached_path.write_bytes(_make_valid_jpeg_bytes())

        asset = manager.get_image_for_prompt("Cached prompt", scene_id=1)
        assert asset.source == "cache"
        mock_get.assert_not_called()

    # ── get_image_for_prompt: corrupted cache forces Pollinations ─────
    @patch("src.media.image_manager.requests.get")
    def test_get_image_invalidates_corrupt_cache(self, mock_get, manager):
        cache_key = manager._get_cache_key("Bad cached prompt")
        cached_path = manager._get_cached_path(cache_key)
        cached_path.write_bytes(b"corrupt")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = _make_valid_jpeg_bytes()
        mock_response.iter_content.return_value = [_make_valid_jpeg_bytes()]
        mock_get.return_value = mock_response

        asset = manager.get_image_for_prompt("Bad cached prompt", scene_id=2)
        # Pollinations should have been called after cache was purged
        mock_get.assert_called_once()
        assert asset is not None

    # ── fetch_images_for_story (sequential) ──────────────────────────
    @patch("src.media.image_manager.requests.get")
    def test_fetch_images_for_story(self, mock_get, manager):
        from src.narrative import Story, Scene

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = _make_valid_jpeg_bytes()
        mock_response.iter_content.return_value = [_make_valid_jpeg_bytes()]
        mock_get.return_value = mock_response

        scenes = [Scene(1, "O1", "V1", "Prompt 1"), Scene(2, "O2", "V2", "Prompt 2")]
        story = Story("theme", "Title", "Phil", scenes)

        assets = manager.fetch_images_for_story(story)
        assert len(assets) == 2
        for scene in story.scenes:
            assert scene.image_path is not None

    # ── fetch_images_for_story_async ─────────────────────────────────
    @patch("src.media.image_manager.requests.get")
    def test_fetch_images_async(self, mock_get, manager):
        from src.narrative import Story, Scene

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = _make_valid_jpeg_bytes()
        mock_response.iter_content.return_value = [_make_valid_jpeg_bytes()]
        mock_get.return_value = mock_response

        scenes = [Scene(i, f"O{i}", f"V{i}", f"Prompt {i}") for i in range(1, 4)]
        story = Story("theme", "Title", "Phil", scenes)

        assets = asyncio.run(manager.fetch_images_for_story_async(story, max_concurrent=2))
        assert len(assets) == 3
        for scene in story.scenes:
            assert scene.image_path is not None


class TestGetImageManager:
    def test_singleton(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = get_settings(root)
            m1 = get_image_manager(settings)
            m2 = get_image_manager()
            assert m1 is m2
        reset_settings()