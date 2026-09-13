"""Tests for audio/tts_engine module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.audio import TTSEngine, TTSResult, get_tts_engine
from config import get_settings, reset_settings


class TestTTSResult:
    def test_tts_result_creation(self):
        result = TTSResult(
            audio_path=Path("/tmp/test.mp3"),
            duration=10.5,
            text="Test text",
            voice="es-ES-AlvaroNeural",
            provider="edge-tts",
        )
        assert result.duration == 10.5
        assert result.text == "Test text"

    def test_tts_result_to_dict(self):
        result = TTSResult(
            audio_path=Path("/tmp/test.mp3"),
            duration=10.5,
            text="Test",
            voice="voice",
            provider="edge-tts",
        )
        d = result.to_dict()
        assert d["duration"] == 10.5
        assert Path(d["audio_path"]).name == "test.mp3"


class TestTTSEngine:
    @pytest.fixture
    def settings(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = get_settings(root)
            yield settings
        reset_settings()

    @pytest.fixture
    def engine(self, settings):
        return TTSEngine(settings)

    def test_engine_initialization(self, engine):
        assert engine.tts_settings.provider == "edge-tts"
        assert engine._cache_dir.exists()

    def test_get_cache_key(self, engine):
        key1 = engine._get_cache_key("Hello world", "voice1", "-10%")
        key2 = engine._get_cache_key("Hello world", "voice1", "-10%")
        key3 = engine._get_cache_key("Different", "voice1", "-10%")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 16

    def test_get_cached_path(self, engine):
        path = engine._get_cached_path("abc123")
        assert path.name == "abc123.mp3"
        assert path.parent == engine._cache_dir

    @patch("src.audio.tts_engine.subprocess.run")
    def test_get_audio_duration_success(self, mock_run, engine):
        mock_run.return_value = MagicMock(returncode=0, stdout="15.5\n")
        duration = engine._get_audio_duration(Path("/tmp/test.mp3"))
        assert duration == 15.5

    @patch("src.audio.tts_engine.subprocess.run")
    def test_get_audio_duration_failure(self, mock_run, engine):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        duration = engine._get_audio_duration(Path("/tmp/test.mp3"))
        assert duration == 0.0

    @pytest.mark.asyncio
    @patch("edge_tts.Communicate.save")
    async def test_generate_edge_tts_success(self, mock_save, engine):
        mock_save.return_value = None

        with patch.object(engine, "_get_audio_duration", return_value=10.0):
            duration = await engine._generate_edge_tts("Test text", Path("/tmp/out.mp3"))
            assert duration == 10.0

    @pytest.mark.asyncio
    @patch("edge_tts.Communicate.save")
    async def test_generate_edge_tts_failure(self, mock_save, engine):
        mock_save.side_effect = RuntimeError("edge-tts failed")

        with pytest.raises(RuntimeError, match="edge-tts failed"):
            await engine._generate_edge_tts("Test text", Path("/tmp/out.mp3"))

    @patch("gtts.gTTS", create=True)
    def test_generate_gtts(self, mock_gtts, engine):
        mock_tts = MagicMock()
        mock_gtts.return_value = mock_tts

        with patch.object(engine, "_get_audio_duration", return_value=8.0):
            output_path = Path("/tmp/out.mp3")
            output_path.write_bytes(b"fake")
            duration = engine._generate_gtts("Test text", output_path)
            assert duration == 8.0
            mock_tts.save.assert_called_once_with(str(output_path))

    @pytest.mark.asyncio
    async def test_generate_cached(self, engine):
        # Create a fake cached file with correct cache key
        # The generate method uses self.tts_settings.voice ("es-ES-AlvaroNeural") and self.tts_settings.rate ("-15%")
        cache_key = engine._get_cache_key("Test text", engine.tts_settings.voice, engine.tts_settings.rate)
        cached_path = engine._get_cached_path(cache_key)
        cached_path.write_bytes(b"fake audio")

        with patch.object(engine, "_get_audio_duration", return_value=5.0):
            with patch.object(engine, "_generate_edge_tts", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = 5.0
                result = await engine.generate("Test text", use_cache=True)

        assert result.audio_path == cached_path
        assert result.duration == 5.0
        assert result.provider == "edge-tts"
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.audio.tts_engine.TTSEngine._generate_edge_tts")
    async def test_generate_new(self, mock_gen, engine):
        mock_gen.return_value = 7.5

        output_path = engine.settings.paths.temp_dir / f"tts_{engine._get_cache_key('Test', 'v', 'r')}.mp3"
        output_path.write_bytes(b"audio")

        with patch.object(engine, "_get_audio_duration", return_value=7.5):
            result = await engine.generate("Test text", use_cache=False)

        assert result.duration == 7.5
        assert result.text == "Test text"

    @pytest.mark.asyncio
    async def test_generate_for_scene(self, engine):
        from src.narrative import Scene

        scene = Scene(
            scene_number=1,
            text_overlay="Test",
            voiceover_text="This is a test voiceover for the scene",
            image_prompt="Test prompt",
        )

        with patch.object(engine, "generate") as mock_gen:
            mock_result = TTSResult(
                audio_path=Path("/tmp/test.mp3"),
                duration=5.0,
                text=scene.voiceover_text,
                voice="voice",
                provider="edge-tts",
            )
            mock_gen.return_value = mock_result

            result = await engine.generate_for_scene(scene)

        assert scene.audio_path == Path("/tmp/test.mp3")
        assert scene.estimated_duration == 5.0
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_generate_for_story(self, engine):
        from src.narrative import Story, Scene

        scenes = [
            Scene(1, "O1", "V1", "P1"),
            Scene(2, "O2", "V2", "P2"),
        ]
        story = Story("theme", "Title", "Phil", scenes)

        with patch.object(engine, "generate_for_scene") as mock_gen:
            mock_gen.side_effect = [
                TTSResult(Path("/tmp/1.mp3"), 5.0, "V1", "v", "p"),
                TTSResult(Path("/tmp/2.mp3"), 6.0, "V2", "v", "p"),
            ]

            progress_calls = []
            def progress_cb(current, total, scene_num):
                progress_calls.append((current, total, scene_num))

            results = await engine.generate_for_story(story, progress_callback=progress_cb)

        assert len(results) == 2
        assert len(progress_calls) == 3  # 2 scenes + final
        assert progress_calls[-1] == (2, 2, None)


class TestGetTTSEngine:
    def test_singleton(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = get_settings(root)
            e1 = get_tts_engine(settings)
            e2 = get_tts_engine()
            assert e1 is e2
        reset_settings()