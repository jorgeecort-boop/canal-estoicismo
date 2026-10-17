"""Tests for video/composer module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.video import VideoComposer, CompositionResult, get_video_composer
from src.narrative import Scene, Story
from config import get_settings, reset_settings


class TestCineVignetteMap:
    """The vignette is a static per-scene layer; recomputing it per frame was
    a sqrt() over the whole 1080x1920 canvas inside the moviepy hot loop."""

    def test_shape_dtype_and_bounds(self):
        m = VideoComposer._cine_vignette_map(64, 48, 0.45)
        assert m.shape == (48, 64)
        assert str(m.dtype) == "float32"
        assert 0.72 <= float(m.min()) and float(m.max()) <= 1.0

    def test_matches_the_inline_formula(self):
        import numpy as np

        w, h, strength = 32, 20, 0.6
        yy, xx = np.ogrid[:h, :w]
        distance = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        expected = np.clip(1.0 - strength * np.maximum(distance - 0.15, 0.0), 0.72, 1.0)

        got = VideoComposer._cine_vignette_map(w, h, strength)
        assert np.allclose(got, expected, atol=1e-6)

    def test_center_is_bright_and_edges_darken(self):
        m = VideoComposer._cine_vignette_map(100, 100, 0.5)
        assert m[50, 50] == pytest.approx(1.0)
        assert m[0, 0] < m[50, 50]


class TestCompositionResult:
    def test_result_creation(self):
        result = CompositionResult(
            output_path=Path("/tmp/video.mp4"),
            duration=60.0,
            scenes_count=5,
            resolution=(1920, 1080),
            fps=30,
            file_size=1024 * 1024,
        )
        assert result.duration == 60.0
        assert result.scenes_count == 5

    def test_result_to_dict(self):
        result = CompositionResult(
            output_path=Path("/tmp/video.mp4"),
            duration=60.0,
            scenes_count=5,
            resolution=(1920, 1080),
            fps=30,
            file_size=1024 * 1024,
        )
        d = result.to_dict()
        assert d["duration"] == 60.0
        assert Path(d["output_path"]).name == "video.mp4"


class TestVideoComposer:
    @pytest.fixture
    def settings(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = get_settings(root)
            yield settings
        reset_settings()

    @pytest.fixture
    def composer(self, settings):
        return VideoComposer(settings)

    @pytest.fixture
    def sample_story(self):
        scenes = [
            Scene(1, "Overlay 1", "Voiceover 1", "Prompt 1", 5.0),
            Scene(2, "Overlay 2", "Voiceover 2", "Prompt 2", 5.0),
        ]
        return Story("test_theme", "Test Title", "Test Philosopher", scenes)

    def test_composer_initialization(self, composer):
        assert composer.video_settings.width == 1920
        assert composer.font_settings.size == 72
        assert composer.kenburns_settings.enabled is True
        assert composer._temp_dir.exists()

    def test_cine_filter_off_by_default(self, composer):
        assert composer._build_cine_filter() == ""

    def test_visual_engine_off_by_default(self, composer):
        assert composer.visual_engine_settings.enabled is False
        frame = __import__("numpy").zeros((8, 8, 3), dtype="uint8")
        assert (composer._ambient_fx(frame, 0.0, 1, ()) == frame).all()

    def test_xfade_round_robin(self, composer):
        assert [composer._xfade_transition(i) for i in range(6)] == [
            "fade", "dissolve", "wipeleft", "slideright", "fade", "dissolve"
        ]

    def test_varied_xfade_single_clip_copies(self, composer, tmp_path):
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")
        out = tmp_path / "out.mp4"
        assert composer._concatenate_videos_varied_xfade([src], out) is True
        assert out.exists()

    def test_transitions_off_by_default(self, composer):
        assert composer.settings.transitions.varied is False

    @patch("src.video.composer.subprocess.run")
    def test_pad_scene_audio_appends_pause(self, mock_run, composer, tmp_path):
        def _make_padded(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"padded")
            return MagicMock(returncode=0)

        mock_run.side_effect = _make_padded
        src = tmp_path / "scene.mp3"
        src.write_bytes(b"fake")
        out = composer._pad_scene_audio(src, extra_seconds=0.6)
        # Default audio codec is aac -> .m4a. Writing aac into a .mp3 file is
        # what made ffmpeg fail on Colab ("Audio padding failed" per scene).
        assert out.name == "scene.pad.m4a"
        assert out.exists()
        cmd = mock_run.call_args[0][0]
        af = cmd[cmd.index("-af") + 1]
        assert "apad=pad_dur=0.6" in af

    @patch("src.video.composer.subprocess.run")
    def test_pad_scene_audio_follows_an_mp3_codec(self, mock_run, composer, tmp_path):
        from dataclasses import replace

        def _make_padded(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"padded")
            return MagicMock(returncode=0)

        mock_run.side_effect = _make_padded
        composer.video_settings = replace(
            composer.video_settings, audio_codec="libmp3lame"
        )
        src = tmp_path / "scene.mp3"
        src.write_bytes(b"fake")
        out = composer._pad_scene_audio(src, extra_seconds=0.6)
        assert out.name == "scene.pad.mp3"

    def test_engine_moviepy_by_default(self, composer):
        assert composer.video_settings.engine == "moviepy"

    def test_kenburns_vertical_directions(self, composer):
        # In vertical format (height > width), pan_up and pan_down are supported
        kb_up = composer._build_kenburns_filter(540, 960, 5.0, direction="pan_up")
        assert "pan_up" not in kb_up  # filter string contains expressions
        assert "ih-ih/zoom" in kb_up

        kb_down = composer._build_kenburns_filter(540, 960, 5.0, direction="pan_down")
        assert "ih-ih/zoom" in kb_down

    def test_resolve_video_codec_default(self, composer):
        codec, preset, extra = composer._resolve_video_codec()
        assert codec == "libx264"
        assert preset == "slow"
        assert extra == []

    def test_resolve_video_codec_nvenc_success(self, composer):
        from dataclasses import replace
        c_nvenc = VideoComposer(replace(composer.settings, video=replace(composer.video_settings, nvenc=True)))
        with patch.object(c_nvenc, "_is_nvenc_available", return_value=True):
            codec, preset, extra = c_nvenc._resolve_video_codec()
            assert codec == "h264_nvenc"
            assert preset == "p5"
            assert "-tune" in extra and "hq" in extra

    def test_resolve_video_codec_nvenc_fallback(self, composer):
        from dataclasses import replace
        c_nvenc = VideoComposer(replace(composer.settings, video=replace(composer.video_settings, nvenc=True)))
        with patch.object(c_nvenc, "_is_nvenc_available", return_value=False):
            codec, preset, extra = c_nvenc._resolve_video_codec()
            assert codec == "libx264"
            assert preset == "slow"
            assert extra == []

    def test_compose_ffmpeg_engine_skips_moviepy(self, composer, tmp_path):
        from dataclasses import replace
        from src.narrative import Scene, Story
        settings = replace(composer.settings, video=replace(composer.video_settings, engine="ffmpeg"))
        composer_ffmpeg = type(composer)(settings)
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake-img")
        aud = tmp_path / "aud.mp3"
        aud.write_bytes(b"fake-aud")
        scene = Scene(1, "Overlay", "Voiceover text here", "Prompt",
                      estimated_duration=5.0, audio_path=aud, image_path=img)
        story = Story("theme", "Title", "Phil", [scene])
        with patch.object(composer_ffmpeg, "_compose_scene_ffmpeg", return_value=True) as mock_ff, \
             patch.object(composer_ffmpeg, "_apply_kenburns_and_text") as mock_mp, \
             patch.object(composer_ffmpeg, "_concatenate_videos_ffmpeg", return_value=True), \
             patch.object(composer_ffmpeg, "_get_video_duration", return_value=5.0):
            out = tmp_path / "final.mp4"
            out.write_bytes(b"x")
            with patch.object(Path, "exists", return_value=True):
                result = composer_ffmpeg.compose(story, output_path=out)
        mock_ff.assert_called_once()
        mock_mp.assert_not_called()
        assert result.scenes_count == 1

    def test_scene_filter_renders_zoompan_at_the_canvas(self, composer):
        """The filter used to zoompan at 540x960 and upscale afterwards, which
        doubled every integer crop step on screen: a slow pan froze for five
        frames and then jumped one pixel. Measured frame by frame on the
        delivered reel (kb_diag2/3), so zoompan now runs straight to the canvas
        over a pre-upscaled still."""
        fc = composer._build_scene_filter_complex(1920, 1080, 10.0, "zoom_in", "Hi")
        assert "s=1920x1080" in fc            # zoompan straight to the canvas
        assert "scale=1920:1080" not in fc    # no half-res + upscale any more
        assert fc.index("zoompan") < fc.index("drawtext")  # crisp text

    def test_prepared_still_disabled_returns_the_original(self, composer, tmp_path):
        from dataclasses import replace

        composer.kenburns_settings = replace(composer.kenburns_settings, upscale=1)
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        assert composer._prepared_still(img, 1) == img

    @patch("src.video.composer.subprocess.run")
    def test_prepared_still_upscales_once(self, mock_run, composer, tmp_path):
        def _write(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"big")
            return MagicMock(returncode=0)

        mock_run.side_effect = _write
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        out = composer._prepared_still(img, 3)
        assert out.name == "still_003_2x.png"
        cmd = mock_run.call_args[0][0]
        vf = cmd[cmd.index("-vf") + 1]
        assert "scale=3840:2160" in vf    # canvas 1920x1080, doubled
        assert "lanczos" in vf

        mock_run.reset_mock()
        assert composer._prepared_still(img, 3) == out
        mock_run.assert_not_called()      # reused, no second encode

    def test_wrap_to_canvas_fits_every_line(self, composer):
        """44 characters measured 1283 px at size 72 inside a 1080 px canvas,
        which is why the reel's subtitles were clipped on both sides."""
        text = "el estoicismo, el cosmos no comete error"
        size = composer.font_settings.size
        max_px = 1080 * composer.font_settings.width_fraction
        wrapped = composer._wrap_to_canvas(text, size, max_px)
        lines = wrapped.split("\n")
        assert len(lines) >= 2
        assert all(composer._measure_text(line, size) <= max_px for line in lines)
        assert " ".join(lines) == text     # nothing dropped

    def test_fit_subtitle_shrinks_instead_of_dropping_words(self, composer):
        # Font metrics stubbed so the check does not depend on the installed face.
        with patch.object(VideoComposer, "_measure_text",
                          lambda self, t, s: len(t) * s * 0.5):
            text = " ".join(["palabra"] * 8)
            wrapped, size = composer._fit_subtitle(text, 1080, composer.font_settings.size)
        assert size < composer.font_settings.size
        assert wrapped.count("\n") + 1 <= composer.font_settings.max_lines
        assert " ".join(wrapped.split()) == text

    def test_drawtext_splits_wrapped_lines_into_their_own_filters(self, composer):
        text = "el estoicismo, el cosmos no comete error"
        dt = composer._build_single_drawtext(text, 1080, 1920, 0.0, 3.0)
        wrapped, _ = composer._fit_subtitle(text, 1080, composer.font_settings.size)
        expected = len([line for line in wrapped.split("\n") if line.strip()])
        # drawtext does not wrap and its \n escape renders a literal "n", so
        # each line is its own filter.
        assert dt.count("drawtext=") == expected >= 2
        assert r"\n" not in dt
        assert "estoicismo" in dt and "error" in dt

    @patch("src.video.composer.subprocess.run")
    def test_get_video_duration_success(self, mock_run, composer):
        mock_run.return_value = MagicMock(returncode=0, stdout="45.5\n")
        duration = composer._get_video_duration(Path("/tmp/video.mp4"))
        assert duration == 45.5

    @patch("src.video.composer.subprocess.run")
    def test_get_video_duration_failure(self, mock_run, composer):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        duration = composer._get_video_duration(Path("/tmp/video.mp4"))
        assert duration == 0.0

    def test_build_kenburns_filter_zoom_in(self, composer):
        filter_str = composer._build_kenburns_filter(1920, 1080, 10.0, "zoom_in")
        assert "zoompan" in filter_str
        assert "zoom=1+(1.24-1)*on/300" in filter_str
        assert "d=300" in filter_str  # 10s * 30fps

    def test_build_kenburns_filter_zoom_out(self, composer):
        filter_str = composer._build_kenburns_filter(1920, 1080, 10.0, "zoom_out")
        assert "zoompan" in filter_str
        assert "zoom=1.24-(1.24-1)*on/300" in filter_str

    def test_build_kenburns_filter_pan_left(self, composer):
        filter_str = composer._build_kenburns_filter(1920, 1080, 10.0, "pan_left")
        assert "zoompan" in filter_str
        assert "zoom=1.24" in filter_str
        assert "(iw-iw/zoom)*(1-on/300" in filter_str

    def test_build_text_filter_bottom(self, composer):
        filter_str = composer._build_text_filter("Test text", 1920, 1080, 10.0)
        assert "drawtext" in filter_str
        assert "fontsize=72" in filter_str
        assert "fontcolor=0xF5F0E1" in filter_str
        assert "bordercolor=0x1A1A1A" in filter_str
        assert "borderw=4" in filter_str
        assert "h-text_h-240" in filter_str  # bottom position raised 120px

    def test_build_text_filter_special_chars(self, composer):
        # Test escaping
        filter_str = composer._build_text_filter("Test: 50% 'quote'", 1920, 1080, 5.0)
        assert "Test\\: 50\\% '\\''quote" in filter_str or "quote" in filter_str

    def test_overlay_text_wraps_and_caps_lines(self, composer):
        wrapped = composer._format_overlay_text(
            "Esta frase es demasiado larga para una sola línea y debe quedar dentro del reel "
            "aunque el texto original continúe mucho más allá del espacio disponible en pantalla"
        )
        lines = wrapped.splitlines()
        assert len(lines) == 2
        assert all(len(line) <= composer.font_settings.max_chars_per_line for line in lines)
        assert wrapped.endswith("…")

    def test_text_filter_splits_into_timed_cards(self, composer):
        text = "Una frase larga que debe saltar de línea para no salirse del video vertical"
        filter_str = composer._build_text_filter(text, 1080, 1920, 5.0)
        # Two timed cards, each one or more drawtext line filters.
        assert "between(t,0.0,2.5)" in filter_str
        assert "between(t,2.5,5.0)" in filter_str
        assert filter_str.count("drawtext=") >= 2

    def test_subtitles_follow_narration_word_budget(self, composer):
        cards = composer._timed_subtitle_cards(
            "Primero acepta lo que ocurre. Después elige con serenidad tu respuesta.",
            10.0,
        )
        assert len(cards) >= 2
        assert cards[0][1] == 0.0
        assert cards[-1][2] == 10.0
        assert all(end > start for _, start, end in cards)

    def test_subtitle_cards_never_drop_narration_words(self, composer):
        """Formatting each card used to cap it by character count and hide the
        tail behind an ellipsis, so part of the narration was simply not shown
        ("la misma manera los obstaculos las traiciones…")."""
        narration = (
            "de la misma manera los obstaculos las traiciones y las perdidas "
            "son la materia de la virtud y el combustible del caracter"
        )
        cards = composer._timed_subtitle_cards(narration, 14.0, width=1080)
        rebuilt = " ".join(card for card, _, _ in cards)
        assert rebuilt == " ".join(narration.split())
        assert "…" not in rebuilt

    def test_cards_are_budgeted_from_the_measured_font(self, composer):
        """The character budget was blind to the font: 45 characters of Georgia
        measured 1283 px in a 1080 px canvas. Budgeted by pixels, a card needs
        no shrinking to fit its two lines."""
        narration = " ".join(["palabra"] * 60)
        cards = composer._timed_subtitle_cards(narration, 30.0, width=1080)
        assert len(cards) >= 2
        for card, _, _ in cards:
            wrapped, size = composer._fit_subtitle(card, 1080, composer.font_settings.size)
            assert size == composer.font_settings.size
            assert wrapped.count("\n") + 1 <= composer.font_settings.max_lines

    @patch("src.video.composer.subprocess.run")
    def test_compose_scene_ffmpeg_success(self, mock_run, composer):
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "img.jpg"
            img.write_bytes(b"fake")
            aud = Path(tmp) / "aud.mp3"
            aud.write_bytes(b"fake")
            out = Path(tmp) / "scene.mp4"

            result = composer._compose_scene_ffmpeg(img, aud, "Test overlay", out, 5.0, 1)
            assert result is True
            # First call pre-upscales the still, second encodes the scene.
            assert mock_run.call_count == 2
            scene_cmd = mock_run.call_args_list[-1][0][0]
            assert "-filter_complex" in scene_cmd
            assert scene_cmd[-1] == str(out)

    @patch("src.video.composer.subprocess.run")
    def test_compose_scene_ffmpeg_failure(self, mock_run, composer):
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")

        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "img.jpg"
            img.write_bytes(b"fake")
            aud = Path(tmp) / "aud.mp3"
            aud.write_bytes(b"fake")
            out = Path(tmp) / "scene.mp4"

            result = composer._compose_scene_ffmpeg(img, aud, "Test", out, 5.0, 1)
            assert result is False

    def test_concatenate_joins_audio_sequentially(self, composer):
        """Narration follows the video timeline (overlap discounted)."""
        import sys
        import types

        starts = []

        class FakeAudio:
            duration = 5.0

            def with_start(self, offset):
                starts.append(offset)
                return self

        class FakeClip:
            audio = FakeAudio()

            def close(self):
                pass

        final = MagicMock()
        final.with_audio.side_effect = lambda track: final
        final.audio = None

        fake_moviepy = types.ModuleType("moviepy")
        fake_moviepy.VideoFileClip = lambda p: FakeClip()
        fake_moviepy.concatenate_videoclips = lambda clips, **kwargs: final
        fake_moviepy.CompositeVideoClip = MagicMock
        captured = {}

        class FakeCompositeAudio:
            def __init__(self, parts):
                captured["parts"] = parts

        fake_moviepy.CompositeAudioClip = FakeCompositeAudio

        with tempfile.TemporaryDirectory() as tmp:
            scene1 = Path(tmp) / "scene1.mp4"
            scene1.write_bytes(b"fake")
            scene2 = Path(tmp) / "scene2.mp4"
            scene2.write_bytes(b"fake")
            out = Path(tmp) / "final.mp4"
            with patch.dict(sys.modules, {"moviepy": fake_moviepy}):
                assert composer._concatenate_videos([scene1, scene2], out) is True
        assert starts == [0.0, 4.2]  # 5.0s scenes minus 0.8s crossfade overlap
        assert len(captured["parts"]) == 2

    @patch("src.video.composer.subprocess.run")
    def test_concatenate_videos_success(self, mock_run, composer):
        mock_run.return_value = MagicMock(returncode=0)

        with tempfile.TemporaryDirectory() as tmp:
            scene1 = Path(tmp) / "scene1.mp4"
            scene1.write_bytes(b"fake")
            scene2 = Path(tmp) / "scene2.mp4"
            scene2.write_bytes(b"fake")
            out = Path(tmp) / "final.mp4"

            result = composer._concatenate_videos([scene1, scene2], out)
            assert result is True

    @patch("src.video.composer.subprocess.run")
    def test_concatenate_videos_failure(self, mock_run, composer):
        mock_run.return_value = MagicMock(returncode=1)

        with tempfile.TemporaryDirectory() as tmp:
            scene1 = Path(tmp) / "scene1.mp4"
            scene1.write_bytes(b"fake")
            out = Path(tmp) / "final.mp4"

            result = composer._concatenate_videos([scene1], out)
            assert result is False

    @patch("src.video.composer.VideoComposer._apply_kenburns_and_text")
    @patch("src.video.composer.VideoComposer._concatenate_videos")
    @patch("src.video.composer.VideoComposer._get_video_duration")
    @patch("src.video.composer.VideoComposer._create_placeholder_image")
    def test_compose_full_pipeline(
        self, mock_placeholder, mock_duration, mock_concat, mock_apply,
        composer, sample_story
    ):
        # Setup mocks
        mock_apply.return_value = True
        mock_concat.return_value = True
        mock_duration.return_value = 10.0

        with tempfile.TemporaryDirectory() as tmp:
            # Create fake image and audio files
            for scene in sample_story.scenes:
                img = Path(tmp) / f"img_{scene.scene_number}.jpg"
                img.write_bytes(b"fake")
                aud = Path(tmp) / f"aud_{scene.scene_number}.mp3"
                aud.write_bytes(b"fake")
                scene.image_path = img
                scene.audio_path = aud

            output = Path(tmp) / "output.mp4"
            output.write_bytes(b"fake video")  # For file_size

            progress_calls = []
            def progress_cb(current, total, msg):
                progress_calls.append((current, total, msg))

            result = composer.compose(sample_story, output_path=output, progress_callback=progress_cb)

        assert result.output_path == output
        assert result.duration == 10.0
        assert result.scenes_count == 2
        assert result.resolution == (1920, 1080)
        assert mock_apply.call_count == 2
        assert mock_concat.call_count == 1
        assert len(progress_calls) == 3

    @patch("src.video.composer.VideoComposer._apply_kenburns_and_text")
    def test_compose_missing_image_creates_placeholder(self, mock_apply, composer, sample_story):
        mock_apply.return_value = True

        with tempfile.TemporaryDirectory() as tmp:
            # Only provide audio, not image
            for scene in sample_story.scenes:
                aud = Path(tmp) / f"aud_{scene.scene_number}.mp3"
                aud.write_bytes(b"fake")
                scene.audio_path = aud
                scene.image_path = Path("/nonexistent.jpg")

            with patch.object(composer, "_concatenate_videos", return_value=True):
                with patch.object(composer, "_get_video_duration", return_value=5.0):
                    output = Path(tmp) / "output.mp4"
                    output.write_bytes(b"fake")

                    result = composer.compose(sample_story, output_path=output)

            # Should have created placeholder images
            assert mock_apply.call_count == 2

    def test_compose_missing_audio_raises(self, composer, sample_story):
        with tempfile.TemporaryDirectory() as tmp:
            for scene in sample_story.scenes:
                img = Path(tmp) / f"img_{scene.scene_number}.jpg"
                img.write_bytes(b"fake")
                scene.image_path = img
                scene.audio_path = Path("/nonexistent.mp3")

            with pytest.raises(ValueError, match="Missing audio"):
                composer.compose(sample_story, output_path=Path(tmp) / "out.mp4")

    def test_compose_draft_mode_limits_duration(self, composer, sample_story):
        with tempfile.TemporaryDirectory() as tmp:
            for scene in sample_story.scenes:
                img = Path(tmp) / f"img_{scene.scene_number}.jpg"
                img.write_bytes(b"fake")
                aud = Path(tmp) / f"aud_{scene.scene_number}.mp3"
                aud.write_bytes(b"fake")
                scene.image_path = img
                scene.audio_path = aud
                scene.estimated_duration = 10.0  # Long duration

            with patch.object(composer, "_apply_kenburns_and_text", return_value=True) as mock_compose:
                with patch.object(composer, "_concatenate_videos", return_value=True):
                    with patch.object(composer, "_get_video_duration", return_value=4.0):
                        output = Path(tmp) / "output.mp4"
                        output.write_bytes(b"fake")

                        result = composer.compose(sample_story, output_path=output, draft_mode=True)

            # Check that draft mode was passed (duration limited to 2s)
            for call in mock_compose.call_args_list:
                args, kwargs = call
                duration = args[4]  # 5th argument is duration
                assert duration <= 2.0


class TestGetVideoComposer:
    def test_singleton(self):
        reset_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = get_settings(root)
            c1 = get_video_composer(settings)
            c2 = get_video_composer()
            assert c1 is c2
        reset_settings()
