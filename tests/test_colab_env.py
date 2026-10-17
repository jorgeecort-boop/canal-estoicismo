"""Tests for the Colab helpers the notebooks rely on.

None of this needs a GPU: it checks the setup/verification logic that decides
whether a Colab run is allowed to spend compute units.
"""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from config.profiles import FORMAT_CANVAS
from scripts import colab_env
from src.media import ImageManager


class TestConstants:
    def test_canvas_matches_the_profiles(self):
        """One canvas definition, shared by the pipeline and the notebooks."""
        for fmt, canvas in FORMAT_CANVAS.items():
            assert colab_env.CANVAS[fmt] == tuple(canvas["video"])

    def test_temp_is_not_persisted(self):
        assert not any("temp" in d for d in colab_env.PERSISTED_DIRS)

    def test_image_cache_is_persisted(self):
        """The SDXL cache is the most expensive artefact: it must survive."""
        assert "assets/images" in colab_env.PERSISTED_DIRS

    def test_torch_is_never_pip_installed_by_default(self):
        """Reinstalling torch breaks Colab's CUDA-matched build."""
        assert not any(dep.lower().startswith("torch") for dep in colab_env.PYTHON_DEPS)

    def test_real_esrgan_is_not_installed(self):
        """basicsr/realesrgan is the classic Colab breakage; the hires pass
        replaced it, so the notebook must not pull it in."""
        joined = " ".join(colab_env.PYTHON_DEPS).lower()
        assert "realesrgan" not in joined and "basicsr" not in joined

    def test_gtts_is_not_pip_installed(self):
        """gTTS caps click<8.2, which forces pip to downgrade click and breaks
        Colab's preinstalled wandb/spacy. The channel voice is edge-tts and the
        gTTS import is lazy, so it stays out of the Colab dependency list."""
        assert not any(dep.lower().startswith("gtts") for dep in colab_env.PYTHON_DEPS)


class TestSdxlSharedComponents:
    """The SDXL two-stage loading must keep tokenizers/encoders aligned.

    Real failure this guards: sharing the base's ``text_encoder`` with the
    refiner repo (which declares ``text_encoder``/``tokenizer`` as null) makes
    diffusers' ``encode_prompt`` zip 1 tokenizer against 2 encoders, so the
    negative pooled embedding is never computed and the refiner dies with
    ``'NoneType' object has no attribute 'repeat'``.
    """

    @staticmethod
    def _stack():
        from src.media.image_manager import _SDXLStack

        stack = _SDXLStack()
        stack.base = MagicMock()  # components all mocked; only presence matters
        return stack

    def test_refiner_gets_the_official_pattern(self):
        shared = ImageManager._sdxl_shared(self._stack(), "refiner")
        assert set(shared) == {"text_encoder_2", "vae"}
        assert "text_encoder" not in shared and "tokenizer" not in shared

    def test_base_repo_img2img_gets_the_full_aligned_set(self):
        shared = ImageManager._sdxl_shared(self._stack(), "img2img")
        assert set(shared) == {
            "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae"
        }

    def test_essentials_are_installed(self):
        joined = " ".join(colab_env.PYTHON_DEPS).lower()
        for package in ("diffusers", "transformers", "accelerate", "moviepy", "edge-tts"):
            assert package in joined

    def test_secret_keys_cover_the_llm_chain(self):
        assert "GEMINI_API_KEY" in colab_env.SECRET_KEYS
        assert "OPENROUTER_API_KEY" in colab_env.SECRET_KEYS


class TestGate:
    def test_pass_does_not_raise(self, capsys):
        colab_env.gate("something", True, "all good")
        assert "[PASS] something" in capsys.readouterr().out

    def test_fail_raises(self):
        with pytest.raises(SystemExit):
            colab_env.gate("something", False, "broken")


class TestLinkPersistence:
    @staticmethod
    def _can_symlink(tmp: Path) -> bool:
        try:
            (tmp / "probe_target").mkdir()
            (tmp / "probe_link").symlink_to(tmp / "probe_target", target_is_directory=True)
            return True
        except (OSError, NotImplementedError):
            return False

    def test_links_and_preserves_existing_files(self, tmp_path):
        if not self._can_symlink(tmp_path):
            pytest.skip("symlinks not permitted on this platform")
        project = tmp_path / "project"
        drive = tmp_path / "drive"
        (project / "output").mkdir(parents=True)
        (project / "output" / "already_here.mp4").write_text("keep me")

        report = colab_env.link_persistence(project, drive)

        link = project / "output"
        assert link.is_symlink()
        assert link.resolve() == (drive / "output").resolve()
        # The pre-existing file was moved to Drive, never dropped.
        assert (drive / "output" / "already_here.mp4").read_text() == "keep me"
        assert len(report) == len(colab_env.PERSISTED_DIRS)

    def test_idempotent(self, tmp_path):
        if not self._can_symlink(tmp_path):
            pytest.skip("symlinks not permitted on this platform")
        project = tmp_path / "project"
        drive = tmp_path / "drive"
        colab_env.link_persistence(project, drive)
        report = colab_env.link_persistence(project, drive)
        assert all("already linked" in line for line in report)


class TestProbe:
    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
    def test_probe_reads_a_real_file(self, tmp_path):
        video = tmp_path / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video),
            ],
            check=True,
            timeout=120,
        )
        info = colab_env.probe(video)
        assert (info["width"], info["height"]) == (320, 240)
        assert info["has_audio"] is True
        assert 0.5 < info["duration"] < 2.0

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
    def test_verify_videos_flags_a_wrong_canvas(self, tmp_path):
        video = tmp_path / "wrong.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", str(video),
            ],
            check=True,
            timeout=120,
        )
        rows = colab_env.verify_videos(tmp_path, "reel")
        assert len(rows) == 1
        # 320x240 is not the reel canvas: the report must not claim success.
        assert (rows[0]["width"], rows[0]["height"]) != colab_env.CANVAS["reel"]

    def test_verify_videos_on_empty_dir(self, tmp_path, capsys):
        assert colab_env.verify_videos(tmp_path, "reel") == []
        assert "no .mp4 files" in capsys.readouterr().out


class TestManifest:
    def test_manifest_is_written_as_utf8_json(self, tmp_path):
        rows = [{"name": "a.mp4", "duration": 1.0}]
        path = colab_env.write_manifest(tmp_path, rows, extra={"run": "prueba_ñ"})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run"] == "prueba_ñ"
        assert payload["videos"] == rows
        assert "created_at" in payload
