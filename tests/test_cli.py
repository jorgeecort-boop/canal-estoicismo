"""Tests for the CLI surface: parser flags and the batch orchestration.

The batch regression test exists because `batch` used to be unusable: its
per-video `class Args` did `theme = theme`, and a class body cannot read an
enclosing function local of the same name, so every run died with
`NameError: name 'theme' is not defined` before generating anything.
"""

import argparse
from pathlib import Path

import pytest

import main as cli
from config import get_settings
from config.profiles import PROFILE_CHOICES


class TestParser:
    def test_every_command_accepts_profile(self):
        parser = cli.create_parser()
        for command in ("generate", "batch", "test", "doctor"):
            args = parser.parse_args([command, "--profile", "a100"])
            assert args.profile == "a100"

    def test_profile_defaults_to_auto(self):
        parser = cli.create_parser()
        assert parser.parse_args(["generate"]).profile == "auto"

    def test_profile_choices_match_the_module(self):
        parser = cli.create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "--profile", "bogus"])
        for choice in PROFILE_CHOICES:
            assert parser.parse_args(["generate", "--profile", choice]).profile == choice

    def test_batch_controls_duration_and_scenes(self):
        """Without these the batch cost per video was left to chance."""
        parser = cli.create_parser()
        args = parser.parse_args(["batch", "--count", "5", "--duration", "6.5", "--scenes", "12"])
        assert args.count == 5
        assert args.duration == 6.5
        assert args.scenes == 12

    def test_batch_defaults(self):
        args = cli.create_parser().parse_args(["batch"])
        assert args.duration == 6.0
        assert args.scenes == 10

    def test_story_json_still_available(self):
        args = cli.create_parser().parse_args(["generate", "--story-json", "s.json"])
        assert args.story_json == Path("s.json")

    def test_doctor_has_its_own_format_default(self):
        args = cli.create_parser().parse_args(["doctor"])
        assert args.format == "reel"


class TestLlmChain:
    def test_none_means_no_provider(self):
        assert cli._llm_chain(None) == []

    def test_auto_uses_the_declared_order(self):
        assert cli._llm_chain("auto") == list(cli.LLM_AUTO_ORDER)

    def test_explicit_provider_is_alone(self):
        assert cli._llm_chain("gemini") == ["gemini"]


class TestBatchGenerate:
    @staticmethod
    def _args(tmp_path, count=3, themes=None):
        return argparse.Namespace(
            count=count,
            output_dir=tmp_path,
            themes=themes,
            duration=6.0,
            scenes=10,
            tts=None,
            cinematic=False,
            visual_engine=False,
            image_provider="local_assets",
            upscale=False,
            transitions="crossfade",
            gemini=False,
            llm=None,
            story_json=None,
        )

    @pytest.mark.asyncio
    async def test_passes_theme_output_and_settings_per_video(self, monkeypatch, tmp_path):
        captured = []

        async def fake_generate_video(args, settings):
            captured.append(args)
            return Path(args.output)

        monkeypatch.setattr(cli, "generate_video", fake_generate_video)

        themes = ["amor_fati", "memento_mori", "virtue_ethics"]
        args = self._args(tmp_path, count=3, themes=themes)
        results = await cli.batch_generate(args, get_settings(tmp_path))

        # The NameError regression: without a working args object this list is empty.
        assert [item.theme for item in captured] == themes
        for item in captured:
            assert Path(item.output).parent == tmp_path
            assert item.theme in Path(item.output).name
            assert item.duration == 6.0
            assert item.scenes == 10
            assert item.image_provider == "local_assets"
            assert item.story_json is None
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_themes_cycle_when_there_are_fewer_than_videos(self, monkeypatch, tmp_path):
        captured = []

        async def fake_generate_video(args, settings):
            captured.append(args)
            return Path(args.output)

        monkeypatch.setattr(cli, "generate_video", fake_generate_video)
        args = self._args(tmp_path, count=4, themes=["amor_fati", "memento_mori"])
        await cli.batch_generate(args, get_settings(tmp_path))

        assert [item.theme for item in captured] == [
            "amor_fati", "memento_mori", "amor_fati", "memento_mori"
        ]

    @pytest.mark.asyncio
    async def test_one_failure_does_not_stop_the_batch(self, monkeypatch, tmp_path, capsys):
        completed = []

        async def flaky_generate_video(args, settings):
            if args.theme == "memento_mori":
                raise RuntimeError("boom")
            completed.append(args.theme)
            return Path(args.output)

        monkeypatch.setattr(cli, "generate_video", flaky_generate_video)
        args = self._args(tmp_path, count=3, themes=["amor_fati", "memento_mori", "virtue_ethics"])
        results = await cli.batch_generate(args, get_settings(tmp_path))

        assert completed == ["amor_fati", "virtue_ethics"]
        assert len(results) == 2
        assert "[FAIL]" in capsys.readouterr().out
