#!/usr/bin/env python3
"""Main entry point for Stoic Video Generator."""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from config import get_settings
from src.narrative import StoryGenerator
from src.audio import TTSEngine
from src.media import ImageManager
from src.video import VideoComposer


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stoic Video Generator - Automated YouTube video creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s generate --theme control_dichotomy --output video.mp4
  %(prog)s generate --theme impermanence --draft
  %(prog)s batch --count 5 --output-dir ./output
  %(prog)s test --theme memento_mori
        """,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--draft", action="store_true", help="Draft mode (2s per scene)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a single video")
    gen_parser.add_argument(
        "--theme",
        choices=[
            "control_dichotomy", "impermanence", "virtue_ethics", "amor_fati",
            "memento_mori", "inner_fortress", "adversity_growth", "present_moment",
            "ego_death", "cosmic_perspective",
        ],
        help="Stoic theme (random if not specified)",
    )
    gen_parser.add_argument("--output", "-o", type=Path, help="Output video path")
    gen_parser.add_argument("--duration", type=float, help="Target duration in minutes")
    gen_parser.add_argument("--scenes", type=int, help="Number of scenes")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Generate multiple videos")
    batch_parser.add_argument("--count", type=int, default=3, help="Number of videos")
    batch_parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory")
    batch_parser.add_argument("--themes", nargs="+", help="Specific themes to use")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test pipeline with a single scene")
    test_parser.add_argument("--theme", default="control_dichotomy", help="Theme to test")

    return parser


async def generate_video(args, settings) -> Path:
    """Generate a complete video."""
    print("[VIDEO] Generating stoic video...")
    if args.theme:
        print(f"   Theme: {args.theme}")
    if args.duration:
        print(f"   Target duration: {args.duration} min")
    if args.scenes:
        print(f"   Scenes: {args.scenes}")
    if settings.draft_mode:
        print("   [DRAFT] DRAFT MODE: 2 seconds per scene")

    # 1. Generate story
    print("\n[STORY] Generating narrative...")
    story_gen = StoryGenerator(settings)
    story = story_gen.generate(
        theme=args.theme,
        target_duration=args.duration,
        num_scenes=args.scenes,
    )
    print(f"   [OK] Story generated: {story.title} ({len(story.scenes)} scenes, ~{story.total_estimated_duration:.1f}s)")

    # 2. Generate audio
    print("\n[TTS] Generating voiceovers...")
    tts_engine = TTSEngine(settings)

    async def audio_progress(current, total, scene_num):
        if scene_num:
            print(f"   Scene {scene_num}/{total}...", end="\r")

    await tts_engine.generate_for_story(story, progress_callback=audio_progress)
    print(f"   [OK] Audio generated for {len(story.scenes)} scenes")

    # 3. Fetch images
    print("\n[IMG] Fetching images...")
    img_manager = ImageManager(settings)
    img_manager.fetch_images_for_story(story)
    print(f"   [OK] Images ready for {len(story.scenes)} scenes")

    # 4. Compose video
    print("\n[VID] Composing video...")
    composer = VideoComposer(settings)

    def video_progress(current, total, msg):
        print(f"   {msg} ({current}/{total})", end="\r")

    result = composer.compose(story, output_path=args.output, draft_mode=settings.draft_mode, progress_callback=video_progress)
    print(f"\n   [OK] Video composed: {result.output_path}")
    print(f"   Duration: {result.duration:.1f}s | Size: {result.file_size / 1e6:.1f}MB")

    return result.output_path


async def batch_generate(args, settings) -> list[Path]:
    """Generate multiple videos."""
    print(f"[BATCH] Batch generating {args.count} videos...")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    story_gen = StoryGenerator(settings)
    themes = args.themes or [
        "control_dichotomy", "impermanence", "virtue_ethics", "amor_fati",
        "memento_mori", "inner_fortress", "adversity_growth", "present_moment",
        "ego_death", "cosmic_perspective",
    ]

    results = []
    for i in range(args.count):
        theme = themes[i % len(themes)]
        print(f"\n--- Video {i+1}/{args.count}: {theme} ---")

        output_path = output_dir / f"stoic_{theme}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

        # Temporarily override output path
        class Args:
            theme = theme
            output = output_path
            duration = None
            scenes = None

        try:
            path = await generate_video(Args(), settings)
            results.append(path)
        except Exception as e:
            print(f"   [FAIL] Failed: {e}")

    print(f"\n[OK] Batch complete: {len(results)}/{args.count} videos generated")
    return results


async def test_pipeline(args, settings) -> None:
    """Test the full pipeline with a single scene."""
    print("[TEST] Testing pipeline...")
    # Create new settings with draft mode enabled
    from config import get_settings
    settings = get_settings(draft_mode=True)

    story_gen = StoryGenerator(settings)
    story = story_gen.generate(theme=args.theme, num_scenes=1)
    scene = story.scenes[0]

    print(f"   Scene: {scene.text_overlay[:50]}...")
    print(f"   Voiceover: {scene.voiceover_text[:80]}...")
    print(f"   Image prompt: {scene.image_prompt[:80]}...")

    # Test TTS
    print("\n[TTS] Testing TTS...")
    tts_engine = TTSEngine(settings)
    result = await tts_engine.generate(scene.voiceover_text)
    print(f"   [OK] Audio: {result.duration:.2f}s ({result.audio_path})")

    # Test image
    print("\n[IMG] Testing image fetch...")
    img_manager = ImageManager(settings)
    asset = img_manager.get_image_for_prompt(scene.image_prompt)
    print(f"   [OK] Image: {asset.path} (source: {asset.source})")

    # Test video composition
    print("\n[VID] Testing video composition (2s draft)...")
    composer = VideoComposer(settings)
    scene.estimated_duration = 2.0
    scene.audio_path = result.audio_path
    scene.image_path = asset.path

    test_output = settings.paths.temp_dir / "test_output.mp4"
    comp_result = composer.compose(story, output_path=test_output, draft_mode=True)
    print(f"   [OK] Video: {comp_result.duration:.2f}s ({comp_result.output_path})")

    print("\n[OK] Pipeline test passed!")


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load settings
    settings = get_settings(debug=args.debug, draft_mode=args.draft)

    # Run command
    try:
        if args.command == "generate":
            asyncio.run(generate_video(args, settings))
        elif args.command == "batch":
            asyncio.run(batch_generate(args, settings))
        elif args.command == "test":
            asyncio.run(test_pipeline(args, settings))
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[WARN] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()