#!/usr/bin/env python3
"""colab_check.py — A100 acceptance gate. Run it BEFORE any batch.

Every gate is independent and reports PASS/FAIL, its wall time and the peak
VRAM it used, so a failure tells you exactly which stage broke and how much of
a run it would have wasted. Exit code 0 means "start the batch".

    !python scripts/colab_check.py                  # full check, reel format
    !python scripts/colab_check.py --format landscape
    !python scripts/colab_check.py --skip-video      # images + audio only
    !python scripts/colab_check.py --skip-sdxl       # compose/audio only

This is deliberately a script (not a pytest file): pytest must stay runnable on
a laptop with no GPU, and this one needs a real CUDA device.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings, reset_settings  # noqa: E402
from config.profiles import (  # noqa: E402
    apply_profile,
    assert_a100_ready,
    describe_profile,
    detect_profile,
)

BAR = "=" * 68
RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str = "", seconds: float | None = None) -> None:
    """Print and store one gate result."""
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail), "seconds": seconds})
    mark = "PASS" if ok else "FAIL"
    timing = f" ({seconds:.1f}s)" if seconds is not None else ""
    print(f"  [{mark}] {name}{timing}")
    for line in str(detail).splitlines():
        print(f"         {line}")


def ffprobe(path: Path) -> dict:
    """Return {width, height, duration, has_audio, size_mb} for a media file."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    return {
        "width": int(video.get("width", 0)) if video else 0,
        "height": int(video.get("height", 0)) if video else 0,
        "duration": float(data.get("format", {}).get("duration", 0.0) or 0.0),
        "has_audio": audio is not None,
        "size_mb": path.stat().st_size / 1e6 if path.exists() else 0.0,
    }


def integrated_lufs(path: Path) -> float | None:
    """Measure integrated loudness with ffmpeg's loudnorm pass."""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", "loudnorm=I=-14:print_format=json", "-f", "null", "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception:
        return None
    blob = out.stderr
    start = blob.rfind("{")
    end = blob.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return float(json.loads(blob[start:end + 1]).get("input_i"))
    except Exception:
        return None


def gate_environment() -> tuple[bool, float]:
    """Gate 0: the runtime really has a usable CUDA device."""
    try:
        import torch
    except Exception as exc:
        return False, 0.0
    if not torch.cuda.is_available():
        return False, 0.0
    props = torch.cuda.get_device_properties(0)
    return True, props.total_memory / 1024 ** 3


def main() -> int:
    parser = argparse.ArgumentParser(description="A100 acceptance gate")
    parser.add_argument("--format", choices=["reel", "landscape"], default="reel")
    parser.add_argument("--theme", default="control_dichotomy")
    parser.add_argument("--skip-sdxl", action="store_true", help="Skip the image gate")
    parser.add_argument("--skip-tts", action="store_true", help="Skip the voice gate")
    parser.add_argument("--skip-video", action="store_true", help="Skip the compose gate")
    args = parser.parse_args()

    started = time.perf_counter()
    print(BAR)
    print("  STOIC PIPELINE — A100 ACCEPTANCE GATE")
    print(BAR)

    # ── Gate 0: environment ──────────────────────────────────────────
    print("\n[G0] Environment")
    t0 = time.perf_counter()
    has_gpu, vram_gb = gate_environment()
    if not has_gpu:
        record(
            "CUDA device",
            False,
            "no CUDA GPU visible — in Colab use Entorno de ejecución > "
            "Cambiar tipo de entorno > A100 GPU",
            time.perf_counter() - t0,
        )
        return 1
    import torch  # noqa: PLC0415

    try:
        import diffusers

        diffusers_version = diffusers.__version__
    except Exception:
        diffusers_version = "MISSING"
    device_name = torch.cuda.get_device_name(0)
    record(
        "CUDA device",
        diffusers_version != "MISSING",
        f"{device_name} | {vram_gb:.1f} GB VRAM | torch {torch.__version__} | "
        f"diffusers {diffusers_version}",
        time.perf_counter() - t0,
    )
    if vram_gb < 30:
        print(
            f"         [WARN] {vram_gb:.0f} GB is not an A100 (40 GB). The SDXL-full\n"
            "         profile needs ~24 GB; use --profile t4 on smaller cards."
        )

    # ── Gate 1: profile ──────────────────────────────────────────────
    print("\n[G1] Profile")
    t0 = time.perf_counter()
    reset_settings()
    settings = apply_profile(get_settings(ROOT), "auto", fmt=args.format)
    problems = assert_a100_ready(settings) if detect_profile() == "a100" else []
    record(
        "Profile resolution",
        not problems,
        "\n".join(problems) if problems else f"detected profile: {detect_profile()}",
        time.perf_counter() - t0,
    )
    print("         effective configuration:")
    for key, value in describe_profile(settings).items():
        print(f"           {key:<16} {value}")

    canvas = (settings.image.width, settings.image.height)
    video_canvas = (settings.video.width, settings.video.height)

    # ── Gate 2: SDXL full image ──────────────────────────────────────
    image_path = None
    cold_seconds = None
    warm_seconds = None
    if args.skip_sdxl:
        print("\n[G2] SDXL image — SKIPPED")
    else:
        print("\n[G2] SDXL image (base -> refiner -> hires)")
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            import asyncio  # noqa: F401  (kept for symmetry with the other gates)
            from dataclasses import replace

            from PIL import Image

            from src.media import ImageManager

            settings = apply_profile(
                get_settings(ROOT, force_reload=True), "auto", fmt=args.format
            )
            settings = replace(
                settings, image=replace(settings.image, provider="sdxl_local")
            )
            manager = ImageManager(settings)

            # Cold image: pays the one-off model download + load.
            cold_started = time.perf_counter()
            asset = manager.get_image_for_prompt(
                "a weathered marble bust of a stoic philosopher, candlelight", scene_id=1
            )
            cold_seconds = time.perf_counter() - cold_started
            image_path = asset.path
            with Image.open(image_path) as img:
                produced = img.size
            load_peak = torch.cuda.max_memory_allocated() / 1e9
            stack = manager._sdxl_pipeline
            hires_loaded = bool(stack is not None and stack.img2img not in (None, False))

            # Warm image: the steady-state cost a batch actually pays per scene.
            warm_started = time.perf_counter()
            manager.get_image_for_prompt(
                "an ancient roman column at dawn, volumetric fog", scene_id=2
            )
            warm_seconds = time.perf_counter() - warm_started
            peak = torch.cuda.max_memory_allocated() / 1e9

            ok = (
                produced == canvas
                and asset.source.startswith("sdxl")
                and warm_seconds < cold_seconds
            )
            record(
                "SDXL full image",
                ok,
                f"source={asset.source} size={produced[0]}x{produced[1]} "
                f"(canvas {canvas[0]}x{canvas[1]}) hires={'yes' if hires_loaded else 'NO'}\n"
                f"cold (with model load) {cold_seconds:.1f}s | warm {warm_seconds:.1f}s per image\n"
                f"peak VRAM {peak:.1f} GB (load {load_peak:.1f} GB)",
                time.perf_counter() - t0,
            )
            manager.unload_models()
        except Exception as exc:
            record("SDXL full image", False, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)

    # ── Gate 3: voice ────────────────────────────────────────────────
    audio_path = None
    audio_duration = 0.0
    if args.skip_tts:
        print("\n[G3] Voice — SKIPPED")
    else:
        print("\n[G3] Voice (edge-tts + -14 LUFS master)")
        t0 = time.perf_counter()
        try:
            import asyncio

            from src.audio import TTSEngine

            engine = TTSEngine(settings)
            line = (
                "Acepta lo que no depende de ti y elige con serenidad tu respuesta."
            )
            result = asyncio.run(engine.generate(line))
            audio_path = Path(result.audio_path)
            audio_duration = float(result.duration)
            lufs = integrated_lufs(audio_path) if audio_path.exists() else None
            ok = audio_path.exists() and audio_duration > 0.5
            detail = f"{audio_duration:.2f}s at {audio_path.name} ({audio_path.stat().st_size / 1024:.0f} KB)"
            if lufs is not None:
                detail += f" | integrated loudness {lufs:.1f} LUFS (target -14)"
                ok = ok and -22.0 <= lufs <= -8.0
            record("Voice", ok, detail, time.perf_counter() - t0)
        except Exception as exc:
            record("Voice", False, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)

    # ── Gate 4: compose ──────────────────────────────────────────────
    if args.skip_video:
        print("\n[G4] Compose — SKIPPED")
    else:
        print("\n[G4] Compose (Ken Burns + subtitles + libx264)")
        t0 = time.perf_counter()
        try:
            import asyncio

            from src.audio import TTSEngine
            from src.media import ImageManager
            from src.narrative import Scene, Story
            from src.video import VideoComposer

            out_dir = settings.paths.output_dir / "_checks"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"smoke_{args.format}.mp4"

            if image_path is None:
                from dataclasses import replace as _replace

                m = ImageManager(_replace(settings, image=_replace(settings.image, provider="sdxl_local")))
                asset = m.get_image_for_prompt("marble stoic bust, candlelight", scene_id=1)
                image_path = asset.path
                m.unload_models()
            if audio_path is None:
                line = "Acepta lo que no depende de ti y elige con serenidad tu respuesta."
                result = asyncio.run(TTSEngine(settings).generate(line))
                audio_path = Path(result.audio_path)
                audio_duration = float(result.duration)

            scene = Scene(
                1,
                "Acepta lo que no depende de ti",
                "Acepta lo que no depende de ti y elige con serenidad tu respuesta.",
                "marble stoic bust",
                max(2.0, audio_duration),
            )
            scene.audio_path = str(audio_path)
            scene.image_path = str(image_path)
            story = Story(args.theme, "Smoke test", "Marcus Aurelius", [scene])
            story.total_estimated_duration = scene.estimated_duration

            result = VideoComposer(settings).compose(
                story, output_path=out_path, draft_mode=True
            )
            probe = ffprobe(result.output_path)
            ok = (
                probe["width"] == video_canvas[0]
                and probe["height"] == video_canvas[1]
                and probe["duration"] > 0
                and probe["has_audio"]
            )
            record(
                "Compose",
                ok,
                f"{probe['width']}x{probe['height']} | {probe['duration']:.2f}s | "
                f"audio={'yes' if probe['has_audio'] else 'NO'} | {probe['size_mb']:.2f} MB | "
                f"{result.output_path}",
                time.perf_counter() - t0,
            )
        except Exception as exc:
            record("Compose", False, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0)

    # ── Verdict ──────────────────────────────────────────────────────
    failed = [r for r in RESULTS if not r["ok"]]
    total = time.perf_counter() - started

    # Machine-readable report: notebook 01 turns this into a cost projection,
    # so the batch is budgeted from measured numbers instead of guesswork.
    try:
        report = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "format": args.format,
            "device": device_name,
            "vram_gb": round(vram_gb, 1),
            "canvas": list(canvas),
            "cold_image_seconds": round(cold_seconds, 1) if cold_seconds else None,
            "warm_image_seconds": round(warm_seconds, 1) if warm_seconds else None,
            "gates": RESULTS,
            "total_seconds": round(total, 1),
            "passed": not failed,
        }
        report_dir = get_settings(ROOT, force_reload=True).paths.output_dir / "_checks"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "acceptance.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n  report -> {report_dir / 'acceptance.json'}")
    except Exception as exc:  # never fail the gate over a report write
        print(f"  [WARN] could not write the report: {exc}")

    print("\n" + BAR)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} gates passed in {total:.1f}s")
    if failed:
        print("  DO NOT start the batch. Failing gates:")
        for item in failed:
            print(f"    - {item['name']}: {item['detail'] or 'see log above'}")
        print(BAR)
        return 1
    print("  READY — the pipeline is verified end-to-end on this runtime.")
    print(BAR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
