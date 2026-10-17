#!/usr/bin/env python3
"""setup_colab.py — one-shot bootstrap for canal-estoicismo on Colab.

The notebooks in ``notebooks/`` are the primary path: they add Google Drive
persistence, the test gate and the acceptance gate. This script stays as a
minimal bootstrap for a terminal or CI run that needs no Drive:

    !python setup_colab.py
    !python setup_colab.py --check-only
    !python setup_colab.py --generate --format reel --profile auto

It deliberately does NOT install torch: Colab ships a torch built against its
own CUDA runtime, and replacing it is the classic way to break
``torch.cuda.is_available()``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/jorgeecort-boop/canal-estoicismo.git"
PROJECT_DIR = Path("/content/canal-estoicismo")
NOTEBOOKS = (
    "notebooks/00_entorno_A100.ipynb",
    "notebooks/01_verificacion_A100.ipynb",
    "notebooks/02_lote_reels.ipynb",
    "notebooks/03_entrega.ipynb",
)


def run(cmd, cwd=None, check=True):
    """Run a command, streaming output live."""
    print(f"\n$ {' '.join(str(part) for part in cmd)}")
    return subprocess.run([str(part) for part in cmd], cwd=cwd, check=check)


def step(message):
    print(f"\n{'=' * 60}")
    print(f"  {message}")
    print("=" * 60)


def gpu_report() -> dict:
    """Detect the Colab GPU without importing torch (it may not exist yet)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda": True,
            "name": torch.cuda.get_device_name(0),
            "vram_gb": round(props.total_memory / 1024 ** 3, 1),
            "torch": torch.__version__,
        }
    except Exception:
        return {"cuda": False}


def setup(install: bool = True):
    # ── 1. System packages ─────────────────────────────────────────
    step("1/5 · System packages (ffmpeg + fonts + espeak-ng)")
    run(["apt-get", "-qq", "install", "-y",
         "ffmpeg", "libsndfile1", "espeak-ng", "fonts-dejavu-core", "fonts-dejavu-extra"],
        check=False)

    # ── 2. Repository ──────────────────────────────────────────────
    step("2/5 · Getting the repository")
    if (PROJECT_DIR / ".git").exists():
        run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"], check=False)
    else:
        run(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT_DIR)])

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    # One source of truth for the dependency list.
    from scripts import colab_env

    # ── 3. Python packages ─────────────────────────────────────────
    if install:
        step("3/5 · Python packages (torch untouched)")
        colab_env.pip_install(colab_env.PYTHON_DEPS)
    else:
        step("3/5 · Python packages — skipped")

    # ── 4. GPU + packages ──────────────────────────────────────────
    step("4/5 · GPU and dependencies")
    info = colab_env.check_torch()
    for key, value in info.items():
        print(f"  {key:<15} {value}")
    if not info.get("cuda_available"):
        print("\n  [WARN] No CUDA GPU visible.")
        print("         Entorno de ejecución > Cambiar tipo de entorno > A100 GPU.")

    # ── 5. Tests ───────────────────────────────────────────────────
    step("5/5 · Test suite")
    code = run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_DIR, check=False).returncode
    print(f"\n  pytest exit code: {code}")

    gpu = gpu_report()
    profile = "a100" if gpu.get("cuda") and gpu.get("vram_gb", 0) >= 30 else "auto"

    print("\n" + "=" * 60)
    print("  Setup finished")
    print("=" * 60)
    if gpu.get("cuda"):
        print(f"  GPU : {gpu.get('name')} ({gpu.get('vram_gb')} GB)  ->  --profile {profile}")
    else:
        print("  GPU : none detected  ->  --profile auto falls back to sdxl-turbo/CPU")
    print("\n  Recommended one-liner (reel, best quality on this GPU):")
    print(
        "  python main.py generate --theme amor_fati --scenes 10 --duration 6 "
        "--output ./reel.mp4 --tts edge-tts --format reel --profile auto "
        "--cinematic --visual-engine --image-provider sdxl_local --llm auto"
    )
    print("\n  Better: use the notebooks (Drive persistence + gates):")
    for notebook in NOTEBOOKS:
        print(f"    {notebook}")
    print("\n  Note: A100/H100 have no NVENC — encoding is libx264 on CPU.")
    print("=" * 60 + "\n")
    return code


def generate(theme="control_dichotomy", scenes=6, duration=5,
             output="./video_oficial_5min.mp4", format_type="landscape",
             profile="auto", image_provider="sdxl_local"):
    """Run one generation after setup."""
    step(f"Generating — theme={theme}, scenes={scenes}, duration={duration}min, format={format_type}")
    cmd = [
        sys.executable, "main.py", "generate",
        "--theme", theme,
        "--scenes", str(scenes),
        "--duration", str(duration),
        "--format", format_type,
        "--profile", profile,
        "--output", output,
    ]
    if image_provider:
        cmd += ["--image-provider", image_provider]
    run(cmd, cwd=PROJECT_DIR, check=False)
    print(f"\n  Video saved: {PROJECT_DIR / output}")
    try:
        from google.colab import files  # type: ignore

        files.download(str(PROJECT_DIR / output))
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap canal-estoicismo on Colab")
    parser.add_argument("--generate", action="store_true", help="Also run one generation")
    parser.add_argument("--check-only", action="store_true", help="Skip the pip install")
    parser.add_argument("--theme", default="control_dichotomy")
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--format", dest="format_type", choices=["landscape", "reel"], default="reel")
    parser.add_argument("--profile", default="auto", help="Hardware profile (auto/a100/t4/none)")
    parser.add_argument("--image-provider", default="sdxl_local",
                        choices=["local_assets", "sdxl_local", "pollinations", "none"])
    parser.add_argument("--output", default="./video_oficial_5min.mp4")
    parsed = parser.parse_args()

    exit_code = setup(install=not parsed.check_only)

    if parsed.generate:
        provider = None if parsed.image_provider == "none" else parsed.image_provider
        generate(parsed.theme, parsed.scenes, parsed.duration, parsed.output,
                 parsed.format_type, parsed.profile, provider)

    sys.exit(0 if exit_code == 0 else exit_code)
