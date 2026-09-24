#!/usr/bin/env python3
"""
setup_colab.py — One-click setup for canal-estoicismo on Google Colab.

Usage (run from any Colab cell):
    !python setup_colab.py
    !python setup_colab.py --generate   # also runs a test generation after setup
    !python setup_colab.py --theme amor_fati --scenes 6 --duration 5
"""

import os
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/jorgeecort-boop/canal-estoicismo.git"
PROJECT_DIR = Path("/content/canal-estoicismo")
REQUIRED_APT = ["ffmpeg", "fonts-dejavu-core", "fonts-dejavu-extra"]
REQUIRED_PIP = [
    "edge-tts>=6.1.0",
    "gtts>=2.5.0",
    "moviepy>=2.0.0",
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
]


def run(cmd: list[str], cwd=None, check=True) -> subprocess.CompletedProcess:
    """Run a shell command, streaming output live."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result


def step(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


def setup():
    # ── 1. Install system packages ─────────────────────────────────
    step("1/5 · Installing system packages (ffmpeg + fonts)")
    run(["apt-get", "install", "-y", "-q"] + REQUIRED_APT)

    # ── 2. Clone or update repo ────────────────────────────────────
    step("2/5 · Cloning / updating repository")
    if PROJECT_DIR.exists():
        print("  → Repo already cloned, pulling latest changes...")
        run(["git", "pull"], cwd=PROJECT_DIR)
    else:
        run(["git", "clone", REPO_URL, str(PROJECT_DIR)])

    # ── 3. Install Python dependencies ────────────────────────────
    step("3/5 · Installing Python dependencies")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + REQUIRED_PIP)

    # ── 4. Smoke test: import check ───────────────────────────────
    step("4/5 · Running import smoke test")
    smoke_cmd = [
        sys.executable, "-c",
        (
            "import sys; sys.path.insert(0, '.'); "
            "from config import get_settings; "
            "from src.narrative import StoryGenerator; "
            "from src.audio import TTSEngine; "
            "from src.media import ImageManager; "
            "from src.video import VideoComposer; "
            "print('  [OK] All imports successful')"
        ),
    ]
    run(smoke_cmd, cwd=PROJECT_DIR)

    # ── 5. Verify ffprobe / ffmpeg ────────────────────────────────
    step("5/5 · Verifying FFmpeg")
    run(["ffmpeg", "-version"])

    print("\n" + "="*60)
    print("  ✅  Setup complete! Run your video generation:")
    print(f"  cd {PROJECT_DIR}")
    print("  python main.py generate --theme control_dichotomy --scenes 6 --duration 5 --output ./video.mp4")
    print("="*60 + "\n")


def generate(theme="control_dichotomy", scenes=6, duration=5, output="./video_oficial_5min.mp4"):
    """Run video generation after setup."""
    step(f"Generating video — theme={theme}, scenes={scenes}, duration={duration}min")
    run(
        [
            sys.executable, "main.py", "generate",
            "--theme", theme,
            "--scenes", str(scenes),
            "--duration", str(duration),
            "--output", output,
        ],
        cwd=PROJECT_DIR,
    )
    print(f"\n  ✅  Video saved: {PROJECT_DIR / output}")
    # Auto-download in Colab
    try:
        from google.colab import files  # type: ignore
        files.download(str(PROJECT_DIR / output))
    except ImportError:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup canal-estoicismo on Colab")
    parser.add_argument("--generate", action="store_true", help="Run video generation after setup")
    parser.add_argument("--theme", default="control_dichotomy")
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", default="./video_oficial_5min.mp4")
    args = parser.parse_args()

    setup()

    if args.generate:
        generate(args.theme, args.scenes, args.duration, args.output)
