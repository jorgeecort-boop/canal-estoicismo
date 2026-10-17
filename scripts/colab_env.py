"""Colab helpers shared by the notebooks.

The logic lives here — not scattered inside the .ipynb files — so a fix is a
`git pull` in Colab instead of hand-editing cells, and so the setup path is
versioned and reviewable. The notebooks only wire these calls together and
assert the results.

Nothing here installs or downloads on import: every action is an explicit call.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/jorgeecort-boop/canal-estoicismo.git"
PROJECT_DIR = Path("/content/canal-estoicismo")
DRIVE_ROOT = Path("/content/drive/MyDrive/canal-estoicismo")
CANVAS = {"reel": (1080, 1920), "landscape": (1920, 1080)}

# Directories that must outlive the Colab session. `temp` is deliberately not
# here: it is scratch space and would only slow the Drive filesystem down.
PERSISTED_DIRS = ("assets/images", "output", "logs")

# torch is NOT in this list on purpose: Colab ships a torch built against its
# own CUDA runtime, and reinstalling it from another index is the classic way
# to break `torch.cuda.is_available()`.
PYTHON_DEPS = (
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "diffusers>=0.30.0",
    "safetensors>=0.4.0",
    "opencv-python-headless>=4.8.0",
    "soundfile>=0.12.1",
    "edge-tts>=6.1.0",
    "gtts>=2.5.0",
    "moviepy>=2.0.0",
    "python-dotenv>=1.0.0",
    "google-genai>=1.0.0",
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
)

SYSTEM_DEPS = (
    "ffmpeg",
    "libsndfile1",
    "espeak-ng",
    "fonts-dejavu-core",
    "fonts-dejavu-extra",
)

SECRET_KEYS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "NVIDIA_API_KEY",
    "OPENROUTER_API_KEY",
    "HF_TOKEN",
)


# ── output helpers ───────────────────────────────────────────────────

def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def gate(name: str, ok: bool, detail: str = "") -> None:
    """Print a PASS/FAIL line and raise when it fails.

    Raising is intentional: a red cell in Colab is a stop sign, and every
    gate below this one costs units.
    """
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")
    if not ok:
        raise SystemExit(f"gate failed: {name}")


# ── command execution ────────────────────────────────────────────────

def run(
    cmd,
    cwd: Path | str | None = None,
    check: bool = False,
    env: dict | None = None,
) -> int:
    """Run a command streaming its output live; return the exit code.

    Streaming matters: a batch runs for an hour and the user needs to watch
    progress, not wait for a silent cell.
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    cmd = [str(part) for part in cmd]
    print("$ " + " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if check and process.returncode != 0:
        raise SystemExit(f"command failed with exit {process.returncode}: {' '.join(cmd)}")
    return process.returncode


def pip_install(packages, quiet: bool = True) -> int:
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd += [str(p) for p in packages]
    return run(cmd, check=True)


def apt_install(packages) -> int:
    return run(
        ["apt-get", "-qq", "install", "-y", *[str(p) for p in packages]],
        check=False,
    )


# ── repository + persistence ─────────────────────────────────────────

def ensure_repo(url: str = REPO_URL, dest: Path = PROJECT_DIR) -> str:
    """Clone the repo shallow, or fast-forward an existing clone."""
    if (dest / ".git").exists():
        code = run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
        if code != 0:
            print("[WARN] git pull failed (dirty clone?) — keeping the current checkout")
            return "stale"
        return "updated"
    run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return "cloned"


def link_persistence(project_dir: Path = PROJECT_DIR, drive_root: Path = DRIVE_ROOT) -> list[str]:
    """Symlink the repo's output/cache dirs to Drive, preserving anything local.

    This is what makes a run survivable: the generated images are the most
    expensive artefact in the pipeline, and the symlinked cache means a second
    run never pays for them again.
    """
    report: list[str] = []
    for relative in PERSISTED_DIRS:
        local = project_dir / relative
        target = drive_root / relative
        target.mkdir(parents=True, exist_ok=True)
        if local.is_symlink():
            if local.resolve() == target.resolve():
                report.append(f"{relative}: already linked")
                continue
            local.unlink()
        elif local.exists():
            # Never delete user data: move it to Drive first.
            for item in list(local.iterdir()):
                destination = target / item.name
                if not destination.exists():
                    shutil.move(str(item), str(destination))
            shutil.rmtree(local, ignore_errors=True)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.symlink_to(target, target_is_directory=True)
        report.append(f"{relative}: linked -> {target}")
    return report


def install_runtime(format_name: str = "reel") -> None:
    """Apt + pip for the A100 pipeline (torch left untouched)."""
    banner("Installing system packages")
    apt_install(SYSTEM_DEPS)
    banner("Installing Python packages (torch untouched)")
    pip_install(PYTHON_DEPS)


def check_torch() -> dict:
    """Report the preinstalled torch; install it only when it is missing."""
    try:
        import torch

        info = {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)
        return info
    except Exception:
        print("[INFO] torch not installed — installing a CUDA build")
        pip_install(["torch"], quiet=False)
        import torch  # noqa: PLC0415

        return {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}


def load_secrets(keys=SECRET_KEYS) -> dict:
    """Copy Colab secrets into os.environ. Missing keys are reported, not fatal."""
    found: dict[str, bool] = {}
    try:
        from google.colab import userdata  # noqa: PLC0415
    except Exception:
        userdata = None
    for key in keys:
        value = os.environ.get(key)
        if not value and userdata is not None:
            try:
                value = userdata.get(key)
            except Exception:
                value = None
        if value:
            os.environ[key] = value
        found[key] = bool(value)
    for key, present in found.items():
        print(f"  {'[set]  ' if present else '[unset]'} {key}")
    return found


# ── verification ─────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    """ffprobe one media file into the fields the reports care about."""
    path = Path(path)
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {result.stderr.strip()[:200]}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "name": path.name,
        "path": str(path),
        "width": int(video.get("width", 0)) if video else 0,
        "height": int(video.get("height", 0)) if video else 0,
        "duration": round(float(data.get("format", {}).get("duration", 0.0) or 0.0), 2),
        "has_audio": audio is not None,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "story_json": path.with_suffix(".story.json").exists(),
    }


def verify_videos(directory: Path, format_name: str = "reel") -> list[dict]:
    """Probe every mp4 in a directory and print a manifest table."""
    canvas = CANVAS[format_name]
    rows = []
    for video in sorted(Path(directory).glob("*.mp4")):
        try:
            rows.append(probe(video))
        except Exception as exc:
            rows.append(
                {
                    "name": video.name, "path": str(video), "width": 0, "height": 0,
                    "duration": 0.0, "has_audio": False,
                    "size_mb": round(video.stat().st_size / 1e6, 1), "story_json": False,
                    "error": str(exc),
                }
            )
    if not rows:
        print(f"  [WARN] no .mp4 files in {directory}")
        return rows

    header = f"{'video':<52} {'res':>11} {'dur':>8} {'MB':>7} {'audio':>6} {'story':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        bad = (row["width"], row["height"]) != canvas or not row["has_audio"] or row["duration"] <= 0
        flag = "  <-- REVIEW" if bad else ""
        print(
            f"{row['name'][:52]:<52} {row['width']}x{row['height']:>5} "
            f"{row['duration']:>7.2f} {row['size_mb']:>7.1f} "
            f"{'yes' if row['has_audio'] else 'NO':>6} "
            f"{'yes' if row['story_json'] else 'no':>6}{flag}"
        )
    print(f"\n  Expected canvas: {canvas[0]}x{canvas[1]}")
    good = sum(
        1 for r in rows
        if (r["width"], r["height"]) == canvas and r["has_audio"] and r["duration"] > 0
    )
    print(f"  {good}/{len(rows)} videos match the expected canvas with audio")
    return rows


def write_manifest(directory: Path, rows: list[dict], extra: dict | None = None) -> Path:
    """Persist the run's results next to the videos, on Drive."""
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "directory": str(directory),
        "videos": rows,
    }
    if extra:
        payload.update(extra)
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  manifest -> {path}")
    return path
