"""Hardware profiles — adapt the whole pipeline to the Colab runtime in one step.

The defaults in :mod:`config.settings` are tuned for the *smallest* runtime
(T4 16 GB) so the project stays runnable everywhere. A profile upgrades them
for a bigger GPU **without editing a single default**, which keeps the test
suite honest and the change reversible.

Profiles
--------
``none``   no-op: keep whatever the settings already say.
``auto``   detect from the live CUDA device (``a100`` / ``t4`` / ``cpu``).
``colab``  alias of ``auto``.
``a100``   SDXL full (base → refiner → hires), no CPU offload, all
           encoder threads, ``medium`` preset. A100 has **no NVENC**.
``t4``     only the canvas per format; SDXL stays on sdxl-turbo 2 steps.

Usage
-----
>>> from config import get_settings
>>> from config.profiles import apply_profile, describe_profile
>>> settings = apply_profile(get_settings(), "auto", fmt="reel")
>>> describe_profile(settings)["video"]
'1080x1920'
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

PROFILE_CHOICES = ("auto", "a100", "t4", "colab", "none")

# A100 40 GB is the Colab Pro+ card. Anything with >= 30 GB of VRAM and a
# compute-class name is treated as the "big" profile.
A100_MIN_VRAM_GB = 30.0

# ── Canvas per format ────────────────────────────────────────────────
# SDXL was trained at ~1M pixels. Generating well above that duplicates
# limbs and wrecks compositions, so the diffuser output stays at the native
# size for each orientation (832x1216 = 1.01 MP) and the final canvas is
# reached with the hires pass instead of a plain Lanczos stretch.
FORMAT_CANVAS: dict[str, dict[str, Any]] = {
    "reel": {
        "video": (1080, 1920),
        "sdxl": (832, 1216),
        "font_size": 54,
        "font_margin": 440,
        "font_chars": 26,
    },
    "landscape": {
        "video": (1920, 1080),
        "sdxl": (1216, 832),
        "font_size": 72,
        "font_margin": 240,
        "font_chars": 45,
    },
}

# ── A100 image stack ─────────────────────────────────────────────────
# Two-stage SDXL (base + refiner) followed by an img2img hires pass at the
# final canvas. Everything here is diffusers-only: no Real-ESRGAN/basicsr,
# which is the classic thing that breaks a fresh Colab runtime.
SDXL_FULL_A100: dict[str, Any] = {
    "sdxl_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "sdxl_steps": 30,
    "sdxl_refiner_steps": 8,
    "sdxl_refiner_split": 0.8,
    "sdxl_guidance": 6.5,
    "sdxl_hires": True,
    "sdxl_hires_strength": 0.32,
    "sdxl_hires_steps": 12,
    "sdxl_cpu_offload": False,
    "upscale_enabled": False,
}


def detect_profile() -> str:
    """Return ``"a100"``, ``"t4"`` or ``"cpu"`` from the live CUDA device.

    Never raises: a machine without torch (or without CUDA) is ``"cpu"``.
    """
    try:  # torch is a heavy optional dependency — import it only here.
        import torch
    except Exception:
        return "cpu"
    try:
        if not torch.cuda.is_available():
            return "cpu"
        name = str(torch.cuda.get_device_name(0)).upper()
        props = torch.cuda.get_device_properties(0)
        total_gb = float(props.total_memory) / (1024 ** 3)
    except Exception:
        return "cpu"
    if "A100" in name or "H100" in name or total_gb >= A100_MIN_VRAM_GB:
        return "a100"
    return "t4"


def apply_profile(settings, profile: str = "auto", fmt: str = "landscape"):
    """Return a new :class:`Settings` with the profile applied.

    Args:
        settings: the frozen ``Settings`` dataclass to adapt.
        profile: one of :data:`PROFILE_CHOICES`. ``auto``/``colab`` detect.
        fmt: ``"reel"`` (9:16) or ``"landscape"`` (16:9).
    """
    profile = (profile or "auto").lower()
    if profile == "off":
        profile = "none"
    if profile not in PROFILE_CHOICES:
        raise ValueError(
            f"Unknown profile {profile!r}; expected one of {PROFILE_CHOICES}"
        )
    if profile == "none":
        return settings
    if profile in ("auto", "colab"):
        profile = detect_profile()

    canvas = FORMAT_CANVAS.get(fmt) or FORMAT_CANVAS["landscape"]
    vw, vh = canvas["video"]
    sw, sh = canvas["sdxl"]

    settings = replace(
        settings,
        video=replace(settings.video, width=vw, height=vh),
        image=replace(
            settings.image,
            width=vw,
            height=vh,
            sdxl_width=sw,
            sdxl_height=sh,
        ),
        font=replace(
            settings.font,
            size=canvas["font_size"],
            margin_bottom=canvas["font_margin"],
            max_chars_per_line=canvas["font_chars"],
        ),
    )

    if profile == "a100":
        settings = replace(
            settings,
            # A100/H100 have no NVENC block: encoding is CPU (libx264).
            # `threads=0` lets libx264 use every vCPU; `medium` is the
            # quality/speed sweet spot on 12 vCPUs at 1080x1920.
            video=replace(
                settings.video, nvenc=False, preset="medium", threads=0
            ),
            image=replace(settings.image, **SDXL_FULL_A100),
        )
    return settings


def describe_profile(settings) -> dict:
    """Effective configuration, for the notebooks to print **and assert on**.

    This is the contract between the repo and the Colab notebooks: if a
    notebook cell prints this and the values are wrong, the run is wrong.
    """
    image = settings.image
    video = settings.video
    return {
        "video": f"{video.width}x{video.height}",
        "fps": video.fps,
        "encoder": "h264_nvenc" if video.nvenc else video.codec,
        "preset": video.preset,
        "crf": video.crf,
        "threads": video.threads,
        "engine": video.engine,
        "image_provider": image.provider,
        "sdxl_model": image.sdxl_model,
        "sdxl_refiner": image.sdxl_refiner or "(none)",
        "sdxl_native": f"{image.sdxl_width}x{image.sdxl_height}",
        "sdxl_steps": image.sdxl_steps,
        "sdxl_guidance": image.sdxl_guidance,
        "sdxl_hires": image.sdxl_hires,
        "sdxl_cpu_offload": image.sdxl_cpu_offload,
        "esrgan_upscale": image.upscale_enabled,
        "tts": f"{settings.tts.provider}:{settings.tts.voice}",
        "cine": settings.cine.enabled,
        "visual_engine": settings.visual_engine.enabled,
    }


def assert_a100_ready(settings) -> list[str]:
    """Return the list of problems found for an A100 run (empty = all good).

    Used by the notebook gate so a misconfigured run dies **before** it
    burns compute units, not ten minutes into a batch.
    """
    problems: list[str] = []
    image, video = settings.image, settings.video

    if video.threads != 0:
        problems.append(
            f"video.threads={video.threads} (expected 0 on A100 so libx264 uses every vCPU)"
        )
    if video.nvenc:
        problems.append("video.nvenc=True but A100 has no NVENC — encode would fail")
    if "turbo" in image.sdxl_model.lower():
        problems.append(
            f"sdxl_model={image.sdxl_model} is the fast/draft turbo model, not SDXL full"
        )
    if not image.sdxl_refiner:
        problems.append("sdxl_refiner is empty — two-stage refinement is off")
    if image.sdxl_steps < 20:
        problems.append(
            f"sdxl_steps={image.sdxl_steps} is too low for SDXL full (use >= 20)"
        )
    if not image.sdxl_hires:
        problems.append("sdxl_hires=False — output stays at Lanczos upscale quality")
    if image.sdxl_cpu_offload:
        problems.append(
            "sdxl_cpu_offload=True — on a 40 GB A100 this only makes it slower"
        )
    if image.upscale_enabled:
        problems.append(
            "upscale_enabled=True needs Real-ESRGAN/basicsr; the A100 profile uses "
            "the hires pass instead. Drop --upscale."
        )
    return problems
