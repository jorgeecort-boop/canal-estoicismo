"""Application settings and configuration constants."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os


@dataclass(frozen=True)
class VideoSettings:
    """Video output configuration."""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec: str = "libx264"
    bitrate: str = "12000k"
    preset: str = "slow"
    crf: int = 18
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> str:
        return "16:9"


@dataclass(frozen=True)
class FontSettings:
    """Typography configuration for text overlays."""
    family: str = "Georgia"
    # Windows font paths (Georgia is standard on Windows)
    windows_font_path: str = "C:/Windows/Fonts/georgia.ttf"
    # Linux font paths
    linux_font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    size: int = 72
    color: str = "#F5F0E1"
    stroke_color: str = "#1A1A1A"
    stroke_width: int = 3
    position: Literal["bottom", "center", "top"] = "bottom"
    margin_bottom: int = 240
    max_chars_per_line: int = 45
    max_lines: int = 2
    line_spacing: float = 1.3

    @property
    def font_path(self) -> str:
        """Get platform-appropriate font path."""
        import sys
        if sys.platform == "win32":
            return self.windows_font_path
        return self.linux_font_path


@dataclass(frozen=True)
class KenBurnsSettings:
    """Ken Burns effect configuration."""
    enabled: bool = True
    zoom_factor: float = 1.24
    duration_factor: float = 1.0
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "ease_in_out"
    direction: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "random"] = "random"


@dataclass(frozen=True)
class TTSSettings:
    """Text-to-Speech configuration - Solo opciones GRATIS sin facturación."""
    provider: Literal["gtts", "edge-tts", "hf", "hf-gpu", "piper-cpu"] = "edge-tts"
    
    # Edge-TTS voices (gratis, sin API key, alta calidad)
    # Voces masculinas maduras recomendadas:
    # es-MX-JorgeNeural - Mexicano, profundo (RECOMENDADO)
    # es-ES-AlvaroNeural - Español, maduro
    # es-AR-TomasNeural - Argentino, maduro
    # en-US-GuyNeural - Inglés US, profundo
    # en-GB-RyanNeural - Inglés UK, maduro
    voice: str = "es-MX-JorgeNeural"  # Mejor voz masculina madura español
    
    # Hugging Face TTS models (gratis, local, GPU opcional)
    # Modelos recomendados para español:
    # facebook/mms-tts-spa - MMS-TTS Spanish (buena calidad, multilingüe)
    # facebook/mms-tts-eng - English
    # microsoft/speecht5_tts - SpeechT5 (requiere speaker embeddings)
    hf_model: str = "facebook/mms-tts-spa"  # Modelo HF para español
    # Optional GPU provider. It is deliberately opt-in because it adds large
    # model dependencies and requires a CUDA runtime.
    hf_gpu_model: str = "hexgrad/Kokoro-82M"
    hf_gpu_voice: str = "es_male"
    hf_gpu_lang: str = "es"
    piper_voice: str = "es_ES-sharvard-medium"
    # Piper alternatives (deeper): "es_MX-ald-medium" (grave mexicana).
    # >1.0 = slower, more solemn. 1.15 ≈ philosopher pace.
    piper_length_scale: float = 1.15
    
    # Voces alternativas para testing
    voice_alternatives: tuple = (
        "es-MX-JorgeNeural",    # Mexicano, profundo
        "es-ES-AlvaroNeural",   # Español, maduro
        "es-AR-TomasNeural",    # Argentino, maduro
        "en-US-GuyNeural",      # Inglés US, profundo
        "en-GB-RyanNeural",     # Inglés UK, maduro
    )
    
    rate: str = "-15%"
    volume: str = "+0%"
    pitch: str = "-10Hz"
    language: str = "es"

    # gTTS settings (fallback gratis)
    gtts_lang: str = "es"
    gtts_tld: str = "com.mx"  # Español mexicano

    # edge-tts CLI format (sin Hz para compatibilidad)
    edge_rate: str = "-15%"
    edge_volume: str = "+0%"
    edge_pitch: str = "-10Hz"


@dataclass(frozen=True)
class ImageSettings:
    """Image generation/asset configuration."""
    provider: Literal["dalle", "stable_diffusion", "sdxl_local", "local_assets", "pexels", "unsplash"] = "local_assets"
    style_prompt: str = (
        "cinematic stoic aesthetic, marble statue, classical art, "
        "dark moody lighting, volumetric fog, dramatic shadows, "
        "ancient roman greek philosophy, high detail, 8k, masterpiece"
    )
    negative_prompt: str = (
        "bright, colorful, cartoon, anime, modern, text, watermark, "
        "signature, blurry, low quality, distorted, ugly"
    )
    local_assets_path: Path = Path("assets/images")
    width: int = 1920
    height: int = 1080
    # SDXL-Turbo local (T4 opt-in). 16:9 native 1024x576, 2-4 steps.
    sdxl_model: str = "stabilityai/sdxl-turbo"
    sdxl_steps: int = 2
    sdxl_width: int = 1024
    sdxl_height: int = 576
    sdxl_guidance: float = 0.0
    # Real-ESRGAN upscale (T4 opt-in, only when image < 1080p)
    upscale_enabled: bool = False
    upscale_model: str = "RealESRGAN_x4plus"


@dataclass(frozen=True)
class CineSettings:
    """Optional cinematic post-processing. Disabled by default."""
    enabled: bool = False
    saturation: float = 1.1
    contrast: float = 1.05
    vignette: float = 0.25
    grain: int = 6


@dataclass(frozen=True)
class VisualEngineSettings:
    """Optional image-to-animation layer for the MoviePy composer."""
    enabled: bool = False
    particle_count: int = 120
    smoke_alpha: float = 0.075
    lamp_flicker: float = 0.05
    seed: int = 0


@dataclass(frozen=True)
class NarrativeSettings:
    """Story generation configuration."""
    # Long-form videos are intentionally constrained to 5-8 minutes.
    target_duration_min: float = 5.0
    target_duration_max: float = 8.0
    scenes_min: int = 5
    scenes_max: int = 8
    words_per_minute: int = 140
    text_overlay_max_chars: int = 80
    voiceover_min_words: int = 30
    voiceover_max_words: int = 120

    themes: tuple[str, ...] = (
        "control_dichotomy",
        "impermanence",
        "virtue_ethics",
        "amor_fati",
        "memento_mori",
        "inner_fortress",
        "adversity_growth",
        "present_moment",
        "ego_death",
        "cosmic_perspective",
    )

    philosophers: tuple[str, ...] = (
        "Marcus Aurelius",
        "Seneca",
        "Epictetus",
        "Zeno of Citium",
        "Chrysippus",
        "Cleanthes",
        "Musonius Rufus",
    )


@dataclass(frozen=True)
class PathSettings:
    """Filesystem paths."""
    project_root: Path
    output_dir: Path
    temp_dir: Path
    assets_dir: Path
    logs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "PathSettings":
        return cls(
            project_root=root,
            output_dir=root / "output",
            temp_dir=root / "temp",
            assets_dir=root / "assets",
            logs_dir=root / "logs",
        )

    def ensure_dirs(self) -> None:
        for path in [self.output_dir, self.temp_dir, self.assets_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TransitionSettings:
    """Varied xfade transitions (opt-in). Crossfade remains the fallback."""
    varied: bool = False
    duration: float = 0.5


@dataclass(frozen=True)
class Settings:
    """Main application settings container."""
    video: VideoSettings
    font: FontSettings
    kenburns: KenBurnsSettings
    tts: TTSSettings
    image: ImageSettings
    cine: CineSettings
    visual_engine: VisualEngineSettings
    transitions: TransitionSettings
    narrative: NarrativeSettings
    paths: PathSettings
    debug: bool = False
    draft_mode: bool = False

    @classmethod
    def load(cls, project_root: Path | None = None, debug: bool = False, draft_mode: bool = False) -> "Settings":
        if project_root is None:
            project_root = Path(__file__).parent.parent

        paths = PathSettings.from_root(project_root)
        paths.ensure_dirs()

        return cls(
            video=VideoSettings(),
            font=FontSettings(),
            kenburns=KenBurnsSettings(),
            tts=TTSSettings(),
            image=ImageSettings(),
            cine=CineSettings(),
            visual_engine=VisualEngineSettings(),
            transitions=TransitionSettings(),
            narrative=NarrativeSettings(),
            paths=paths,
            debug=debug,
            draft_mode=draft_mode,
        )


_settings: Settings | None = None


def get_settings(
    project_root: Path | None = None,
    debug: bool = False,
    draft_mode: bool = False,
    force_reload: bool = False,
) -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None or force_reload:
        _settings = Settings.load(project_root, debug, draft_mode)
    return _settings


def reset_settings() -> None:
    """Reset the global settings (mainly for testing)."""
    global _settings
    _settings = None
