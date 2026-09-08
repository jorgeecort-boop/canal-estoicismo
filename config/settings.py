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
    bitrate: str = "8000k"
    preset: str = "medium"
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
    margin_bottom: int = 120
    max_chars_per_line: int = 45
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
    zoom_factor: float = 1.15
    duration_factor: float = 1.0
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "ease_in_out"
    direction: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "random"] = "random"


@dataclass(frozen=True)
class TTSSettings:
    """Text-to-Speech configuration."""
    provider: Literal["gtts", "edge-tts"] = "gtts"
    voice: str = "es-ES-AlvaroNeural"
    rate: str = "-15%"
    volume: str = "+0%"
    pitch: str = "-10Hz"
    language: str = "es"

    # gTTS settings
    gtts_lang: str = "es"
    gtts_tld: str = "com"

    # edge-tts CLI format (without % for Windows compatibility)
    edge_rate: str = "-15"
    edge_volume: str = "+0"
    edge_pitch: str = "-10Hz"


@dataclass(frozen=True)
class ImageSettings:
    """Image generation/asset configuration."""
    provider: Literal["dalle", "stable_diffusion", "local_assets", "pexels", "unsplash"] = "local_assets"
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


@dataclass(frozen=True)
class NarrativeSettings:
    """Story generation configuration."""
    target_duration_min: float = 3.0
    target_duration_max: float = 5.0
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
class Settings:
    """Main application settings container."""
    video: VideoSettings
    font: FontSettings
    kenburns: KenBurnsSettings
    tts: TTSSettings
    image: ImageSettings
    narrative: NarrativeSettings
    paths: PathSettings
    debug: bool = False
    draft_mode: bool = False

    @classmethod
    def load(cls, project_root: Path | None = None, debug: bool = False, draft_mode: bool = False) -> "Settings":
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        paths = PathSettings.from_root(project_root)
        paths.ensure_dirs()

        return cls(
            video=VideoSettings(),
            font=FontSettings(),
            kenburns=KenBurnsSettings(),
            tts=TTSSettings(),
            image=ImageSettings(),
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