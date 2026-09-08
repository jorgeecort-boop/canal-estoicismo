"""Stoic Video Generator - Main package."""

from .narrative import StoryGenerator, Scene, Story
from .audio import TTSEngine, TTSResult
from .media import ImageManager, ImageAsset
from .video import VideoComposer, CompositionResult

__version__ = "1.0.0"

__all__ = [
    "StoryGenerator",
    "Scene",
    "Story",
    "TTSEngine",
    "TTSResult",
    "ImageManager",
    "ImageAsset",
    "VideoComposer",
    "CompositionResult",
]