"""Audio package for TTS generation."""
from .tts_engine import TTSEngine, TTSResult, get_tts_engine

__all__ = ["TTSEngine", "TTSResult", "get_tts_engine"]