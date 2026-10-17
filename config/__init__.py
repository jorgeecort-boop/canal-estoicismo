"""Configuration package for Stoicism Video Generator."""
from .settings import CineSettings, Settings, TransitionSettings, VisualEngineSettings, get_settings, reset_settings
from .profiles import (
    PROFILE_CHOICES,
    apply_profile,
    assert_a100_ready,
    describe_profile,
    detect_profile,
)

__all__ = [
    "CineSettings",
    "Settings",
    "TransitionSettings",
    "VisualEngineSettings",
    "get_settings",
    "reset_settings",
    "PROFILE_CHOICES",
    "apply_profile",
    "assert_a100_ready",
    "describe_profile",
    "detect_profile",
]
