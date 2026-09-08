"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    from config import reset_settings
    from src.audio.tts_engine import _engine
    from src.media.image_manager import _manager
    from src.video.composer import _composer

    reset_settings()

    # Reset module-level singletons
    import src.audio.tts_engine as tts_module
    import src.media.image_manager as img_module
    import src.video.composer as vid_module

    tts_module._engine = None
    img_module._manager = None
    vid_module._composer = None

    yield

    # Cleanup after test
    reset_settings()
    tts_module._engine = None
    img_module._manager = None
    vid_module._composer = None