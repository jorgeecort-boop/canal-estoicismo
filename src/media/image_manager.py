"""Image manager for fetching and generating visual assets."""

import hashlib
import random
import requests
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

from config import get_settings


@dataclass
class ImageAsset:
    """Represents an image asset."""
    path: Path
    prompt: str
    source: str
    width: int
    height: int
    downloaded_at: float = time.time()

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "prompt": self.prompt,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "downloaded_at": self.downloaded_at,
        }


class ImageManager:
    """Manages image assets for video generation."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.image_settings = self.settings.image
        # Handle both PathSettings object and SimpleNamespace
        paths = self.settings.paths
        if hasattr(paths, 'assets_dir'):
            assets_root = paths.assets_dir
        elif hasattr(paths, 'project_root'):
            assets_root = paths.project_root / "assets"
        else:
            # Fallback
            assets_root = Path("assets")
        self._assets_dir = assets_root / "images"
        self._assets_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random()

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _get_cached_path(self, cache_key: str) -> Path:
        """Get cached image path."""
        return self._assets_dir / f"{cache_key}.jpg"

    def get_local_asset(self, prompt: str) -> Optional[Path]:
        """Try to find a matching local asset."""
        cache_key = self._get_cache_key(prompt)
        cached = self._get_cached_path(cache_key)
        if cached.exists():
            return cached

        # Search for any local assets
        local_path = self.image_settings.local_assets_path
        if local_path.exists():
            images = list(local_path.glob("*.jpg")) + list(local_path.glob("*.png")) + list(local_path.glob("*.jpeg"))
            if images:
                return self._rng.choice(images)

        return None

    def download_image(self, url: str, prompt: str) -> Optional[Path]:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            cache_key = self._get_cache_key(prompt)
            output_path = self._get_cached_path(cache_key)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path
        except Exception as e:
            print(f"Failed to download image: {e}")
            return None

    def get_image_for_prompt(
        self,
        prompt: str,
        provider: Optional[Literal["local", "pexels", "unsplash", "dalle", "stable_diffusion"]] = None,
    ) -> Optional[ImageAsset]:
        """Get an image for the given prompt."""
        provider = provider or self.image_settings.provider

        # Try cache first
        cache_key = self._get_cache_key(prompt)
        cached = self._get_cached_path(cache_key)
        if cached.exists():
            return ImageAsset(
                path=cached,
                prompt=prompt,
                source="cache",
                width=self.image_settings.width,
                height=self.image_settings.height,
            )

        # Try local assets
        if provider == "local" or provider == "local_assets":
            local = self.get_local_asset(prompt)
            if local:
                return ImageAsset(
                    path=local,
                    prompt=prompt,
                    source="local",
                    width=self.image_settings.width,
                    height=self.image_settings.height,
                )

        # For other providers, return None (would need API keys)
        # In production, implement actual API calls here
        print(f"Image provider '{provider}' not fully implemented. Using placeholder.")
        return self._create_placeholder(prompt)

    def _create_placeholder(self, prompt: str) -> ImageAsset:
        """Create a placeholder image using ImageMagick or similar."""
        cache_key = self._get_cache_key(prompt)
        output_path = self._get_cached_path(cache_key)

        # Create a simple gradient placeholder using ImageMagick
        try:
            import subprocess
            cmd = [
                "magick",
                "-size", f"{self.image_settings.width}x{self.image_settings.height}",
                "gradient:#1a1a2e-#16213e",
                "-gravity", "center",
                "-pointsize", "48",
                "-fill", "#f5f0e1",
                "-annotate", "+0+0", "STOIC VISUAL",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception:
            # Fallback: create minimal file
            output_path.write_bytes(b"")

        return ImageAsset(
            path=output_path,
            prompt=prompt,
            source="placeholder",
            width=self.image_settings.width,
            height=self.image_settings.height,
        )

    def fetch_images_for_story(self, story) -> list[ImageAsset]:
        """Fetch images for all scenes in a story."""
        assets = []
        for scene in story.scenes:
            asset = self.get_image_for_prompt(scene.image_prompt)
            if asset:
                scene.image_path = asset.path
                assets.append(asset)
        return assets


# Global instance
_manager: Optional[ImageManager] = None


def get_image_manager(settings=None) -> ImageManager:
    """Get or create the global image manager instance."""
    global _manager
    if _manager is None or settings is not None:
        _manager = ImageManager(settings)
    return _manager