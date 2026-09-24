"""Media manager for fetching, verifying, and generating visual assets."""

import hashlib
import random
import requests
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from PIL import Image, ImageDraw, ImageFont

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


class MediaManager:
    """Manages media assets for video generation, unifying image and placeholder generation."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.image_settings = self.settings.image
        
        paths = self.settings.paths
        if hasattr(paths, 'assets_dir'):
            assets_root = paths.assets_dir
        elif hasattr(paths, 'project_root'):
            assets_root = paths.project_root / "assets"
        else:
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

    def _verify_image(self, path: Path) -> bool:
        """Verify if the image file is valid and not empty."""
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception as e:
            print(f"   [WARN] Corrupt image detected and removed: {path.name} ({e})")
            return False

    def clean_legacy_caches(self):
        """Clean 0-byte or corrupted images in the cache directory."""
        if not self._assets_dir.exists():
            return
            
        cleaned = 0
        for img_path in self._assets_dir.glob("*.jpg"):
            if not self._verify_image(img_path):
                try:
                    img_path.unlink()
                    cleaned += 1
                except Exception:
                    pass
        if cleaned > 0:
            print(f"   [INFO] Cleaned {cleaned} corrupted images from cache.")

    def get_local_asset(self, prompt: str) -> Optional[Path]:
        """Try to find a matching local asset that is valid."""
        cache_key = self._get_cache_key(prompt)
        cached = self._get_cached_path(cache_key)
        
        if self._verify_image(cached):
            return cached
        elif cached.exists():
            cached.unlink(missing_ok=True)

        local_path = self.image_settings.local_assets_path
        if local_path.exists():
            images = list(local_path.glob("*.jpg")) + list(local_path.glob("*.png")) + list(local_path.glob("*.jpeg"))
            valid_images = [img for img in images if self._verify_image(img)]
            if valid_images:
                return self._rng.choice(valid_images)

        return None

    def _generate_pollinations(self, prompt: str, scene_id: int) -> Optional[ImageAsset]:
        """Generate a cinematic scene image via Pollinations/Flux API with retries."""
        style_enhancement = (
            "photorealistic 35mm film still, stoic aesthetic, "
            "dramatic chiaroscuro lighting, marble statue texture, deep shadows, "
            "Rembrandt lighting, moody color grading, highly detailed, 8k, masterpiece"
        )
        
        full_prompt = f"{prompt}, {style_enhancement}"
        encoded = urllib.parse.quote(full_prompt)
        width, height = self.image_settings.width, self.image_settings.height
        
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&private=true&enhance=true&seed={scene_id}&model=flux"
        
        cache_key = self._get_cache_key(prompt)
        output_path = self._get_cached_path(cache_key)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=45, stream=True)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    if self._verify_image(output_path):
                        return ImageAsset(
                            path=output_path,
                            prompt=prompt,
                            source="pollinations",
                            width=width,
                            height=height,
                        )
                    else:
                        output_path.unlink(missing_ok=True)
                        print(f"   [WARN] Pollinations returned corrupted image on attempt {attempt+1}")
                else:
                    print(f"   [WARN] Pollinations HTTP {response.status_code} on attempt {attempt+1}")
            except Exception as e:
                print(f"   [WARN] Pollinations error on attempt {attempt+1}: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2)

        return None

    def _create_placeholder(self, prompt: str, scene_id: int) -> ImageAsset:
        """Create a resilient fallback placeholder image using PIL."""
        cache_key = self._get_cache_key(prompt)
        output_path = self._get_cached_path(cache_key)

        try:
            img = Image.new('RGB', (self.image_settings.width, self.image_settings.height), color='#1a1a2e')
            draw = ImageDraw.Draw(img)

            # Dark gradient background
            for y in range(self.image_settings.height):
                r = int(26 + (22 - 26) * y / self.image_settings.height)
                g = int(26 + (33 - 26) * y / self.image_settings.height)
                b = int(46 + (62 - 46) * y / self.image_settings.height)
                draw.line([(0, y), (self.image_settings.width, y)], fill=(r, g, b))

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 72)
            except Exception:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/georgia.ttf", 72)
                except Exception:
                    font = ImageFont.load_default()

            text = f"ESCENA {scene_id} - STOIC"
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((self.image_settings.width - w) // 2, (self.image_settings.height - h) // 2), text, font=font, fill='#f5f0e1')
            except Exception:
                pass # If text fails, just save gradient

            img.save(output_path, "JPEG", quality=90)
        except Exception as e:
            print(f"   [ERROR] Failed to create even a PIL placeholder: {e}")
            raise RuntimeError("Cannot generate images or placeholders.")

        return ImageAsset(
            path=output_path,
            prompt=prompt,
            source="placeholder",
            width=self.image_settings.width,
            height=self.image_settings.height,
        )

    def get_image_for_prompt(
        self,
        prompt: str,
        scene_id: int = 1,
        provider: Optional[Literal["local", "pollinations", "dalle", "stable_diffusion"]] = None,
    ) -> ImageAsset:
        """Get an image for the given prompt, applying retries and verifications."""
        self.clean_legacy_caches()
        provider = provider or self.image_settings.provider

        # Try cache first (now validated)
        cache_key = self._get_cache_key(prompt)
        cached = self._get_cached_path(cache_key)
        if self._verify_image(cached):
            return ImageAsset(
                path=cached,
                prompt=prompt,
                source="cache",
                width=self.image_settings.width,
                height=self.image_settings.height,
            )

        # Try local assets if configured
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

        # Pollinations generation
        print(f"   [INFO] Generando imagen real (Pollinations AI) para escena {scene_id}...")
        pollinations_asset = self._generate_pollinations(prompt, scene_id)
        if pollinations_asset:
            return pollinations_asset

        # Final Fallback
        print(f"   [WARN] Generación de imagen falló. Usando placeholder PIL para escena {scene_id}.")
        return self._create_placeholder(prompt, scene_id)

    def fetch_images_for_story(self, story) -> list[ImageAsset]:
        """Fetch images for all scenes in a story."""
        assets = []
        for scene in story.scenes:
            asset = self.get_image_for_prompt(scene.image_prompt, scene_id=scene.scene_number)
            if asset:
                scene.image_path = asset.path
                assets.append(asset)
        return assets


# Backward compatibility bindings
ImageManager = MediaManager

# Global instance
_manager: Optional[MediaManager] = None

def get_image_manager(settings=None) -> MediaManager:
    """Get or create the global media manager instance."""
    global _manager
    if _manager is None or settings is not None:
        _manager = MediaManager(settings)
    return _manager