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
        self._sdxl_pipeline = None

    def _get_sdxl_pipeline(self):
        """Load SDXL-Turbo once on T4 (lazy singleton)."""
        if self._sdxl_pipeline is not None:
            return self._sdxl_pipeline
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise RuntimeError("SDXL requires diffusers+torch; install requirements-gpu.txt") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SDXL local requires CUDA GPU (T4)")
        cfg = self.image_settings
        pipe = AutoPipelineForText2Image.from_pretrained(
            cfg.sdxl_model, torch_dtype=torch.float16, variant="fp16",
        ).to("cuda")
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
        self._sdxl_pipeline = pipe
        print(f"   [INFO] SDXL loaded once: {cfg.sdxl_model}")
        return pipe

    def _generate_sdxl_local(self, prompt: str, scene_id: int) -> Optional[ImageAsset]:
        """Generate 16:9 image with SDXL-Turbo (2-4 steps, seed per scene)."""
        cfg = self.image_settings
        style_35mm = (
            "photorealistic 35mm film still, stoic aesthetic, dramatic chiaroscuro, "
            "marble texture, deep shadows, Rembrandt lighting, moody grade, highly detailed, 8k"
        )
        full_prompt = f"{prompt}, {style_35mm}"
        cache_key = self._get_cache_key(f"{prompt}|sdxl:{scene_id}")
        output_path = self._get_cached_path(cache_key)
        if self._verify_image(output_path):
            return ImageAsset(path=output_path, prompt=prompt, source="sdxl-cache",
                              width=cfg.width, height=cfg.height)
        try:
            import torch
            pipe = self._get_sdxl_pipeline()
            image = pipe(
                prompt=full_prompt,
                negative_prompt=cfg.style_prompt and self.image_settings.negative_prompt,
                width=cfg.sdxl_width, height=cfg.sdxl_height,
                num_inference_steps=max(1, min(4, cfg.sdxl_steps)),
                guidance_scale=cfg.sdxl_guidance,
                generator=torch.Generator(device="cuda").manual_seed(1000 + scene_id),
            ).images[0]
            image = image.resize((cfg.width, cfg.height), Image.LANCZOS)
            image.save(output_path, "JPEG", quality=92)
            if not self._verify_image(output_path):
                output_path.unlink(missing_ok=True)
                return None
            return ImageAsset(path=output_path, prompt=prompt, source="sdxl-local",
                              width=cfg.width, height=cfg.height)
        except Exception as exc:
            print(f"   [WARN] SDXL local failed scene {scene_id}: {exc}")
            return None

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
        
        cache_key = self._get_cache_key(f"{prompt}|scene:{scene_id}")
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
        cache_key = self._get_cache_key(f"{prompt}|scene:{scene_id}")
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

    def _maybe_upscale(self, path: Path) -> Path:
        """Upscale to 1920x1080 only if smaller; Real-ESRGAN on CUDA, else Lanczos."""
        if not self.image_settings.upscale_enabled:
            return path
        try:
            with Image.open(path) as img:
                w, h = img.size
            if w >= 1920 and h >= 1080:
                return path
            try:
                import torch
                if torch.cuda.is_available():
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                    num_block=23, num_grow_ch=32, scale=4)
                    upsampler = RealESRGANer(scale=4, model_path=None, model=model,
                                             tile=512, tile_pad=10, pre_pad=0,
                                             half=True, gpu_id=0)
                    import numpy as np
                    import cv2
                    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    out, _ = upsampler.enhance(raw, outscale=2)
                    cv2.imwrite(str(path), out)
            except Exception as exc:
                print(f"   [INFO] Real-ESRGAN skipped ({exc}); using Lanczos")
            with Image.open(path) as img:
                if img.size != (1920, 1080):
                    img.resize((1920, 1080), Image.LANCZOS).save(path, "JPEG", quality=92)
            # Subtle color after upscale
            with Image.open(path) as img:
                from PIL import ImageEnhance
                img = ImageEnhance.Color(img).enhance(1.05)
                img = ImageEnhance.Contrast(img).enhance(1.03)
                img.save(path, "JPEG", quality=92)
            return path
        except Exception as exc:
            print(f"   [WARN] Upscale failed: {exc}")
            return path

    def get_image_for_prompt(
        self,
        prompt: str,
        scene_id: int = 1,
        provider: Optional[Literal["local", "pollinations", "dalle", "stable_diffusion"]] = None,
    ) -> ImageAsset:
        """Get an image for the given prompt, applying retries and verifications."""
        self.clean_legacy_caches()
        provider = provider or self.image_settings.provider

        # Try cache first (now validated, scoped per scene to avoid collisions)
        cache_key = self._get_cache_key(f"{prompt}|scene:{scene_id}")
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

        # SDXL local primary on T4 (opt-in), Pollinations fallback
        if provider == "sdxl_local":
            sdxl_asset = self._generate_sdxl_local(prompt, scene_id)
            if sdxl_asset:
                sdxl_asset.path = self._maybe_upscale(sdxl_asset.path)
                return sdxl_asset
            print(f"   [INFO] SDXL failed, falling back to Pollinations for scene {scene_id}...")

        # Pollinations generation
        print(f"   [INFO] Generando imagen real (Pollinations AI) para escena {scene_id}...")
        pollinations_asset = self._generate_pollinations(prompt, scene_id)
        if pollinations_asset:
            pollinations_asset.path = self._maybe_upscale(pollinations_asset.path)
            return pollinations_asset

        # Final Fallback
        print(f"   [WARN] Generación de imagen falló. Usando placeholder PIL para escena {scene_id}.")
        return self._create_placeholder(prompt, scene_id)

    def fetch_images_for_story(self, story) -> list[ImageAsset]:
        """Fetch images for all scenes in a story (sequential)."""
        assets = []
        for scene in story.scenes:
            asset = self.get_image_for_prompt(scene.image_prompt, scene_id=scene.scene_number)
            if asset:
                scene.image_path = asset.path
                assets.append(asset)
        return assets

    async def fetch_images_for_story_async(self, story, max_concurrent: int = 3) -> list[ImageAsset]:
        """Fetch images for all scenes concurrently with Semaphore throttle."""
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)
        loop = asyncio.get_event_loop()

        # Pre-clean cache once before parallel run
        self.clean_legacy_caches()

        async def _fetch_one(scene) -> ImageAsset:
            async with semaphore:
                # Run blocking I/O in thread pool so it doesn't block the event loop
                asset = await loop.run_in_executor(
                    None,
                    lambda: self.get_image_for_prompt(
                        scene.image_prompt,
                        scene_id=scene.scene_number,
                    ),
                )
                scene.image_path = asset.path
                return asset

        tasks = [_fetch_one(scene) for scene in story.scenes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assets = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"   [ERROR] Image scene {i+1} failed: {result}")
            else:
                assets.append(result)

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