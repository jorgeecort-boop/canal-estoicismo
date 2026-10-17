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


class _SDXLStack:
    """The SDXL pipelines that stay resident for the whole run.

    One container instead of three attributes so ``unload_models`` and the
    smoke tests have a single thing to check. On sdxl-turbo only ``base`` is
    ever populated; ``refiner``/``img2img`` are the SDXL-full (A100) path.
    """

    __slots__ = ("torch", "base", "refiner", "img2img", "has_refiner")

    def __init__(self):
        self.torch = None
        self.base = None
        self.refiner = None
        self.img2img = None
        self.has_refiner = False


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
        """Load the SDXL stack once (lazy singleton).

        Small GPU (T4, 16 GB): single-stage sdxl-turbo with CPU offload —
        unchanged. Big GPU (A100, 40 GB): SDXL base + refiner kept resident,
        because the offload round-trip is pure overhead when it all fits.
        """
        if self._sdxl_pipeline is not None:
            return self._sdxl_pipeline
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "SDXL requires diffusers+torch; install requirements-gpu.txt"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SDXL local requires a CUDA GPU")

        cfg = self.image_settings
        stack = _SDXLStack()
        stack.torch = torch
        stack.base = self._load_sdxl_pipe(cfg.sdxl_model, "text2img", torch)

        # sdxl-turbo is a distilled single-stage model: there is nothing for a
        # refiner to add, so never load one for it.
        stack.has_refiner = bool(cfg.sdxl_refiner) and "turbo" not in cfg.sdxl_model.lower()
        if stack.has_refiner:
            try:
                # Official two-stage pattern: the refiner repo declares
                # `text_encoder`/`tokenizer` as null, so sharing the base's
                # text_encoder gives encode_prompt 1 tokenizer vs 2 encoders;
                # its zip then drops the pooled-negative embedding and the
                # refiner dies at `.repeat()` on None. Share te2 + vae only.
                stack.refiner = self._load_sdxl_pipe(
                    cfg.sdxl_refiner,
                    "img2img",
                    torch,
                    shared=self._sdxl_shared(stack, "refiner"),
                )
            except Exception as exc:
                print(f"   [WARN] SDXL refiner unavailable, base-only: {exc}")
                stack.has_refiner = False

        for pipe in (stack.base, stack.refiner):
            if pipe is None:
                continue
            if cfg.sdxl_cpu_offload:
                try:
                    pipe.enable_model_cpu_offload()
                    continue
                except Exception:
                    pass
            pipe.to("cuda")

        self._sdxl_pipeline = stack
        print(
            f"   [INFO] SDXL ready: {cfg.sdxl_model}"
            f"{' + refiner' if stack.has_refiner else ''}"
            f" | {cfg.sdxl_steps} steps, guidance {cfg.sdxl_guidance},"
            f" hires {'on' if cfg.sdxl_hires else 'off'}"
        )
        return stack

    @staticmethod
    def _sdxl_shared(stack: "_SDXLStack", kind: str) -> dict:
        """Components shared between the base pipeline and a secondary one.

        diffusers' encode_prompt zips tokenizers and text encoders, so the
        shared set must stay *aligned* with what the target repo declares:

        - refiner repo: only ``text_encoder_2``/``tokenizer_2`` exist -> share
          ``text_encoder_2`` + ``vae`` (the official two-stage pattern).
        - img2img built from the base repo: all four exist -> share the full
          aligned set so nothing gets loaded twice.
        """
        base = stack.base
        if kind == "refiner":
            return {"text_encoder_2": base.text_encoder_2, "vae": base.vae}
        return {
            "text_encoder": base.text_encoder,
            "text_encoder_2": base.text_encoder_2,
            "tokenizer": base.tokenizer,
            "tokenizer_2": base.tokenizer_2,
            "vae": base.vae,
        }

    @staticmethod
    def _load_sdxl_pipe(repo: str, kind: str, torch, shared: Optional[dict] = None):
        """Load a diffusers pipeline, tolerating repos without an fp16 variant."""
        from diffusers import (
            AutoPipelineForImage2Image,
            AutoPipelineForText2Image,
        )

        cls = AutoPipelineForText2Image if kind == "text2img" else AutoPipelineForImage2Image
        kwargs = dict(torch_dtype=torch.float16, **(shared or {}))
        last_exc: Exception | None = None
        for variant in ("fp16", None):
            try:
                return cls.from_pretrained(repo, variant=variant, **kwargs)
            except Exception as exc:
                last_exc = exc
        raise last_exc if last_exc else RuntimeError(f"cannot load {repo}")

    def _ensure_sdxl_img2img(self, stack: "_SDXLStack"):
        """Lazily build the hires-fix pipeline, sharing the base components.

        Returns the pipeline, or ``False`` (a sentinel) once loading has
        failed so we do not retry the download for every scene.
        """
        if stack.img2img is not None:
            return stack.img2img
        cfg = self.image_settings
        try:
            pipe = self._load_sdxl_pipe(
                cfg.sdxl_model,
                "img2img",
                stack.torch,
                shared=self._sdxl_shared(stack, "img2img"),
            )
            if cfg.sdxl_cpu_offload:
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe.to("cuda")
            else:
                pipe.to("cuda")
            stack.img2img = pipe
        except Exception as exc:
            print(f"   [WARN] SDXL hires unavailable, Lanczos only: {exc}")
            stack.img2img = False
        return stack.img2img

    def _sdxl_hires_fix(self, stack: "_SDXLStack", image, prompt: str, generator):
        """Refine a Lanczos-upscaled image at the final canvas size.

        SDXL was trained at ~1 MP, so asking for 1080x1920 in one go produces
        duplicated limbs. Upscaling and then re-diffusing at low strength
        ("hires fix") is the standard way to reach the canvas with real
        detail, and it needs nothing beyond diffusers. Any failure returns
        the plain Lanczos result rather than killing the scene.
        """
        cfg = self.image_settings
        target = (cfg.width, cfg.height)
        if not cfg.sdxl_hires or tuple(image.size) == target:
            return image
        upscaled = image.resize(target, Image.LANCZOS)
        pipe = self._ensure_sdxl_img2img(stack)
        if not pipe:
            return upscaled
        try:
            strength = min(0.9, max(0.05, float(cfg.sdxl_hires_strength)))
            # diffusers counts the *whole* schedule: only `strength` of it is
            # actually denoised, so scale up to land on the intended steps.
            num_steps = max(1, int(round(max(1, cfg.sdxl_hires_steps) / strength)))
            return pipe(
                prompt=prompt,
                negative_prompt=cfg.negative_prompt,
                image=upscaled,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=cfg.sdxl_guidance,
                generator=generator,
            ).images[0]
        except Exception as exc:
            print(f"   [WARN] SDXL hires pass failed, Lanczos only: {exc}")
            return upscaled

    def unload_models(self) -> None:
        """Release heavy generative models (SDXL ~7GB) before video compose.

        Prevents the Linux OOM killer on 12GB Colab runtimes: images are
        already on disk, so the pipeline is dead weight during MoviePy/FFmpeg.
        """
        if self._sdxl_pipeline is not None:
            del self._sdxl_pipeline
            self._sdxl_pipeline = None
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            print("   [INFO] SDXL unloaded, memory freed for compose")

    def _generate_sdxl_local(self, prompt: str, scene_id: int) -> Optional[ImageAsset]:
        """Generate one scene image with the local SDXL stack.

        Turbo (T4): a single 2-4 step pass. Full (A100): base → refiner →
        hires fix at the final canvas. Seeded per scene, so re-running the
        same story reproduces the same images.
        """
        cfg = self.image_settings
        style_35mm = (
            "photorealistic 35mm film still, stoic aesthetic, dramatic chiaroscuro, "
            "marble texture, deep shadows, Rembrandt lighting, moody grade, highly detailed, 8k"
        )
        full_prompt = f"{prompt[:200]}, {style_35mm}"[:300]
        # The generation profile is part of the cache key: switching from
        # turbo to SDXL full must never serve an image rendered by the old
        # settings, which is exactly what a prompt-only key would do.
        signature = (
            f"{cfg.sdxl_model}|{cfg.sdxl_refiner}|{cfg.sdxl_steps}"
            f"|{cfg.sdxl_guidance}|{cfg.sdxl_width}x{cfg.sdxl_height}"
            f"|h{int(cfg.sdxl_hires)}{cfg.sdxl_hires_strength}"
        )
        cache_key = self._get_cache_key(f"{prompt}|sdxl:{scene_id}|{signature}")
        output_path = self._get_cached_path(cache_key)
        if self._verify_image(output_path):
            w, h = self._image_size(output_path)
            return ImageAsset(path=output_path, prompt=prompt, source="sdxl-cache",
                              width=w, height=h)
        try:
            stack = self._get_sdxl_pipeline()
            is_turbo = "turbo" in cfg.sdxl_model.lower()
            steps = max(1, min(4, cfg.sdxl_steps)) if is_turbo else max(1, cfg.sdxl_steps)
            generator = stack.torch.Generator(device="cuda").manual_seed(1000 + scene_id)
            base_kwargs = dict(
                prompt=full_prompt,
                negative_prompt=cfg.negative_prompt,
                width=cfg.sdxl_width,
                height=cfg.sdxl_height,
                guidance_scale=cfg.sdxl_guidance,
                generator=generator,
            )

            if stack.has_refiner:
                split = min(0.95, max(0.05, float(cfg.sdxl_refiner_split)))
                # Same schedule-length trick as the hires pass: the refiner
                # only denoises the tail after `split`, so its step count is
                # scaled up to land on the intended number of real steps.
                refiner_total = max(
                    1, int(round(max(1, cfg.sdxl_refiner_steps) / (1.0 - split)))
                )
                latents = stack.base(
                    num_inference_steps=steps,
                    denoising_end=split,
                    output_type="latent",
                    **base_kwargs,
                ).images
                image = stack.refiner(
                    prompt=full_prompt,
                    num_inference_steps=refiner_total,
                    denoising_start=split,
                    image=latents,
                    generator=generator,
                ).images[0]
            else:
                image = stack.base(num_inference_steps=steps, **base_kwargs).images[0]

            image = self._sdxl_hires_fix(stack, image, full_prompt, generator)
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

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        """Real pixel size of a cached file.

        Cache hits must report what is actually on disk, not the configured
        canvas: a 576x1024 Pollinations fallback served through the generic
        cache once claimed to be 1080x1920 and the acceptance gate caught
        the lie (and it is also what the run manifest should state).
        """
        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return (0, 0)

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
        """Upscale to the *configured canvas* when the image is smaller.

        Real-ESRGAN on CUDA if it is importable, plain Lanczos otherwise.
        The target is ``image.width x image.height`` — NOT a hardcoded
        1920x1080, which silently turned every 9:16 reel into a squashed
        landscape frame.
        """
        if not self.image_settings.upscale_enabled:
            return path
        target = (self.image_settings.width, self.image_settings.height)
        try:
            with Image.open(path) as img:
                w, h = img.size
            if w >= target[0] and h >= target[1]:
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
                if img.size != target:
                    img.resize(target, Image.LANCZOS).save(path, "JPEG", quality=92)
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

        # Try cache first (validated, scoped per scene to avoid collisions).
        # Provider-aware: an sdxl_local run must never be served a cached
        # Pollinations/placeholder image that merely shares the generic
        # scene key — a failed SDXL fallback run poisons that key, and the
        # next sdxl_local run then renders nothing at all. The SDXL stack
        # keeps its own scope key inside _generate_sdxl_local.
        if provider != "sdxl_local":
            cache_key = self._get_cache_key(f"{prompt}|scene:{scene_id}")
            cached = self._get_cached_path(cache_key)
            if self._verify_image(cached):
                w, h = self._image_size(cached)
                return ImageAsset(
                    path=cached,
                    prompt=prompt,
                    source="cache",
                    width=w,
                    height=h,
                )

        # Try local assets if configured
        if provider == "local" or provider == "local_assets":
            local = self.get_local_asset(prompt)
            if local:
                w, h = self._image_size(local)
                return ImageAsset(
                    path=local,
                    prompt=prompt,
                    source="local",
                    width=w,
                    height=h,
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