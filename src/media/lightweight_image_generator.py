"""Lightweight image generator using Pollinations/Flux API - Zero VRAM, fast, free."""

import os
import urllib.parse
import requests
from pathlib import Path
from typing import Optional


class LightweightImageGenerator:
    """
    Generates cinematic images via Pollinations/Flux API - Zero VRAM, free, fast.
    No local model loading, no GPU required, works anywhere with internet.
    """

    def __init__(self, output_dir: str = "assets/images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://image.pollinations.ai/prompt/"

    def generate_scene_image(self, prompt: str, scene_id: int, style: str = "cinematic_stoic") -> str:
        """
        Generate a cinematic scene image via Pollinations/Flux.
        
        Args:
            prompt: Base prompt from Gemini
            scene_id: Scene number (used as seed for reproducibility)
            style: Style variant (cinematic_stoic, dark_moody, ethereal)
        """
        # Enhanced prompt for cinematic 35mm film look
        style_enhancements = {
            "cinematic_stoic": (
                "photorealistic 35mm film still, stoic aesthetic, "
                "dramatic chiaroscuro lighting, marble statue texture, deep shadows, "
                "Rembrandt lighting, moody color grading, highly detailed, 8k"
            ),
            "dark_moody": (
                "dark moody atmosphere, low key lighting, mysterious shadows, "
                "ancient ruins, stone textures, philosophical symbols, 8k"
            ),
            "ethereal": (
                "ethereal divine light, transcendent atmosphere, cosmic perspective, "
                "stars and galaxies, philosophical enlightenment, soft glow, 8k"
            ),
        }
        
        enhancement = style_enhancements.get(style, style_enhancements["cinematic_stoic"])
        full_prompt = f"{prompt}, {enhancement}"
        
        encoded_prompt = urllib.parse.quote(full_prompt)

        # Pollinations URL with Flux model, 16:9 HD, reproducible seed
        image_url = (
            f"https://image.pollinations.ai/prompt/{urllib.parse.quote(full_prompt)}"
            f"?width=1920&height=1080&seed={scene_id}&model=flux&nologo=true&enhance=true"
        )

        file_path = self.output_dir / f"scene_{scene_id:03d}.png"

        # Fast download with timeout
        try:
            response = requests.get(image_url, timeout=40, stream=True)
            response.raise_for_status()
            
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(file_path)
            
        except requests.RequestException as e:
            raise RuntimeError(f"Error downloading image for scene {scene_id}: {e}")

    def generate_batch(self, prompts: list, base_scene_id: int = 1) -> list:
        """Generate multiple images sequentially."""
        results = []
        for i, prompt in enumerate(prompts):
            scene_id = base_scene_id + i
            try:
                path = self.generate_scene_image(prompt, scene_id)
                results.append(path)
                print(f"   [OK] Scene {scene_id}: {Path(path).name}")
            except Exception as e:
                print(f"   [ERROR] Scene {scene_id}: {e}")
                raise
        return results


# Backward compatibility alias
ImageGenerator = LightweightImageGenerator