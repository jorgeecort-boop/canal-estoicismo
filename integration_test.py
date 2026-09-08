#!/usr/bin/env python3
"""
Script de Integración Completa - Test End-to-End
Genera un video de 2 escenas usando:
- Imágenes: Pollinations AI (gratuito, sin API key) / Diffusers SDXL (si hay GPU)
- Voz: Edge-TTS (Microsoft, gratuito)
- Video: MoviePy + FFmpeg
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.narrative import StoryGenerator, Scene, Story
from src.audio import TTSEngine
from src.media import ImageManager
from src.video import VideoComposer
from config import get_settings


class IntegrationTest:
    """Test de integración completa del pipeline."""

    def __init__(self, output_dir: Path, draft_mode: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings = get_settings(draft_mode=draft_mode)
        self.draft_mode = draft_mode

        # Inicializar módulos
        self.story_gen = StoryGenerator(self.settings)
        self.tts_engine = TTSEngine(self.settings)
        self.img_manager = ImageManager(self.settings)
        self.composer = VideoComposer(self.settings)

    async def generate_images_pollinations(self, scene: Scene, scene_num: int) -> Optional[Path]:
        """Genera imagen usando Pollinations AI (gratuito, sin key)."""
        import aiohttp
        import urllib.parse

        # Prompt optimizado para Pollinations
        prompt = scene.image_prompt
        # Añadir estilo cinematográfico estoico
        enhanced = f"{prompt}, cinematic stoic aesthetic, marble statue, classical art, dark moody lighting, volumetric fog, dramatic shadows, 8k, masterpiece"
        encoded = urllib.parse.quote(enhanced)

        # Pollinations URL - ancho x alto
        width, height = self.settings.video.width, self.settings.video.height
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&private=true&enhance=true"

        output_path = self.settings.paths.temp_dir / f"scene_{scene_num:03d}_pollinations.jpg"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        output_path.write_bytes(data)
                        print(f"   [OK] Imagen generada: {output_path.name} ({len(data)/1024:.1f} KB)")
                        return output_path
                    else:
                        print(f"   [WARN] Pollinations HTTP {resp.status}")
        except Exception as e:
            print(f"   [ERROR] Pollinations: {e}")

        return None

    async def generate_images_diffusers(self, scene: Scene, scene_num: int) -> Optional[Path]:
        """Genera imagen usando Diffusers SDXL (requiere GPU)."""
        try:
            import torch
            from diffusers import DiffusionPipeline

            # Verificar GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   [INFO] Device: {device}")

            if device == "cpu":
                print("   [WARN] Sin GPU, saltando Diffusers")
                return None

            # Cargar pipeline (se cachea en ~/.cache/huggingface)
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(device)

            # Prompt optimizado para SDXL Turbo
            prompt = f"{scene.image_prompt}, cinematic stoic aesthetic, marble statue, classical art, dark moody lighting, volumetric fog, dramatic shadows, 8k, masterpiece, highly detailed"
            negative = "bright, colorful, cartoon, anime, modern, text, watermark, signature, blurry, low quality, distorted, ugly, oversaturated"

            # SDXL Turbo: 1-4 steps, guidance_scale=0
            image = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=2,
                guidance_scale=0.0,
                width=1024,
                height=1024,
            ).images[0]

            # Redimensionar a 1920x1080
            image = image.resize((1920, 1080), Image.LANCZOS)

            output_path = self.settings.paths.temp_dir / f"scene_{scene_num:03d}_diffusers.jpg"
            image.save(output_path, quality=95)
            print(f"   [OK] Imagen SDXL generada: {output_path.name}")
            return output_path

        except ImportError:
            print("   [INFO] Diffusers no instalado, saltando")
        except Exception as e:
            print(f"   [ERROR] Diffusers: {e}")

        return None

    async def generate_image(self, scene: Scene, scene_num: int) -> Path:
        """Genera imagen probando Pollinations primero, luego Diffusers."""
        print(f"\n[IMG] Generando imagen escena {scene_num}...")

        # 1. Intentar Pollinations (rápido, gratuito, sin GPU)
        result = await self.generate_images_pollinations(scene, scene_num)
        if result and result.exists():
            return result

        # 2. Fallback a Diffusers si hay GPU
        print("   [INFO] Pollinations falló, intentando Diffusers...")
        result = await self.generate_images_diffusers(scene, scene_num)
        if result and result.exists():
            return result

        # 3. Último recurso: placeholder local
        print("   [WARN] Usando placeholder local")
        placeholder = self.settings.paths.temp_dir / f"scene_{scene_num:03d}_placeholder.jpg"
        self._create_placeholder(placeholder, scene_num)
        return placeholder

    def _create_placeholder(self, path: Path, scene_num: int):
        """Crea imagen placeholder con texto."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (1920, 1080), color='#1a1a2e')
            draw = ImageDraw.Draw(img)

            # Gradiente oscuro
            for y in range(1080):
                r = int(26 + (22 - 26) * y / 1080)
                g = int(26 + (33 - 26) * y / 1080)
                b = int(46 + (62 - 46) * y / 1080)
                draw.line([(0, y), (1920, y)], fill=(r, g, b))

            # Texto
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 72)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 36)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()

            text = f"ESCENA {scene_num}"
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((1920-w)//2, (1080-h)//2 - 50), text, font=font, fill='#f5f0e1')

            sub = "STOIC VIDEO GENERATOR"
            bbox = draw.textbbox((0, 0), sub, font=small_font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((1920-w)//2, (1080-h)//2 + 50), sub, font=small_font, fill='#888')

            img.save(path, quality=90)
        except Exception:
            path.write_bytes(b"")

    async def run_test(self, theme: str = "control_dichotomy") -> Path:
        """Ejecuta test end-to-end completo."""
        print("=" * 60)
        print("TEST DE INTEGRACION END-TO-END")
        print("=" * 60)
        print(f"Tema: {theme}")
        print(f"Modo: {'DRAFT (2s/escena)' if self.draft_mode else 'COMPLETO'}")
        print(f"Salida: {self.output_dir}")

        # 1. GENERAR HISTORIA (2 escenas)
        print("\n[1/4] GENERANDO NARRATIVA...")
        story = self.story_gen.generate(theme=theme, num_scenes=2)
        print(f"   Titulo: {story.title}")
        print(f"   Filosofo: {story.philosopher}")
        print(f"   Escenas: {len(story.scenes)}")
        for s in story.scenes:
            print(f"     Escena {s.scene_number}: {s.text_overlay[:60]}...")

        # 2. GENERAR AUDIO (Edge-TTS)
        print("\n[2/4] GENERANDO AUDIO (Edge-TTS)...")
        for scene in story.scenes:
            print(f"   Escena {scene.scene_number}...")
            result = await self.tts_engine.generate(scene.voiceover_text)
            scene.audio_path = result.audio_path
            scene.estimated_duration = result.duration
            print(f"     Duracion: {result.duration:.1f}s")

        # 3. GENERAR IMÁGENES (Pollinations -> Diffusers -> Placeholder)
        print("\n[3/4] GENERANDO IMAGENES...")
        for i, scene in enumerate(story.scenes, 1):
            img_path = await self.generate_image(scene, i)
            scene.image_path = img_path

        # 4. ENSAMBLAR VIDEO (MoviePy + FFmpeg)
        print("\n[4/4] ENSAMBLANDO VIDEO...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"stoic_integration_{theme}_{timestamp}.mp4"

        # Ajustar duración en draft mode
        if self.draft_mode:
            for scene in story.scenes:
                scene.estimated_duration = min(scene.estimated_duration, 2.0)

        result = self.composer.compose(story, output_path=output_path, draft_mode=self.draft_mode)

        print(f"\n{'='*60}")
        print("TEST COMPLETADO")
        print(f"{'='*60}")
        print(f"Video: {result.output_path}")
        print(f"Duracion: {result.duration:.1f}s")
        print(f"Escenas: {result.scenes_count}")
        print(f"Resolucion: {result.resolution[0]}x{result.resolution[1]}")
        print(f"FPS: {result.fps}")
        print(f"Tamaño: {result.file_size / 1e6:.1f} MB")

        return result.output_path


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test integracion completa - Video estoico 2 escenas")
    parser.add_argument("--theme", default="control_dichotomy",
                       choices=["control_dichotomy", "impermanence", "virtue_ethics", "amor_fati",
                                "memento_mori", "inner_fortress", "adversity_growth", "present_moment",
                                "ego_death", "cosmic_perspective"])
    parser.add_argument("--output", default="./output_integration", help="Directorio de salida")
    parser.add_argument("--full", action="store_true", help="Modo completo (no draft)")
    parser.add_argument("--pollinations-only", action="store_true", help="Solo Pollinations para imágenes")

    args = parser.parse_args()

    output_dir = Path(args.output)
    draft_mode = not args.full

    test = IntegrationTest(output_dir, draft_mode=draft_mode)

    try:
        video_path = await test.run_test(args.theme)
        print(f"\n[EXITO] Video generado: {video_path}")
        return 0
    except KeyboardInterrupt:
        print("\n[INTERRUMPIDO] Por usuario")
        return 130
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Verificar dependencias críticas
    missing = []
    try:
        import aiohttp
    except ImportError:
        missing.append("aiohttp")

    try:
        import moviepy
    except ImportError:
        missing.append("moviepy")

    if missing:
        print(f"Dependencias faltantes: {', '.join(missing)}")
        print("Instala con: pip install " + " ".join(missing))
        sys.exit(1)

    # Verificar ffmpeg
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("FFmpeg no encontrado. Instala: sudo apt-get install ffmpeg")
        sys.exit(1)

    sys.exit(asyncio.run(main()))