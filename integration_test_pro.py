#!/usr/bin/env python3
"""
Script de Integración Completa - Versión PRO
Genera videos de alta calidad usando:
- Historias: Google Gemini API (guiones profesionales)
- Imágenes: SDXL Turbo (GPU T4) con prompts cinematográficos mejorados
- Voz: Edge-TTS (voces masculinas maduras gratis: es-MX-JorgeNeural)
- Video: MoviePy con transiciones crossfade, Ken Burns suave, 1080p alto bitrate
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.narrative import Scene, Story
from src.audio import TTSEngine
from src.media import ImageManager
from src.video import VideoComposer
from src.services.google_ai import GoogleAIService, create_story_from_gemini
from config import get_settings


class ProIntegrationTest:
    """Test de integración profesional para reels de alta calidad."""

    def __init__(self, output_dir: Path, draft_mode: bool = False, use_gemini: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.draft_mode = draft_mode
        self.use_gemini = use_gemini

        # Configurar voz (solo Edge-TTS gratis)
        self.voice = "es-MX-JorgeNeural" if not draft_mode else "es-ES-AlvaroNeural"
        self.provider = "edge-tts"

        # Settings base
        self.settings = get_settings(draft_mode=draft_mode)

        # Inicializar módulos
        self.story_gen = None
        self.tts_engine = TTSEngine(self.settings)
        self.img_manager = ImageManager(self.settings)
        self.composer = VideoComposer(self.settings)
        
        # Google AI Service
        self.google_ai = None
        if use_gemini:
            self.google_ai = GoogleAIService()

    async def generate_images_sdxl(self, scene: Scene, scene_num: int, style: str = "cinematic_stoic") -> Optional[Path]:
        """Genera imagen usando SDXL Turbo con prompts cinematográficos mejorados."""
        try:
            import torch
            from diffusers import DiffusionPipeline
            from PIL import Image
            from config.google_ai import GoogleAIService as GoogleAIConfig

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   [INFO] Device: {device} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if device == "cuda" else "   [INFO] Device: CPU")

            if device == "cpu":
                print("   [WARN] Sin GPU, saltando SDXL")
                return None

            # Cargar pipeline con optimizaciones de memoria
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            ).to(device)

            # Habilitar attention slicing para ahorrar VRAM
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

            # Prompt mejorado con estilo cinematográfico
            base_prompt = scene.image_prompt
            # Use the enhance_image_prompt from google_ai service
            from src.services.google_ai import GoogleAIService
            enhanced = GoogleAIService().enhance_image_prompt(base_prompt, style)

            negative = "bright, colorful, cartoon, anime, modern, text, watermark, signature, blurry, low quality, distorted, ugly, oversaturated, watermark, username, logo, watermark text"

            # SDXL Turbo: 2-4 steps, guidance_scale=0
            image = pipe(
                prompt=enhanced,
                negative_prompt=negative,
                num_inference_steps=3,  # Un poco más de calidad que 2
                guidance_scale=0.0,
                width=1024,
                height=1024,
            ).images[0]

            # Upscale a 1920x1080 con LANCZOS
            image = image.resize((1920, 1080), Image.LANCZOS)

            output_path = self.settings.paths.temp_dir / f"scene_{scene_num:03d}_sdxl.jpg"
            image.save(output_path, quality=95, optimize=True)
            print(f"   [OK] Imagen SDXL generada: {output_path.name} ({output_path.stat().st_size/1024:.1f} KB)")
            
            # Liberar memoria
            del pipe
            torch.cuda.empty_cache()
            
            return output_path

        except ImportError:
            print("   [INFO] Diffusers no instalado")
        except Exception as e:
            print(f"   [ERROR] SDXL: {e}")
            import traceback
            traceback.print_exc()

        return None

    async def generate_images_pollinations(self, scene: Scene, scene_num: int) -> Optional[Path]:
        """Genera imagen usando Pollinations AI (gratuito, sin key)."""
        import aiohttp
        import urllib.parse

        prompt = scene.image_prompt
        enhanced = f"{prompt}, cinematic stoic aesthetic, marble statue, classical art, dark moody lighting, volumetric fog, dramatic shadows, 8k, masterpiece"
        encoded = urllib.parse.quote(enhanced)

        width, height = self.settings.video.width, self.settings.video.height
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&private=true&enhance=true"

        output_path = self.settings.paths.temp_dir / f"scene_{scene_num:03d}_pollinations.jpg"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        output_path.write_bytes(data)
                        print(f"   [OK] Pollinations: {output_path.name} ({len(data)/1024:.1f} KB)")
                        return output_path
                    else:
                        print(f"   [WARN] Pollinations HTTP {resp.status}")
        except Exception as e:
            print(f"   [ERROR] Pollinations: {e}")

        return None

    async def generate_image(self, scene: Scene, scene_num: int) -> Path:
        """Genera imagen: SDXL (GPU) -> Pollinations -> Placeholder."""
        print(f"\n[IMG] Generando imagen escena {scene_num}...")

        # 1. SDXL en GPU (mejor calidad)
        result = await self.generate_images_sdxl(scene, scene_num)
        if result and result.exists():
            return result

        # 2. Pollinations
        print("   [INFO] SDXL no disponible, intentando Pollinations...")
        result = await self.generate_images_pollinations(scene, scene_num)
        if result and result.exists():
            return result

        # 3. Placeholder
        print("   [WARN] Usando placeholder local")
        placeholder = self.settings.paths.temp_dir / f"scene_{scene_num:03d}_placeholder.jpg"
        self._create_placeholder(placeholder, scene_num)
        return placeholder

    def _create_placeholder(self, path: Path, scene_num: int):
        """Crea imagen placeholder con gradiente y texto."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (1920, 1080), color='#1a1a2e')
            draw = ImageDraw.Draw(img)

            # Gradiente oscuro elegante
            for y in range(1080):
                r = int(26 + (22 - 26) * y / 1080)
                g = int(26 + (33 - 26) * y / 1080)
                b = int(46 + (62 - 46) * y / 1080)
                draw.line([(0, y), (1920, y)], fill=(r, g, b))

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

            sub = "STOIC VIDEO GENERATOR PRO"
            bbox = draw.textbbox((0, 0), sub, font=small_font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((1920-w)//2, (1080-h)//2 + 50), sub, font=small_font, fill='#888')

            img.save(path, quality=90)
        except Exception:
            path.write_bytes(b"")

    async def run_test(self, theme: str = "control_dichotomy", num_scenes: int = 5) -> Path:
        """Ejecuta test end-to-end completo PRO."""
        print("=" * 70)
        print("TEST DE INTEGRACIÓN PRO - REELS ESTOICOS ALTA CALIDAD")
        print("=" * 70)
        print(f"Tema: {theme}")
        print(f"Escenas: {num_scenes}")
        print(f"Modo: {'DRAFT' if self.draft_mode else 'PRODUCCIÓN'}")
        print(f"Gemini: {'Sí' if self.use_gemini else 'No (plantillas locales)'}")
        print(f"ElevenLabs: {'Sí' if self.use_elevenlabs else 'No (Edge-TTS)'}")
        print(f"Salida: {self.output_dir}")
        print("=" * 70)

        # 1. GENERAR HISTORIA
        print("\n[1/5] GENERANDO GUIÓN PROFESIONAL...")
        story = None
        
        if self.use_gemini and self.google_ai and self.google_ai.model:
            print("   Usando Google Gemini API...")
            gemini_data = self.google_ai.generate_story(theme, num_scenes=num_scenes, target_duration_min=4.0)
            if gemini_data:
                story = create_story_from_gemini(gemini_data)
                print(f"   [OK] Guión generado por Gemini: {story.title}")
            else:
                print("   [WARN] Gemini falló, usando plantillas locales")
        
        if not story:
            from src.narrative import StoryGenerator
            local_gen = StoryGenerator(self.settings)
            story = local_gen.generate(theme=theme, num_scenes=num_scenes)
            print(f"   [OK] Guión local: {story.title}")

        print(f"   Título: {story.title}")
        print(f"   Filósofo: {story.philosopher}")
        print(f"   Escenas: {len(story.scenes)}")
        for s in story.scenes:
            print(f"     Escena {s.scene_number}: {s.text_overlay[:70]}...")

        # 2. GENERAR AUDIO (Voz masculina madura)
        print("\n[2/5] GENERANDO NARRACIÓN (Voz masculina madura)...")
        print(f"   Proveedor: {self.provider} | Voz: {self.voice}")
        
        for scene in story.scenes:
            print(f"   Escena {scene.scene_number}...")
            result = await self.tts_engine.generate(scene.voiceover_text, voice=self.voice, provider=self.provider)
            scene.audio_path = result.audio_path
            scene.estimated_duration = result.duration
            print(f"     Duración: {result.duration:.1f}s | Proveedor usado: {result.provider}")

        # 3. GENERAR IMÁGENES (SDXL en GPU)
        print("\n[3/5] GENERANDO IMÁGENES CINEMATOGRÁFICAS (SDXL Turbo)...")
        for i, scene in enumerate(story.scenes, 1):
            img_path = await self.generate_image(scene, i)
            scene.image_path = img_path

        # 4. ENSAMBLAR VIDEO (Alta calidad + transiciones)
        print("\n[4/5] ENSAMBLANDO VIDEO PRO (Crossfade + Ken Burns)...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"stoic_pro_{theme}_{timestamp}.mp4"

        if self.draft_mode:
            for scene in story.scenes:
                scene.estimated_duration = min(scene.estimated_duration, 2.0)

        result = self.composer.compose(story, output_path=output_path, draft_mode=self.draft_mode)

        # 5. RESUMEN
        print("\n[5/5] FINALIZADO")
        print(f"\n{'='*70}")
        print("VIDEO PRO GENERADO")
        print(f"{'='*70}")
        print(f"Video: {result.output_path}")
        print(f"Duración: {result.duration:.1f}s")
        print(f"Escenas: {result.scenes_count}")
        print(f"Resolución: {result.resolution[0]}x{result.resolution[1]} @ {result.fps}fps")
        print(f"Bitrate: {self.settings.video.bitrate} | Preset: {self.settings.video.preset}")
        print(f"Tamaño: {result.file_size / 1e6:.1f} MB")
        print(f"Voz: {self.voice} ({self.provider})")
        print(f"{'='*70}")

        return result.output_path


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test integración PRO - Reels estoicos alta calidad")
    parser.add_argument("--theme", default="control_dichotomy",
                       choices=["control_dichotomy", "impermanence", "virtue_ethics", "amor_fati",
                                "memento_mori", "inner_fortress", "adversity_growth", "present_moment",
                                "ego_death", "cosmic_perspective"])
    parser.add_argument("--scenes", type=int, default=5, help="Número de escenas (3-8)")
    parser.add_argument("--output", default="./output_pro", help="Directorio de salida")
    parser.add_argument("--draft", action="store_true", help="Modo draft (2s/escena)")
    parser.add_argument("--no-gemini", action="store_true", help="Usar plantillas locales en lugar de Gemini")

    args = parser.parse_args()

    output_dir = Path(args.output)
    draft_mode = args.draft
    use_gemini = not args.no_gemini

    # Verificar API keys (solo Gemini)
    if use_gemini and not os.getenv("GOOGLE_API_KEY"):
        print("[WARN] GOOGLE_API_KEY no configurado. Usando plantillas locales.")
        use_gemini = False

    test = ProIntegrationTest(output_dir, draft_mode=draft_mode, use_gemini=use_gemini)

    try:
        video_path = await test.run_test(args.theme, args.scenes)
        print(f"\n[ÉXITO] Video generado: {video_path}")
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
    # Verificar dependencias
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

    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("FFmpeg no encontrado. Instala: sudo apt-get install ffmpeg")
        sys.exit(1)

    sys.exit(asyncio.run(main()))