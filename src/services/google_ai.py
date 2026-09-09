"""Google AI services for story generation and image prompts."""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from config import get_settings
from src.narrative import Scene, Story


@dataclass
class StoicStoryData:
    """Structured story data from Gemini."""
    title: str
    philosopher: str
    theme: str
    scenes: List[Dict[str, Any]]
    total_estimated_duration: float


class GoogleAIService:
    """Service for Google Gemini API integration."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model
        self.model = None
        self._init_model()

    def _init_model(self):
        if not GENAI_AVAILABLE:
            print("[WARN] google-generativeai not installed. Run: pip install google-generativeai")
            return
        if not self.api_key:
            print("[WARN] GOOGLE_API_KEY not set. Set via environment variable.")
            return
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"[OK] Gemini model initialized: {self.model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini: {e}")

    def generate_story(self, theme: str, num_scenes: int = 5, target_duration_min: float = 4.0) -> Optional[StoicStoryData]:
        """Generate a stoic story using Gemini."""
        if not self.model:
            return None

        prompt = self._build_story_prompt(theme, num_scenes, target_duration_min)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                )
            )

            if response.text:
                data = json.loads(response.text)
                return StoicStoryData(**data)
        except Exception as e:
            print(f"[ERROR] Gemini story generation failed: {e}")

        return None

    def _build_story_prompt(self, theme: str, num_scenes: int, target_duration_min: float) -> str:
        return f"""
Eres un experto en filosofía estoica y guionista de documentales. Genera una historia estoica para un video de YouTube de {target_duration_min} minutos, dividida en {num_scenes} escenas.

TEMA: {theme}

REQUISITOS:
1. Estilo narrativo: Profundo, reflexivo, tono de narrador maduro y sabio
2. Cada escena debe tener:
   - text_overlay: Frase corta (máx 80 chars) para mostrar en pantalla
   - voiceover_text: Narración completa (150-250 palabras por escena)
   - image_prompt: Prompt detallado en INGLÉS para generación de imagen cinematográfica (estilo: estatua de mármol, arte clásico, iluminación dramática, niebla volumétrica, sombras, 8k, masterpiece)
   - estimated_duration: Duración estimada en segundos

3. Estructura narrativa:
   - Escena 1: Gancho (hook) - presenta el problema/pregunta
   - Escenas 2-3: Enseñanza central (filósofo + concepto)
   - Escena 4: Aplicación práctica
   - Escena 5: Reflexión final / cierre inspirador

4. Filósofos según tema:
   - control_dichotomy: Epicteto
   - impermanence: Marco Aurelio
   - virtue_ethics: Séneca
   - amor_fati: Nietzsche/Estoicos
   - memento_mori: Estoicos
   - inner_fortress: Marco Aurelio
   - adversity_growth: Séneca
   - present_moment: Marco Aurelio/Epicteto
   - ego_death: Epicteto/Marco Aurelio
   - cosmic_perspective: Marco Aurelio

FORMATO DE SALIDA (JSON estricto):
{{
  "title": "Título del video",
  "philosopher": "Nombre del filósofo principal",
  "theme": "{theme}",
  "scenes": [
    {{
      "scene_number": 1,
      "text_overlay": "Frase corta para pantalla",
      "voiceover_text": "Narración completa detallada...",
      "image_prompt": "Detailed cinematic prompt in English...",
      "estimated_duration": 45.0
    }}
  ],
  "total_estimated_duration": 180.0
}}

IMPORTANTE:
- voiceover_text debe ser texto fluido para TTS (sin saltos de línea, puntuación natural)
- image_prompt en INGLÉS, muy descriptivo, estilo cinematográfico estoico
- total_estimated_duration = suma de estimated_duration de todas las escenas
- Sin texto adicional, solo JSON válido
"""

    def enhance_image_prompt(self, base_prompt: str, style: str = "cinematic_stoic") -> str:
        """Enhance image prompt with cinematic stoic style."""
        styles = {
            "cinematic_stoic": (
                "cinematic stoic aesthetic, ancient Greek/Roman philosophy visualization, "
                "marble statue, classical art, dramatic chiaroscuro lighting, "
                "volumetric fog, atmospheric mist, deep shadows, golden hour light rays, "
                "weathered stone textures, timeless atmosphere, 8k resolution, masterpiece, "
                "highly detailed, photorealistic, Unreal Engine 5 render style"
            ),
            "dark_moody": (
                "dark moody atmosphere, low key lighting, mysterious shadows, "
                "ancient ruins, stone textures, philosophical symbols, "
                "cinematic composition, rule of thirds, 8k"
            ),
            "ethereal": (
                "ethereal divine light, transcendent atmosphere, cosmic perspective, "
                "stars and galaxies, philosophical enlightenment visualization, "
                "soft glow, heavenly rays, 8k masterpiece"
            ),
        }
        enhancement = styles.get(style, styles["cinematic_stoic"])
        return f"{base_prompt}, {enhancement}"


def create_story_from_gemini(gemini_data: StoicStoryData) -> Story:
    """Convert Gemini story data to internal Story object."""
    scenes = []
    for i, scene_data in enumerate(gemini_data.scenes, 1):
        scene = Scene(
            scene_number=scene_data["scene_number"],
            text_overlay=scene_data["text_overlay"],
            voiceover_text=scene_data["voiceover_text"],
            image_prompt=scene_data["image_prompt"],
            estimated_duration=scene_data.get("estimated_duration", 30.0),
        )
        scenes.append(scene)

    total_duration = sum(s.estimated_duration for s in scenes)

    return Story(
        theme=gemini_data.theme,
        title=gemini_data.title,
        philosopher=gemini_data.philosopher,
        scenes=scenes,
        total_estimated_duration=total_duration,
    )