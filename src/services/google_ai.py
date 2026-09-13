"""Google AI services for story generation and image prompts."""

import os
import json
import asyncio
import warnings
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Suppress deprecation warning for google.generativeai BEFORE importing
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.generativeai")

# Try new google.genai first, fallback to deprecated google.generativeai
try:
    from google import genai as genai_new
    GENAI_NEW_AVAILABLE = True
except ImportError:
    GENAI_NEW_AVAILABLE = False

try:
    import google.generativeai as genai_old
    GENAI_OLD_AVAILABLE = True
except ImportError:
    GENAI_OLD_AVAILABLE = False

GENAI_AVAILABLE = GENAI_NEW_AVAILABLE or GENAI_OLD_AVAILABLE

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
        self._use_new_api = False
        self._init_model()

    def _init_model(self):
        if not GENAI_AVAILABLE:
            print("[WARN] google-generativeai not installed. Run: pip install google-generativeai")
            return
        if not self.api_key:
            print("[WARN] GOOGLE_API_KEY not set. Set via environment variable.")
            return
        try:
            # Try new google.genai API first
            if GENAI_NEW_AVAILABLE:
                self._client = genai_new.Client(api_key=self.api_key)
                self._use_new_api = True
                print(f"[OK] Gemini model initialized (new API): {self.model_name}")
            elif GENAI_OLD_AVAILABLE:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self._use_new_api = False
                print(f"[OK] Gemini model initialized (legacy API): {self.model_name}")
            else:
                print("[WARN] No Gemini API available")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Gemini: {e}")

    def generate_story(self, theme: str, num_scenes: int = 5, target_duration_min: float = 4.0) -> Optional[StoicStoryData]:
        """Generate a stoic story using Gemini."""
        if not self.model and not getattr(self, '_client', None):
            return None

        prompt = self._build_story_prompt(theme, num_scenes, target_duration_min)

        try:
            if self._use_new_api:
                # New google.genai API
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_output_tokens": 8192,
                        "response_mime_type": "application/json",
                    }
                )
                response_text = response.text
            else:
                # Legacy google.generativeai API
                import google.generativeai as genai
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
                response_text = response.text

            if response_text:
                data = json.loads(response_text)
                return StoicStoryData(**data)
        except Exception as e:
            print(f"[ERROR] Gemini story generation failed: {e}")

        return None

    def _build_story_prompt(self, theme: str, num_scenes: int, target_duration_min: float) -> str:
        return f"""
Eres un experto en filosofía estoica y guionista de documentales. Genera una historia estoica para un video de YouTube de {target_duration_min} minutos, dividida en {num_scenes} escenas.

TEMA: {theme}

REQUISITOS:
1. Estilo narrativo: Profundo, reflexivo, tono de historiador/narrador maduro, voz grave y apacible
2. Cada escena debe tener:
   - text_overlay: Frase corta (máx 80 chars) para mostrar en pantalla como subtítulo
   - voiceover_text: Narración completa (150-250 palabras por escena), tono de historiador sabio
   - image_prompt: Prompt detallado en INGLÉS para generación de imagen CINEMATOGRÁFICA ÚNICA por escena.
     CADA ESCENA DEBE TENER UNA COMPOSICIÓN VISUAL DIFERENTE:
     - Escena 1: Primer plano dramático (rostro, manos, objeto simbólico)
     - Escena 2: Escena amplia (paisaje, arquitectura, atmósfera)
     - Escena 3: Detalle artístico (bajo relieve, mosaico, manuscrito antiguo)
     - Escena 4: Composición humana (filósofo pensando, escribiendo, caminando)
     - Escena 5: Vista cósmica/filosófica (estrellas, horizonte, perspectiva aérea)
   - estimated_duration: Duración estimada en segundos

3. Estructura narrativa:
   - Escena 1: Gancho (hook) - presenta el problema/pregunta existencial
   - Escena 2: Contexto histórico + enseñanza central del filósofo
   - Escena 3: Profundización en el concepto + analogía visual
   - Escena 4: Aplicación práctica en la vida moderna
   - Escena 5: Reflexión final / cierre inspirador con apertura

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
      "voiceover_text": "Narración completa detallada, tono historiador sabio...",
      "image_prompt": "Detailed UNIQUE cinematic prompt in English for THIS specific scene...",
      "estimated_duration": 45.0
    }}
  ],
  "total_estimated_duration": 180.0
}}

IMPORTANTE:
- voiceover_text: Texto fluido para TTS, tono historiador sabio, voz grave y apacible
- image_prompt en INGLÉS, ÚNICO por escena, muy descriptivo, composición cinematográfica VARIADA
- total_estimated_duration = suma de estimated_duration de todas las escenas
- Sin texto adicional, solo JSON válido
"""

    def enhance_image_prompt(self, base_prompt: str, style: str = "cinematic_stoic", scene_number: int = 1, total_scenes: int = 5) -> str:
        """Enhance image prompt with cinematic stoic style - VARIED per scene."""
        # Base cinematic style
        base_style = (
            "cinematic stoic aesthetic, ancient Greek/Roman philosophy visualization, "
            "dramatic chiaroscuro lighting, volumetric fog, atmospheric mist, "
            "deep shadows, golden hour light rays, weathered stone textures, "
            "timeless atmosphere, 8k resolution, masterpiece, highly detailed, "
            "photorealistic, cinematic composition"
        )
        
        # Scene-specific visual variety
        scene_styles = {
            1: "extreme close-up, macro photography, weathered marble face of philosopher, "
               "deep wrinkles showing wisdom, intense eyes, dramatic side lighting, "
               "shallow depth of field, dust motes in light rays",
            2: "wide cinematic shot, ancient Roman forum at golden hour, "
               "marble columns, atmospheric perspective, volumetric light shafts, "
               "small human figure for scale, epic scale, majestic atmosphere",
            3: "medium shot, ancient manuscript on wooden desk, quill pen, "
               "candlelight flickering, wax seal, illuminated manuscript details, "
               "warm amber tones, intimate scholarly atmosphere, depth of field",
            4: "philosophical figure walking through olive grove at sunset, "
               "Marcus Aurelius or Seneca from behind, contemplative posture, "
               "long shadows, golden light through leaves, cinematic backlighting, "
               "solitary thinker, stoic posture, timeless wisdom",
            5: "cosmic perspective, aerial view from above, "
               "Earth from space at twilight, stars emerging, "
               "pale blue dot, overview effect, philosophical transcendence, "
               "vast universe, tiny human concerns, infinite perspective, awe-inspiring",
        }
        
        # Get scene-specific style or default
        scene_style = scene_styles.get(scene_number, scene_styles[1])
        
        # Style variations
        style_variations = {
            "cinematic_stoic": base_style,
            "dark_moody": "dark moody atmosphere, low key lighting, mysterious shadows, ancient ruins",
            "ethereal": "ethereal divine light, transcendent atmosphere, cosmic perspective, stars and galaxies",
        }
        
        enhancement = style_variations.get(style, base_style)
        scene_specific = scene_style
        
        return f"{base_prompt}, {enhancement}, {scene_specific}, {style} composition"


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