"""Google AI configuration - Solo servicios GRATIS."""

import os

# Google AI / Gemini API Configuration
# Set via environment variable: export GOOGLE_API_KEY="your-key"
# Or get from Google AI Studio: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Gemini model for story generation
GEMINI_MODEL = "gemini-1.5-pro"

# Imagen model for image generation (if available)
IMAGEN_MODEL = "imagen-3.0-generate-002"

# TTS - Solo Edge-TTS (gratis, sin API key)
# Voces masculinas maduras recomendadas:
# - es-MX-JorgeNeural (Mexicano, profundo) - RECOMENDADO
# - es-ES-AlvaroNeural (Español, maduro)
# - es-AR-TomasNeural (Argentino, maduro)
# - en-US-GuyNeural (Inglés US, profundo)
# - en-GB-RyanNeural (Inglés UK, maduro)

# Video Quality Settings
VIDEO_CRF = 18  # Lower = better quality (18-23 good range)
VIDEO_PRESET = "slow"  # slower = better compression
VIDEO_BITRATE = "12000k"  # Higher bitrate for 1080p