"""Google AI configuration."""

import os

# Google AI / Gemini API Configuration
# Set via environment variable: export GOOGLE_API_KEY="your-key"
# Or get from Google AI Studio: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Gemini model for story generation
GEMINI_MODEL = "gemini-1.5-pro"

# Imagen model for image generation (if available)
IMAGEN_MODEL = "imagen-3.0-generate-002"

# TTS Configuration - Better voices
# Edge-TTS Spanish male voices (mature narrator style):
# - es-ES-AlvaroNeural (Spanish male, mature)
# - es-MX-JorgeNeural (Mexican male, deep) - RECOMMENDED
# - es-AR-TomasNeural (Argentinian male)
# - en-US-GuyNeural (English US, deep male)
# - en-GB-RyanNeural (English UK, mature male)

# ElevenLabs (premium, best quality - requires API key):
# - "pNInz6obpgDQGcFmaJgB" (Adam - deep, mature male)
# - "onwK4e9ZLuTAKqWW03F9" (Daniel - British, mature)
# - "VR6AewLTigWG4xSOukaG" (Arnold - deep, authoritative)

# Video Quality Settings
VIDEO_CRF = 18  # Lower = better quality (18-23 good range)
VIDEO_PRESET = "slow"  # slower = better compression
VIDEO_BITRATE = "12000k"  # Higher bitrate for 1080p