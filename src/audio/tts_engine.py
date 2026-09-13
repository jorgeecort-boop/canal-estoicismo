"""Text-to-Speech engine - Solo proveedores GRATIS sin facturación."""

import asyncio
import hashlib
import os
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing: Literal, Optional

from config import get_settings


@dataclass
class TTSResult:
    """Result of TTS generation."""
    audio_path: Path
    duration: float
    text: str
    voice: str
    provider: str
    generated_at: float = time.time()

    def to_dict(self) -> dict:
        return {
            "audio_path": str(self.audio_path),
            "duration": self.duration,
            "text": self.text,
            "voice": self.voice,
            "provider": self.provider,
            "generated_at": self.generated_at,
        }


class TTSEngine:
    """Text-to-Speech engine - Proveedores GRATIS (Edge-TTS + HF + gTTS)."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.tts_settings = self.settings.tts
        self._cache_dir = self.settings.paths.temp_dir / "tts_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._hf_pipeline = None

    def _get_cache_key(self, text: str, voice: str, rate: str) -> str:
        """Generate cache key for text."""
        content = f"{text}|{voice}|{rate}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cached_path(self, cache_key: str) -> Path:
        """Get cached audio path."""
        return self._cache_dir / f"{cache_key}.mp3"

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    async def _generate_edge_tts(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> float:
        """Generate audio using edge-tts Python library (Communicate) - más confiable que CLI."""
        try:
            from edge_tts import Communicate
        except ImportError:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

        voice = voice or self.tts_settings.voice
        rate = rate or self.tts_settings.edge_rate
        volume = volume or self.tts_settings.edge_volume
        # pitch sin "Hz" para edge-tts Python library
        pitch = (pitch or self.tts_settings.edge_pitch).replace("Hz", "")

        communicate = Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
        )
        await communicate.save(str(output_path))

        return self._get_audio_duration(output_path)

    def _generate_gtts(
        self,
        text: str,
        output_path: Path,
        lang: Optional[str] = None,
        tld: Optional[str] = None,
    ) -> float:
        """Generate audio using gTTS (gratis, fallback)."""
        from gtts import gTTS

        lang = lang or self.tts_settings.gtts_lang
        tld = tld or self.tts_settings.gtts_tld

        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        tts.save(str(output_path))

        return self._get_audio_duration(output_path)

    def _get_hf_pipeline(self, model_id: str = "facebook/mms-tts-spa"):
        """Get or create Hugging Face TTS pipeline (cached)."""
        if self._hf_pipeline is None:
            try:
                from transformers import pipeline
                # Suppress warnings
                warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
                self._hf_pipeline = pipeline(
                    "text-to-speech",
                    model=model_id,
                    device=0 if __import__('torch').cuda.is_available() else -1,
                )
                print(f"   [INFO] Hugging Face TTS loaded: {model_id}")
            except Exception as e:
                raise RuntimeError(f"Failed to load HF TTS pipeline: {e}")
        return self._hf_pipeline

    async def _generate_hf_tts(
        self,
        text: str,
        output_path: Path,
        model_id: str = "facebook/mms-tts-spa",
    ) -> float:
        """Generate audio using Hugging Face Transformers TTS (gratis, local)."""
        try:
            import torch
            import torchaudio
        except ImportError:
            raise RuntimeError("torch/torchaudio not installed. Run: pip install torch torchaudio")

        # Get pipeline (cached)
        pipe = self._get_hf_pipeline(model_id)

        try:
            # Generate audio
            # For MMS-TTS, the pipeline returns dict with 'audio' and 'sampling_rate'
            result = pipe(text)
            
            audio_array = result["audio"]
            sampling_rate = result["sampling_rate"]
            
            # Convert to torch tensor if needed
            if isinstance(audio_array, list):
                audio_array = torch.tensor(audio_array)
            elif not isinstance(audio_array, torch.Tensor):
                audio_array = torch.tensor(audio_array)
            
            # Ensure 2D tensor [channels, samples]
            if audio_array.dim() == 1:
                audio_array = audio_array.unsqueeze(0)
            
            # Save as wav first, then convert to mp3
            wav_path = output_path.with_suffix('.wav')
            torchaudio.save(str(wav_path), audio_array, sampling_rate)
            
            # Convert to mp3 using ffmpeg
            cmd = [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-codec:a", "libmp3lame", "-q:a", "2",
                str(output_path)
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            
            # Cleanup wav
            if wav_path.exists():
                wav_path.unlink(missing_ok=True)
            
            return self._get_audio_duration(output_path)
            
        except Exception as e:
            raise RuntimeError(f"HF TTS generation failed: {e}")

    def _generate_gtts(
        self,
        text: str,
        output_path: Path,
        lang: Optional[str] = None,
        tld: Optional[str] = None,
    ) -> float:
        """Generate audio using gTTS (gratis, fallback)."""
        from gtts import gTTS

        lang = lang or self.tts_settings.gtts_lang
        tld = tld or self.tts_settings.gtts_tld

        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        tts.save(str(output_path))

        return self._get_audio_duration(output_path)

    async def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        provider: Optional[Literal["edge-tts", "hf", "gtts"]] = None,
        use_cache: bool = True,
    ) -> TTSResult:
        """Generate audio from text - Proveedores gratis."""

        provider = provider or self.tts_settings.provider
        voice = voice or self.tts_settings.voice

        # Check cache
        cache_key = self._get_cache_key(text, voice, self.tts_settings.rate)
        cached_path = self._get_cached_path(cache_key)

        if use_cache and cached_path.exists():
            duration = self._get_audio_duration(cached_path)
            if duration > 0:
                return TTSResult(
                    audio_path=cached_path,
                    duration=duration,
                    text=text,
                    voice=voice,
                    provider=provider,
                )

        # Generate new audio
        output_path = self.settings.paths.temp_dir / f"tts_{cache_key}.mp3"

        # Provider priority order
        providers_to_try = [provider]
        if provider == "edge-tts":
            providers_to_try.extend(["hf", "gtts"])
        elif provider == "hf":
            providers_to_try.extend(["edge-tts", "gtts"])

        last_error = None
        for prov in providers_to_try:
            try:
                if prov == "edge-tts":
                    duration = await self._generate_edge_tts(text, output_path, voice)
                elif prov == "hf":
                    # Use MMS-TTS Spanish model for best quality
                    duration = await self._generate_hf_tts(text, output_path)
                elif prov == "gtts":
                    duration = self._generate_gtts(text, output_path)
                else:
                    raise ValueError(f"Unknown provider: {prov}")

                if duration <= 0:
                    raise RuntimeError("Generated audio has zero duration")

                # Cache the result
                if use_cache:
                    import shutil
                    shutil.copy2(output_path, cached_path)

                return TTSResult(
                    audio_path=output_path,
                    duration=duration,
                    text=text,
                    voice=voice,
                    provider=prov,
                )

            except Exception as e:
                last_error = e
                print(f"   [WARN] {prov} failed: {e}")
                if output_path.exists():
                    output_path.unlink(missing_ok=True)
                continue

        raise RuntimeError(f"All TTS providers failed. Last error: {last_error}") from last_error

    async def generate_for_scene(
        self,
        scene,
        provider: Optional[Literal["edge-tts", "hf", "gtts"]] = None,
    ) -> TTSResult:
        """Generate audio for a Scene object."""
        result = await self.generate(scene.voiceover_text, voice=self.tts_settings.voice, provider=provider)
        scene.audio_path = result.audio_path
        scene.estimated_duration = result.duration
        return result

    async def generate_for_story(
        self,
        story,
        provider: Optional[Literal["edge-tts", "hf", "gtts"]] = None,
        progress_callback=None,
    ) -> list[TTSResult]:
        """Generate audio for all scenes in a story."""
        results = []
        total = len(story.scenes)

        for i, scene in enumerate(story.scenes):
            if progress_callback:
                progress_callback(i, total, scene.scene_number)

            result = await self.generate_for_scene(scene, provider)
            results.append(result)

        if progress_callback:
            progress_callback(total, total, None)

        return results


# Synchronous wrapper for simpler usage
def generate_tts_sync(
    text: str,
    voice: Optional[str] = None,
    provider: Optional[Literal["edge-tts", "hf", "gtts"]] = None,
    settings=None,
) -> TTSResult:
    """Synchronous TTS generation."""
    engine = TTSEngine(settings)
    return asyncio.run(engine.generate(text, voice, provider))


# Global engine instance
_engine: Optional[TTSEngine] = None


def get_tts_engine(settings=None) -> TTSEngine:
    """Get or create the global TTS engine instance."""
    global _engine
    if _engine is None or settings is not None:
        _engine = TTSEngine(settings)
    return _engine