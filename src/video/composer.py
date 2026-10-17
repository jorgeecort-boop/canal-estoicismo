"""Video composer for assembling final video from scenes using MoviePy."""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from config import get_settings


@dataclass
class CompositionResult:
    """Result of video composition."""
    output_path: Path
    duration: float
    scenes_count: int
    resolution: tuple[int, int]
    fps: int
    file_size: int
    created_at: float = time.time()

    def to_dict(self) -> dict:
        return {
            "output_path": str(self.output_path),
            "duration": self.duration,
            "scenes_count": self.scenes_count,
            "resolution": self.resolution,
            "fps": self.fps,
            "file_size": self.file_size,
            "created_at": self.created_at,
        }


class VideoComposer:
    """Composes final video from images, audio, and text overlays using MoviePy."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.video_settings = self.settings.video
        self.font_settings = self.settings.font
        self.kenburns_settings = self.settings.kenburns
        self.visual_engine_settings = self.settings.visual_engine
        self._temp_dir = self.settings.paths.temp_dir / "video_composition"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._nvenc_available: Optional[bool] = None

    def _is_nvenc_available(self) -> bool:
        """Probe FFmpeg to check if h264_nvenc encoder is compiled and functional."""
        if self._nvenc_available is not None:
            return self._nvenc_available
        try:
            cmd = ["ffmpeg", "-hide_banner", "-encoders"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "h264_nvenc" in result.stdout:
                # Validate hardware encoding session initialization
                test_cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                    "-c:v", "h264_nvenc", "-f", "null", "-",
                ]
                probe = subprocess.run(test_cmd, capture_output=True, timeout=10)
                self._nvenc_available = (probe.returncode == 0)
                return self._nvenc_available
        except Exception:
            pass
        self._nvenc_available = False
        return False

    def _resolve_video_codec(self) -> tuple[str, str, list[str]]:
        """Resolve video codec and preset, checking for NVENC support if requested."""
        if getattr(self.video_settings, "nvenc", False):
            if self._is_nvenc_available():
                return "h264_nvenc", "p5", ["-tune", "hq"]
            print("   [INFO] h264_nvenc requested but not available in FFmpeg/hardware; falling back to libx264")
        return self.video_settings.codec, self.video_settings.preset, []

    def _encoder_threads(self) -> Optional[int]:
        """Encoder thread count for libx264, or None to let ffmpeg decide.

        ``threads=4`` protects a 2-vCPU Colab T4; a 12-vCPU A100 runtime
        wants every core, which is what 0 in the profile means.
        """
        threads = getattr(self.video_settings, "threads", 4)
        return None if not threads else int(threads)

    def _build_cine_filter(self) -> str:
        """Build the optional FFmpeg cinematic filter chain."""
        cine = self.settings.cine
        if not cine.enabled:
            return ""
        return (
            f"eq=saturation={cine.saturation}:contrast={cine.contrast},"
            f"vignette=PI/5,noise=alls={cine.grain}:allf=t"
        )

    def _xfade_transition(self, index: int) -> str:
        """Return a deterministic transition name for a scene boundary."""
        return ("fade", "dissolve", "wipeleft", "slideright")[index % 4]

    def _mix_music_bed(self, video_path: Path) -> bool:
        """Mix assets/music.mp3 under narration when an optional bed exists."""
        music_path = self.settings.paths.assets_dir / "music.mp3"
        if not music_path.exists():
            return True

        mixed_path = video_path.with_suffix(".music.mp4")
        filter_complex = (
            "[1:a]volume=0.06[bg];"
            "[bg][0:a]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[duckedbg];"
            "[0:a][duckedbg]amix=inputs=2:duration=first:dropout_transition=2[a]"
        )
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path), "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", filter_complex, "-map", "0:v:0", "-map", "[a]",
            "-c:v", "copy", "-c:a", self.video_settings.audio_codec,
            "-b:a", self.video_settings.audio_bitrate, "-shortest", str(mixed_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0 and mixed_path.exists():
                mixed_path.replace(video_path)
                return True
            print("   [WARN] Optional music bed failed; keeping narration-only audio")
        except Exception as exc:
            print(f"   [WARN] Optional music bed skipped: {exc}")
        mixed_path.unlink(missing_ok=True)
        return False

    def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

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

    def _pad_scene_audio(self, audio_path: Path, extra_seconds: float = 0.6) -> Path:
        """Append a breathing pause so narration never runs scene-to-scene.

        Returns the (possibly new) audio path. Idempotent per output name.
        """
        if extra_seconds <= 0:
            return audio_path
        # The container must match the audio codec: writing the Colab default
        # (aac) into a `.mp3` file makes ffmpeg fail, which is why every scene
        # used to print "Audio padding failed" and the breathing pause between
        # scenes silently disappeared.
        codec = str(self.video_settings.audio_codec).lower()
        container = ".mp3" if "mp3" in codec else ".m4a"
        padded_path = audio_path.with_name(f"{audio_path.stem}.pad{container}")
        if padded_path.exists():
            return padded_path
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-af", f"apad=pad_dur={extra_seconds}",
            "-c:a", self.video_settings.audio_codec,
            "-b:a", self.video_settings.audio_bitrate,
            str(padded_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and padded_path.exists():
                return padded_path
            print("   [WARN] Audio padding failed, using unpadded audio")
        except Exception as exc:
            print(f"   [WARN] Audio padding skipped: {exc}")
        padded_path.unlink(missing_ok=True)
        return audio_path

    def _create_placeholder_image(self, path: Path) -> None:
        """Create a placeholder image using PIL."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (self.video_settings.width, self.video_settings.height), color='#1a1a2e')
            draw = ImageDraw.Draw(img)

            # Dark gradient background
            for y in range(self.video_settings.height):
                r = int(26 + (22 - 26) * y / self.video_settings.height)
                g = int(26 + (33 - 26) * y / self.video_settings.height)
                b = int(46 + (62 - 46) * y / self.video_settings.height)
                draw.line([(0, y), (self.video_settings.width, y)], fill=(r, g, b))

            # Text
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/georgia.ttf", 72)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 72)
                except:
                    font = ImageFont.load_default()

            text = "STOIC VIDEO"
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((self.video_settings.width - w) // 2, (self.video_settings.height - h) // 2), text, font=font, fill='#f5f0e1')

            img.save(path, quality=90)
        except Exception:
            # Minimal fallback
            from PIL import Image
            Image.new('RGB', (self.video_settings.width, self.video_settings.height), color='#1a1a2e').save(path)

    def _build_kenburns_filter(
        self,
        width: int,
        height: int,
        duration: float,
        direction: str = "random",
    ) -> str:
        """Build FFmpeg filter for Ken Burns effect using zoompan."""
        zoom = self.kenburns_settings.zoom_factor
        total_frames = int(duration * self.video_settings.fps)

        if direction == "random":
            import random
            if height > width:
                direction = random.choice(["zoom_in", "zoom_out", "pan_up", "pan_down"])
            else:
                direction = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])

        if direction == "zoom_in":
            zoom_expr = f"zoom=1+({zoom}-1)*on/{total_frames}"
            x_expr = f"x='iw/2-(iw/zoom/2)'"
            y_expr = f"y='ih/2-(ih/zoom/2)'"
        elif direction == "zoom_out":
            zoom_expr = f"zoom={zoom}-({zoom}-1)*on/{total_frames}"
            x_expr = f"x='iw/2-(iw/zoom/2)'"
            y_expr = f"y='ih/2-(ih/zoom/2)'"
        elif direction == "pan_left":
            zoom_expr = f"zoom={zoom}"
            x_expr = f"x='(iw-iw/zoom)*(1-on/{total_frames})'"
            y_expr = f"y='ih/2-(ih/zoom/2)'"
        elif direction == "pan_right":
            zoom_expr = f"zoom={zoom}"
            x_expr = f"x='(iw-iw/zoom)*on/{total_frames}'"
            y_expr = f"y='ih/2-(ih/zoom/2)'"
        elif direction == "pan_up":
            zoom_expr = f"zoom={zoom}"
            x_expr = f"x='iw/2-(iw/zoom/2)'"
            y_expr = f"y='(ih-ih/zoom)*(1-on/{total_frames})'"
        elif direction == "pan_down":
            zoom_expr = f"zoom={zoom}"
            x_expr = f"x='iw/2-(iw/zoom/2)'"
            y_expr = f"y='(ih-ih/zoom)*on/{total_frames}'"
        else:
            zoom_expr = f"zoom=1+({zoom}-1)*on/{total_frames}"
            x_expr = f"x='iw/2-(iw/zoom/2)'"
            y_expr = f"y='ih/2-(ih/zoom/2)'"

        return f"zoompan={zoom_expr}:{x_expr}:{y_expr}:d={total_frames}:s={width}x{height}:fps={self.video_settings.fps}"

    def _build_text_filter(
        self,
        text: str,
        width: int,
        height: int,
        duration: float,
        start: float = 0.0,
        subtitle_text: str | None = None,
    ) -> str:
        """Build timed FFmpeg subtitle cards.

        When the narration is available, cards are made from the spoken text
        and their durations are proportional to their word counts. The old
        overlay-only behaviour remains as a compatibility fallback.
        """
        cards = (
            self._timed_subtitle_cards(subtitle_text, duration, width=width)
            if subtitle_text
            else [(card, start + index * duration / max(1, len(self._overlay_cards(text))),
                   start + (index + 1) * duration / max(1, len(self._overlay_cards(text))))
                  for index, card in enumerate(self._overlay_cards(text))]
        )
        filters = []
        for card, begin, end in cards:
            filters.append(self._build_single_drawtext(card, width, height, begin, end))
        return ",".join(filters)

    def _measure_text(self, text: str, font_size: int) -> float:
        """Width in pixels of ``text`` at ``font_size`` with the real face."""
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(self.font_settings.font_path, int(font_size))
            return float(font.getlength(text))
        except Exception:
            # Crude but safe: serif faces average a bit over half an em.
            return len(text) * font_size * 0.55

    def _wrap_to_canvas(self, text: str, font_size: int, max_px: float) -> str:
        """Greedy word wrap so every line fits inside ``max_px`` pixels.

        Counting characters is not enough: "el estoicismo, el cosmos no comete
        error" is 44 characters and measured 1283 px at size 72 inside a
        1080 px canvas, so the drawtext renderer clipped it on both sides.
        """
        words = str(text or "").split()
        if not words:
            return ""
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self._measure_text(candidate, font_size) <= max_px:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\n".join(lines)

    def _fit_subtitle(self, text: str, width: int, font_size: int) -> tuple[str, int]:
        """Wrap a subtitle card, shrinking the face a little when needed.

        Words are never dropped: narration must stay readable, so a card that
        would need a third line renders slightly smaller instead.
        """
        max_px = width * float(getattr(self.font_settings, "width_fraction", 0.90))
        max_lines = max(1, int(self.font_settings.max_lines))
        size = int(font_size)
        floor = max(24, int(font_size * 0.72))
        wrapped = self._wrap_to_canvas(text, size, max_px)
        while size > floor and wrapped.count("\n") + 1 > max_lines:
            size = int(size * 0.92)
            wrapped = self._wrap_to_canvas(text, size, max_px)
        return wrapped, size

    def _build_single_drawtext(
        self,
        text: str,
        width: int,
        height: int,
        start: float,
        end: float,
    ) -> str:
        """Build the drawtext filter(s) for one timed subtitle card.

        drawtext has no wrapping, and its ``\\n`` escape is **not** a line break
        in current builds: it renders a literal "n" ("linea unonlinea dos" was
        measured locally), while a real newline leaves a missing-glyph box. So
        every wrapped line becomes its own filter, centred and stacked upwards
        from ``margin_bottom``.
        """
        import sys

        font_size = int(self.font_settings.size)
        wrapped, font_size = self._fit_subtitle(text, width, font_size)
        lines = [line for line in wrapped.split("\n") if line.strip()] or ["…"]

        font_color = self.font_settings.color.lstrip("#")
        stroke_color = self.font_settings.stroke_color.lstrip("#")
        stroke_width = self.font_settings.stroke_width
        margin_bottom = self.font_settings.margin_bottom
        line_step = max(1.0, font_size * float(self.font_settings.line_spacing))

        font_path = self.font_settings.font_path.replace("\\", "/")
        if sys.platform == "win32" and ":" in font_path:
            font_path = font_path.replace(":", "\\:")

        filters = []
        for index, line in enumerate(lines):
            escaped = (
                line.replace("'", r"'\''")
                .replace(":", r"\:")
                .replace("%", r"\%")
            )
            above = len(lines) - 1 - index
            if self.font_settings.position == "bottom":
                y_pos = f"h-text_h-{margin_bottom + above * line_step:.0f}"
            elif self.font_settings.position == "top":
                y_pos = f"{margin_bottom + index * line_step:.0f}"
            else:
                block = len(lines) * line_step
                y_pos = f"(h-{block:.0f})/2+{index * line_step:.0f}"

            filters.append(
                f"drawtext="
                f"text='{escaped}':"
                f"fontfile='{font_path}':"
                f"fontsize={font_size}:"
                f"fontcolor=0x{font_color}:"
                f"bordercolor=0x{stroke_color}:"
                f"borderw={stroke_width}:"
                f"x=(w-text_w)/2:"
                f"y={y_pos}:"
                f"line_spacing={self.font_settings.line_spacing}:"
                f"enable='between(t,{start},{end})'"
            )
        return ",".join(filters)

    def _format_overlay_text(self, text: str) -> str:
        """Wrap and cap subtitles so they remain inside the video safe area."""
        import textwrap

        clean = " ".join(str(text or "").split())
        if not clean:
            return ""

        width = max(12, int(self.font_settings.max_chars_per_line))
        max_lines = max(1, int(self.font_settings.max_lines))
        lines = textwrap.wrap(
            clean,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )

        if len(lines) > max_lines:
            lines = lines[:max_lines]
            tail = lines[-1].rstrip(" .,:;!?…")
            suffix = "…"
            lines[-1] = (tail[: max(1, width - len(suffix))] + suffix).rstrip()

        return "\n".join(lines)

    def _overlay_cards(self, text_overlay: str) -> list[str]:
        """Split a formatted overlay into 1-2 timed cards.

        Two-line overlays become one card per line; a long single line is
        split by words. Short phrases stay a single card (no flicker).
        """
        lines = self._format_overlay_text(text_overlay).split("\n")
        lines = [line for line in lines if line.strip()]
        if len(lines) >= 2:
            return lines[:2]
        if len(lines) == 1 and len(lines[0]) > 30:
            words = lines[0].split()
            half = max(1, len(words) // 2)
            return [" ".join(words[:half]), " ".join(words[half:])]
        return lines if lines else ["…"]

    def _timed_subtitle_cards(
        self, narration: str, duration: float, width: int | None = None
    ) -> list[tuple[str, float, float]]:
        """Split narration into readable cards timed by spoken word count.

        ``width`` (the canvas in pixels) budgets each card from the *measured*
        font instead of counting characters: a 44 character card measured
        1283 px inside a 1080 px canvas and used to be clipped.
        """
        import re

        clean = " ".join(str(narration or "").split())
        if not clean:
            return [("…", 0.0, max(0.1, duration))]

        max_lines = max(1, int(self.font_settings.max_lines))
        font_size = int(self.font_settings.size)
        if width:
            reference = self._measure_text("abcdefghij", font_size) / 10.0
            per_line = max(
                12,
                int(
                    width * float(self.font_settings.width_fraction)
                    / max(1.0, reference)
                    * 0.85          # word boundaries never pack as tightly
                ),
            )
            max_chars = per_line * max_lines
        else:
            max_chars = max(24, int(self.font_settings.max_chars_per_line)) * max_lines
        # Break at sentence ends as soon as a card is substantial, and never
        # let a card grow past what the two lines can hold. The previous rule
        # required 7 words before a punctuation break, so "…ocurre. Después…"
        # never split and a whole 10 s narration sat in one static card.
        min_sentence_words = 4
        max_words_per_card = max(8, max_chars // 6)
        words = clean.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            projected = current_len + (1 if current else 0) + len(word)
            current.append(word)
            current_len = projected
            sentence_end = bool(re.search(r"[.!?…]$", word))
            if (
                current_len >= max_chars
                or len(current) >= max_words_per_card
                or (sentence_end and len(current) >= min_sentence_words)
            ):
                chunks.append(" ".join(current))
                current, current_len = [], 0
        if current:
            chunks.append(" ".join(current))

        total_words = max(1, len(words))
        cursor = 0.0
        timed: list[tuple[str, float, float]] = []
        for index, chunk in enumerate(chunks):
            count = len(chunk.split())
            card_duration = duration * count / total_words
            end = duration if index == len(chunks) - 1 else cursor + card_duration
            # Keep the whole chunk: wrapping to the canvas happens at render
            # time, while formatting it here used to drop the card's tail
            # behind an ellipsis ("la misma manera los obstaculos las
            # traiciones…" lost the rest of the narration).
            timed.append((" ".join(chunk.split()), cursor, end))
            cursor = end
        return timed

    def _ambient_fx(self, frame, t: float, scene_number: int, particles):
        """Add optional particles, smoke, and lamp-like light fluctuation."""
        if not self.visual_engine_settings.enabled:
            return frame

        from PIL import Image, ImageDraw, ImageEnhance
        import numpy as np

        settings = self.visual_engine_settings
        height, width = frame.shape[:2]
        result = Image.fromarray(frame.astype(np.uint8), mode="RGB")

        # Particles drift with deterministic Gaussian velocities so renders remain reproducible.
        particle_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(particle_layer)
        for x, y, vx, vy, radius, alpha in particles:
            px = (x + vx * t) % width
            py = (y + vy * t) % height
            a = int(alpha * (0.65 + 0.35 * np.sin(t * 1.7 + x)))
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=(245, 230, 190, max(0, a)))

        # A few broad translucent ellipses provide restrained rising smoke.
        smoke_alpha = int(255 * settings.smoke_alpha * (0.5 + 0.5 * np.sin(t * 0.8)))
        smoke_x = int(width * (0.68 + 0.04 * np.sin(t * 0.35)))
        smoke_y = int(height * (0.72 - 0.08 * (t % 8) / 8))
        draw.ellipse((smoke_x - width * 0.12, smoke_y - height * 0.20,
                      smoke_x + width * 0.12, smoke_y + height * 0.20),
                     fill=(190, 190, 180, max(0, smoke_alpha)))
        result = Image.alpha_composite(result.convert("RGBA"), particle_layer).convert("RGB")

        flicker = 1.0 + settings.lamp_flicker * np.sin(t * 2.4 + scene_number)
        return np.asarray(ImageEnhance.Brightness(result).enhance(float(flicker)))

    def _make_particles(self, width: int, height: int, scene_number: int):
        """Create deterministic ambient particle parameters for one scene."""
        import numpy as np

        settings = self.visual_engine_settings
        rng = np.random.default_rng(settings.seed + scene_number)
        count = max(0, settings.particle_count)
        positions = rng.normal((width / 2, height / 2), (width * 0.36, height * 0.36), (count, 2))
        velocities = rng.normal(0, (width * 0.008, height * 0.004), (count, 2))
        radii = rng.uniform(1.5, 4.0, count)
        alphas = rng.uniform(60.0, 120.0, count)
        return list(zip(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], radii, alphas))

    @staticmethod
    def _cine_vignette_map(width: int, height: int, strength: float):
        """Static radial vignette used by the cinematic grade.

        It only depends on the canvas and the grade strength, so building it
        once per scene instead of once per frame takes a sqrt() over the whole
        frame (2M pixels at 1080x1920) out of the hot loop.
        """
        import numpy as np

        yy, xx = np.ogrid[:height, :width]
        distance = np.sqrt(
            ((xx - width / 2) / (width / 2)) ** 2
            + ((yy - height / 2) / (height / 2)) ** 2
        )
        return np.clip(
            1.0 - float(strength) * np.maximum(distance - 0.15, 0.0), 0.72, 1.0
        ).astype(np.float32)

    def _apply_kenburns_and_text(self, image_path: Path, audio_path: Path, text_overlay: str,
                                  output_path: Path, scene_duration: float, scene_number: int,
                                  voiceover_text: str | None = None) -> bool:
        """Apply Ken Burns effect and text overlay using MoviePy (v2.x)."""
        try:
            # MoviePy imports fix
            try:
                from moviepy import VideoClip, ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip, AudioFileClip
            except ImportError:
                from moviepy.video.VideoClip import VideoClip, ImageClip, TextClip
                from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
                from moviepy.video.compositing.concatenate import concatenate_videoclips
                from moviepy.video.io.VideoFileClip import VideoFileClip
                from moviepy.audio.io.AudioFileClip import AudioFileClip
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            import random
            import os

            # Load image
            img = Image.open(image_path)
            img_w, img_h = img.size
            target_w, target_h = self.video_settings.width, self.video_settings.height

            particles = (
                self._make_particles(target_w, target_h, scene_number)
                if self.visual_engine_settings.enabled else ()
            )

            # Static per-scene layers, built once instead of once per frame:
            # the cinematic vignette (a sqrt over the whole frame) and the
            # visual-engine subject mask (fixed ellipse geometry).
            cine = self.settings.cine
            cine_on = bool(cine.enabled)
            vignette_map = (
                self._cine_vignette_map(target_w, target_h, cine.vignette)
                if cine_on else None
            )
            subject_mask = None
            if self.visual_engine_settings.enabled:
                subject_mask = Image.new("L", (target_w, target_h), 0)
                ImageDraw.Draw(subject_mask).ellipse(
                    (target_w * 0.08, target_h * 0.02, target_w * 0.92, target_h * 1.02),
                    fill=235,
                )

            # Ken Burns effect parameters (deterministic per scene for reproducibility)
            zoom = self.kenburns_settings.zoom_factor
            direction = self.kenburns_settings.direction
            if direction == "random":
                direction = random.Random(scene_number).choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])

            # Create video clip from frames using VideoClip with lazy make_frame function
            def make_frame(t):
                # Calculate progress (0.0 to 1.0)
                progress = t / max(0.001, scene_duration)
                progress = min(1.0, max(0.0, progress))
                
                # Calculate zoom and position based on direction (slight diagonal drift on zooms)
                drift = (img_h / zoom / 2) * 0.12 * progress
                if direction == "zoom_in":
                    curr_zoom = 1 + (zoom - 1) * progress
                    cx, cy = img_w / 2, img_h / 2 + drift
                elif direction == "zoom_out":
                    curr_zoom = zoom - (zoom - 1) * progress
                    cx, cy = img_w / 2, img_h / 2 - drift
                elif direction == "pan_left":
                    curr_zoom = zoom
                    cx = img_w / 2 - (img_w / zoom / 2) * (1 - progress)
                    cy = img_h / 2
                elif direction == "pan_right":
                    curr_zoom = zoom
                    cx = img_w / 2 + (img_w / zoom / 2) * progress
                    cy = img_h / 2
                else:
                    curr_zoom = 1 + (zoom - 1) * progress
                    cx, cy = img_w / 2, img_h / 2

                # Calculate crop box
                crop_w = int(img_w / curr_zoom)
                crop_h = int(img_h / curr_zoom)
                left = max(0, min(int(cx - crop_w / 2), img_w - crop_w))
                top = max(0, min(int(cy - crop_h / 2), img_h - crop_h))
                
                # Crop and resize. The optional visual engine uses a slower background
                # crop and a centered, feathered subject crop as a lightweight 2.5D layer.
                cropped = img.crop((left, top, left + crop_w, top + crop_h))
                resized = cropped.resize((target_w, target_h), Image.LANCZOS)
                if self.visual_engine_settings.enabled:
                    bg_zoom = max(1.0, curr_zoom * 0.94)
                    bg_w = max(1, int(img_w / bg_zoom))
                    bg_h = max(1, int(img_h / bg_zoom))
                    bg_progress = progress * 0.5
                    bg_left = max(0, min(int(cx - bg_w / 2 + (img_w - bg_w) * 0.12 * bg_progress), img_w - bg_w))
                    bg_top = max(0, min(int(cy - bg_h / 2 + (img_h - bg_h) * 0.06 * bg_progress), img_h - bg_h))
                    background = img.crop((bg_left, bg_top, bg_left + bg_w, bg_top + bg_h)).resize((target_w, target_h), Image.LANCZOS).convert("RGBA")
                    subject = resized.convert("RGBA")
                    subject.putalpha(subject_mask)
                    resized = Image.alpha_composite(background, subject).convert("RGB")

                if cine_on:
                    from PIL import ImageEnhance
                    resized = ImageEnhance.Color(resized).enhance(cine.saturation)
                    resized = ImageEnhance.Contrast(resized).enhance(cine.contrast)
                    frame = np.asarray(resized).astype(np.float32)
                    frame *= vignette_map[..., None]
                    if cine.grain:
                        rng = np.random.default_rng(int(t * self.video_settings.fps) + scene_number)
                        frame += rng.normal(0, cine.grain / 2, frame.shape[:2])[..., None]
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                    return self._ambient_fx(frame, t, scene_number, particles)
                
                frame = np.array(resized)
                return self._ambient_fx(frame, t, scene_number, particles)
            
            clip = VideoClip(make_frame, duration=scene_duration)
            clip.fps = self.video_settings.fps

            # Add text overlay using MoviePy's TextClip (one timed card
            # per overlay half so subtitles move with the narration)
            try:
                # MoviePy 2.x compatible TextClip - avoid font parameter conflict
                cards = (
                    self._timed_subtitle_cards(
                        voiceover_text, scene_duration, width=self.video_settings.width
                    )
                    if voiceover_text
                    else [
                        (card, index * scene_duration / max(1, len(self._overlay_cards(text_overlay))),
                         (index + 1) * scene_duration / max(1, len(self._overlay_cards(text_overlay))))
                        for index, card in enumerate(self._overlay_cards(text_overlay))
                    ]
                )
                txt_clips = []
                for card, start, end in cards:
                    txt_clip = TextClip(
                        text=card,
                        font_size=self.font_settings.size,
                        color=self.font_settings.color,
                        stroke_color=self.font_settings.stroke_color,
                        stroke_width=self.font_settings.stroke_width,
                        font=self.font_settings.windows_font_path if os.name == 'nt' else self.font_settings.linux_font_path,
                        method='caption',
                        size=(max(200, target_w - 140), 380) if target_h > target_w else (target_w - 200, 260),
                    ).with_duration(max(0.05, end - start)).with_start(start)

                    # Position by the block's bottom edge.  MoviePy interprets a
                    # numeric y position as the top of the TextClip; placing the
                    # top at ``target_h - margin_bottom`` clips multi-line text.
                    if self.font_settings.position == "bottom":
                        y_position = max(
                            0,
                            target_h - self.font_settings.margin_bottom - txt_clip.h,
                        )
                        txt_clip = txt_clip.with_position(('center', y_position))
                    elif self.font_settings.position == "top":
                        txt_clip = txt_clip.with_position(('center', self.font_settings.margin_bottom))
                    else:
                        txt_clip = txt_clip.with_position(('center', 'center'))
                    txt_clips.append(txt_clip)

                video = CompositeVideoClip([clip, *txt_clips])
            except Exception as e:
                print(f"   [WARN] TextClip failed, using video without text: {e}")
                video = clip

            # Add audio
            audio = AudioFileClip(str(audio_path))
            video = video.with_audio(audio)

            # Write output
            codec, preset, extra_args = self._resolve_video_codec()
            video.write_videofile(
                str(output_path),
                fps=self.video_settings.fps,
                codec=codec,
                bitrate=self.video_settings.bitrate,
                audio_codec=self.video_settings.audio_codec,
                audio_bitrate=self.video_settings.audio_bitrate,
                preset=preset,
                ffmpeg_params=extra_args if extra_args else None,
                threads=self._encoder_threads(),
                logger=None,
            )

            return True

        except Exception as e:
            print(f"   [ERROR] MoviePy composition failed: {e}")
            # Fallback to FFmpeg
            return self._compose_scene_ffmpeg_fallback(
                image_path, audio_path, text_overlay, output_path, scene_duration,
                scene_number, voiceover_text,
            )

    def _build_scene_filter_complex(
        self,
        width: int,
        height: int,
        duration: float,
        direction: str,
        text_overlay: str,
        voiceover_text: str | None = None,
    ) -> str:
        """Full per-scene filter: zoompan straight to the canvas, then text.

        The old chain ran zoompan at half resolution and upscaled afterwards,
        which doubled every integer crop step on screen (a slow pan moved 1 px
        of the 540 px canvas = 2 px of the reel, every ~5 frames: the picture
        froze and then jumped). The still is now pre-upscaled once by
        ``kenburns.upscale``, so an integer crop step is a sub-pixel move and
        the text stays crisp because drawtext still runs last.
        """
        if self.kenburns_settings.enabled:
            if direction == "random":
                import random
                if height > width:
                    direction = random.choice(["zoom_in", "zoom_out", "pan_up", "pan_down"])
                else:
                    direction = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])
            kenburns = self._build_kenburns_filter(
                width, height, duration, direction,
            )
            text_filter = self._build_text_filter(
                text_overlay, width, height, duration, subtitle_text=voiceover_text,
            )
            filter_complex = f"[0:v]{kenburns}[kb];[kb]{text_filter}[v]"
        else:
            text_filter = self._build_text_filter(
                text_overlay, width, height, duration, subtitle_text=voiceover_text,
            )
            filter_complex = f"[0:v]{text_filter}[v]"

        cine_filter = self._build_cine_filter()
        if cine_filter:
            filter_complex = filter_complex.replace("[v]", f",{cine_filter}[v]", 1)
        return filter_complex

    def _prepared_still(self, image_path: Path, scene_number: int) -> Path:
        """Return the still enlarged ``kenburns.upscale`` times (or unchanged).

        zoompan moves an integer crop: on a 1080 px still a slow pan advanced a
        pixel every ~5 frames, so the picture stalled and then jumped (measured
        frame by frame on the delivered reel). Enlarging the still once turns
        that integer step into a sub-pixel move on screen.
        """
        factor = int(getattr(self.kenburns_settings, "upscale", 1) or 1)
        if factor <= 1 or not self.kenburns_settings.enabled:
            return image_path
        out = self._temp_dir / f"still_{scene_number:03d}_{factor}x.png"
        if out.exists():
            return out
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", str(image_path),
            "-vf", f"scale={self.video_settings.width * factor}:"
                   f"{self.video_settings.height * factor}:flags=lanczos",
            "-frames:v", "1", str(out),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0 and out.exists():
                return out
            print("   [WARN] still upscale failed, zoompan uses the original size")
        except Exception:
            pass
        return image_path

    def _compose_scene_ffmpeg_fallback(self, image_path: Path, audio_path: Path, text_overlay: str,
                                        output_path: Path, scene_duration: float, scene_number: int,
                                        voiceover_text: str | None = None) -> bool:
        """Compose a single scene using FFmpeg directly (fallback)."""
        image_path = self._prepared_still(image_path, scene_number)
        # Build filter complex
        filter_complex = self._build_scene_filter_complex(
            self.video_settings.width,
            self.video_settings.height,
            scene_duration,
            self.kenburns_settings.direction,
            text_overlay,
            voiceover_text,
        )

        codec, preset, extra_args = self._resolve_video_codec()
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", codec,
            "-preset", preset,
            *extra_args,
            "-b:v", self.video_settings.bitrate,
            "-c:a", self.video_settings.audio_codec,
            "-b:a", self.video_settings.audio_bitrate,
            "-r", str(self.video_settings.fps),
            "-t", str(scene_duration),
            "-shortest",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"Scene {scene_number} ffmpeg failed: {(result.stderr or '')[-500:]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"Scene {scene_number} composition timed out")
            return False
        except Exception as e:
            print(f"Scene {scene_number} composition failed: {e}")
            return False

    def _compose_scene_ffmpeg(self, image_path: Path, audio_path: Path, text_overlay: str,
                               output_path: Path, scene_duration: float, scene_number: int,
                               voiceover_text: str | None = None) -> bool:
        """Compose a single scene using FFmpeg directly."""
        image_path = self._prepared_still(image_path, scene_number)
        # Build filter complex (shared helper: pre-upscaled still + zoompan to canvas)
        filter_complex = self._build_scene_filter_complex(
            self.video_settings.width,
            self.video_settings.height,
            scene_duration,
            self.kenburns_settings.direction,
            text_overlay,
            voiceover_text,
        )

        codec, preset, extra_args = self._resolve_video_codec()
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", codec,
            "-preset", preset,
            *extra_args,
            "-b:v", self.video_settings.bitrate,
            "-c:a", self.video_settings.audio_codec,
            "-b:a", self.video_settings.audio_bitrate,
            "-r", str(self.video_settings.fps),
            "-t", str(scene_duration),
            "-shortest",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"Scene {scene_number} ffmpeg failed: {(result.stderr or '')[-500:]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"Scene {scene_number} composition timed out")
            return False
        except Exception as e:
            print(f"Scene {scene_number} composition failed: {e}")
            return False

    def _concatenate_videos(self, scene_paths: list[Path], output_path: Path) -> bool:
        """Concatenate scene videos into final video using MoviePy with crossfade transitions."""
        try:
            try:
                from moviepy import VideoFileClip, concatenate_videoclips, CompositeVideoClip
            except ImportError:
                from moviepy.video.io.VideoFileClip import VideoFileClip
                from moviepy.video.compositing.concatenate import concatenate_videoclips
                from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
            
            clips = [VideoFileClip(str(p)) for p in scene_paths]
            
            # Add crossfade transitions between clips
            transition_duration = 0.8  # seconds
            
            # Method: Use concatenate_videoclips with crossfade
            # For smooth transitions, we overlap clips slightly
            final = concatenate_videoclips(
                clips,
                method="compose",
                padding=-transition_duration,  # Overlap for crossfade
            )

            # Narration must follow the VIDEO timeline: each scene starts at
            # S_i = sum(durations) - overlap*i (crossfade), so subtract the
            # overlap per scene. Otherwise subtitles drift 0.8s per scene.
            try:
                try:
                    from moviepy import CompositeAudioClip
                except ImportError:
                    from moviepy.audio.AudioClip import CompositeAudioClip
                offset = 0.0
                audio_parts = []
                for clip in clips:
                    if clip.audio is not None:
                        audio_parts.append(clip.audio.with_start(offset))
                        offset += clip.audio.duration - transition_duration
                if audio_parts:
                    final = final.with_audio(CompositeAudioClip(audio_parts))
            except Exception as exc:
                print(f"   [WARN] Sequential audio join failed, keeping mixed audio: {exc}")
            
            # Alternative: Manual crossfade for more control
            # final = self._apply_crossfade_transitions(clips, transition_duration)
            
            codec, preset, extra_args = self._resolve_video_codec()
            final.write_videofile(
                str(output_path),
                fps=self.video_settings.fps,
                codec=codec,
                bitrate=self.video_settings.bitrate,
                audio_codec=self.video_settings.audio_codec,
                audio_bitrate=self.video_settings.audio_bitrate,
                preset=preset,
                ffmpeg_params=extra_args if extra_args else None,
                threads=self._encoder_threads(),
                logger=None,
            )
            for c in clips:
                c.close()
            final.close()
            return True
        except Exception as e:
            print(f"   [ERROR] Concatenation failed: {e}")
            # Fallback to FFmpeg concat
            return self._concatenate_videos_ffmpeg(scene_paths, output_path)

    def _apply_crossfade_transitions(self, clips: list, transition_duration: float):
        """Apply manual crossfade transitions between clips."""
        try:
            from moviepy import CompositeVideoClip
        except ImportError:
            from moviepy.video.CompositeVideoClip import CompositeVideoClip
        
        if len(clips) <= 1:
            return clips[0]
        
        # Start with first clip
        result = clips[0]
        current_time = result.duration
        
        for i in range(1, len(clips)):
            next_clip = clips[i]
            
            # Create crossfade by overlapping
            # Fade out current, fade in next
            fade_out = result.with_effects([lambda c: c.fadeout(transition_duration)])
            fade_in = next_clip.with_effects([lambda c: c.fadein(transition_duration)])
            
            # Position next clip to start before current ends
            fade_in = fade_in.with_start(current_time - transition_duration)
            
            # Composite them
            result = CompositeVideoClip([result, fade_in])
            current_time = result.duration - transition_duration
        
        return result

    def _concatenate_videos_varied_xfade(self, scene_paths: list[Path], output_path: Path) -> bool:
        """Concatenate with varied FFmpeg xfade + acrossfade (opt-in, 3-scene tested)."""
        if len(scene_paths) <= 1:
            import shutil
            shutil.copy2(scene_paths[0], output_path)
            return True
        try:
            transitions = ["fade", "dissolve", "wipeleft", "slideright"]
            tcfg = self.settings.transitions
            dur = max(0.1, min(1.0, tcfg.duration))
            cmd = ["ffmpeg", "-y"]
            for p in scene_paths:
                cmd += ["-i", str(p)]
            fc = []
            # Video chain
            vlabel = "[0:v]"
            offset = 0.0
            for i in range(1, len(scene_paths)):
                # durations via ffprobe
                d_prev = self._get_video_duration(scene_paths[i - 1]) or 5.0
                offset = offset + d_prev - dur if i == 1 else offset + (self._get_video_duration(scene_paths[i - 1]) or 5.0) - dur
                tr = transitions[(i - 1) % len(transitions)]
                out = f"[vx{i}]" if i < len(scene_paths) - 1 else "[v]"
                fc.append(f"{vlabel}[{i}:v]xfade=transition={tr}:duration={dur}:offset={offset:.3f}{out}")
                vlabel = out
            # Audio chain acrossfade
            alabel = "[0:a]"
            for i in range(1, len(scene_paths)):
                out = f"[ax{i}]" if i < len(scene_paths) - 1 else "[a]"
                fc.append(f"{alabel}[{i}:a]acrossfade=d={dur}:curve=tri{out}")
                alabel = out
            codec, preset, extra_args = self._resolve_video_codec()
            cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
                    "-c:v", codec, "-preset", preset, *extra_args,
                    "-pix_fmt", "yuv420p", "-c:a", self.video_settings.audio_codec, str(output_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"   [WARN] Varied xfade failed, falling back to crossfade")
                return self._concatenate_videos(scene_paths, output_path)
            return True
        except Exception as exc:
            print(f"   [WARN] Varied xfade error ({exc}); falling back to crossfade")
            return self._concatenate_videos(scene_paths, output_path)

    def _concatenate_videos_ffmpeg(self, scene_paths: list[Path], output_path: Path) -> bool:
        """Concatenate using FFmpeg (fallback)."""
        concat_file = self._temp_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for p in scene_paths:
                f.write(f"file '{p.absolute()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            print(f"FFmpeg concat failed: {e}")
            return False

    def compose(self, story, output_path: Optional[Path] = None, draft_mode: bool = False, 
                progress_callback=None) -> CompositionResult:
        """Compose final video from story."""

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.settings.paths.output_dir / f"stoic_{story.theme}_{timestamp}.mp4"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.video_settings.engine != "moviepy" and self.visual_engine_settings.enabled:
            print(
                "   [INFO] visual-engine particles/2.5D are moviepy-only; the "
                f"{self.video_settings.engine} engine renders the reel without them"
            )

        scene_videos = []
        total_duration = 0.0

        for i, scene in enumerate(story.scenes):
            if progress_callback:
                progress_callback(i, len(story.scenes), f"Composing scene {scene.scene_number}")

            # Validate inputs
            if not scene.image_path or not Path(scene.image_path).exists():
                print(f"Warning: Missing image for scene {scene.scene_number}, using placeholder")
                placeholder = self._temp_dir / f"placeholder_{scene.scene_number}.jpg"
                self._create_placeholder_image(placeholder)
                scene.image_path = placeholder

            if not scene.audio_path or not Path(scene.audio_path).exists():
                raise ValueError(f"Missing audio for scene {scene.scene_number}")

            # Determine scene duration
            duration = scene.estimated_duration
            if duration <= 0:
                duration = self._get_audio_duration(Path(scene.audio_path))
            if duration <= 0:
                duration = 10.0

            if draft_mode or self.settings.draft_mode:
                duration = min(duration, 2.0)

            # Breathing pause between scenes (all but last): keeps subtitles
            # aligned with narration boundaries instead of mid-sentence cuts.
            if i < len(story.scenes) - 1:
                padded = self._pad_scene_audio(Path(scene.audio_path))
                if padded != Path(scene.audio_path):
                    scene.audio_path = padded
                    duration += 0.6
                    scene.estimated_duration = duration

            scene_output = self._temp_dir / f"scene_{scene.scene_number:03d}.mp4"

            if self.video_settings.engine == "ffmpeg":
                success = self._compose_scene_ffmpeg(
                    Path(scene.image_path),
                    Path(scene.audio_path),
                    scene.text_overlay,
                    scene_output,
                    duration,
                    scene.scene_number,
                    scene.voiceover_text,
                )
            else:
                success = self._apply_kenburns_and_text(
                    Path(scene.image_path),
                    Path(scene.audio_path),
                    scene.text_overlay,
                    scene_output,
                    duration,
                    scene.scene_number,
                    scene.voiceover_text,
                )

            if not success:
                raise RuntimeError(f"Failed to compose scene {scene.scene_number}")

            scene_videos.append(scene_output)
            total_duration += duration

        if progress_callback:
            progress_callback(len(story.scenes), len(story.scenes), "Concatenating scenes")

        # Concatenate all scenes (varied xfade opt-in, crossfade fallback).
        # The ffmpeg engine uses stream-copy concat (hard cuts, sequential
        # audio, minimal RAM) unless varied transitions are requested.
        transitions = getattr(self.settings, "transitions", None)
        if transitions is not None and transitions.varied:
            success = self._concatenate_videos_varied_xfade(scene_videos, output_path)
        elif self.video_settings.engine == "ffmpeg":
            success = self._concatenate_videos_ffmpeg(scene_videos, output_path)
        else:
            success = self._concatenate_videos(scene_videos, output_path)
        if not success:
            raise RuntimeError("Failed to concatenate scenes")

        self._mix_music_bed(output_path)

        # Cleanup temp scene files
        for sv in scene_videos:
            sv.unlink(missing_ok=True)
        for still in self._temp_dir.glob("still_*.png"):
            still.unlink(missing_ok=True)

        # Get final video info
        final_duration = self._get_video_duration(output_path)
        file_size = output_path.stat().st_size

        return CompositionResult(
            output_path=output_path,
            duration=final_duration or total_duration,
            scenes_count=len(story.scenes),
            resolution=(self.video_settings.width, self.video_settings.height),
            fps=self.video_settings.fps,
            file_size=file_size,
        )


# Global instance
_composer: Optional[VideoComposer] = None


def get_video_composer(settings=None) -> VideoComposer:
    """Get or create the global video composer instance."""
    global _composer
    if _composer is None or settings is not None:
        _composer = VideoComposer(settings)
    return _composer
