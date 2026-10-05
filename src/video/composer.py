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
    ) -> str:
        """Build FFmpeg filter for text overlay."""
        escaped = text.replace("'", r"'\''").replace(":", r"\:").replace("%", r"\%")

        font_size = self.font_settings.size
        font_color = self.font_settings.color.lstrip("#")
        stroke_color = self.font_settings.stroke_color.lstrip("#")
        stroke_width = self.font_settings.stroke_width
        margin_bottom = self.font_settings.margin_bottom

        font_path = self.font_settings.font_path.replace("\\", "/")
        import sys
        if sys.platform == "win32" and ":" in font_path:
            font_path = font_path.replace(":", "\\:")

        if self.font_settings.position == "bottom":
            y_pos = f"h-text_h-{margin_bottom}"
        elif self.font_settings.position == "top":
            y_pos = f"{margin_bottom}"
        else:
            y_pos = "(h-text_h)/2"

        return (
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
            f"enable='between(t,0,{duration})'"
        )

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

    def _apply_kenburns_and_text(self, image_path: Path, audio_path: Path, text_overlay: str, 
                                  output_path: Path, scene_duration: float, scene_number: int) -> bool:
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
                    mask = Image.new("L", (target_w, target_h), 0)
                    mask_draw = ImageDraw.Draw(mask)
                    mask_draw.ellipse((target_w * 0.08, target_h * 0.02, target_w * 0.92, target_h * 1.02), fill=235)
                    subject.putalpha(mask)
                    resized = Image.alpha_composite(background, subject).convert("RGB")

                if self.settings.cine.enabled:
                    from PIL import ImageEnhance
                    cine = self.settings.cine
                    resized = ImageEnhance.Color(resized).enhance(cine.saturation)
                    resized = ImageEnhance.Contrast(resized).enhance(cine.contrast)
                    frame = np.asarray(resized).astype(np.float32)
                    yy, xx = np.ogrid[:target_h, :target_w]
                    distance = np.sqrt(
                        ((xx - target_w / 2) / (target_w / 2)) ** 2
                        + ((yy - target_h / 2) / (target_h / 2)) ** 2
                    )
                    vignette = np.clip(
                        1.0 - cine.vignette * np.maximum(distance - 0.15, 0.0),
                        0.72,
                        1.0,
                    )
                    frame *= vignette[..., None]
                    if cine.grain:
                        rng = np.random.default_rng(int(t * self.video_settings.fps) + scene_number)
                        frame += rng.normal(0, cine.grain / 2, frame.shape[:2])[..., None]
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                    return self._ambient_fx(frame, t, scene_number, particles)
                
                frame = np.array(resized)
                return self._ambient_fx(frame, t, scene_number, particles)
            
            clip = VideoClip(make_frame, duration=scene_duration)
            clip.fps = self.video_settings.fps

            # Add text overlay using MoviePy's TextClip
            try:
                # MoviePy 2.x compatible TextClip - avoid font parameter conflict
                txt_clip = TextClip(
                    text=text_overlay,
                    font_size=self.font_settings.size,
                    color=self.font_settings.color,
                    stroke_color=self.font_settings.stroke_color,
                    stroke_width=self.font_settings.stroke_width,
                    font=self.font_settings.windows_font_path if os.name == 'nt' else self.font_settings.linux_font_path,
                    method='caption',
                    size=(target_w - 200, 260),
                ).with_duration(scene_duration)
                
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
                
                video = CompositeVideoClip([clip, txt_clip])
            except Exception as e:
                print(f"   [WARN] TextClip failed, using video without text: {e}")
                video = clip

            # Add audio
            audio = AudioFileClip(str(audio_path))
            video = video.with_audio(audio)

            # Write output
            video.write_videofile(
                str(output_path),
                fps=self.video_settings.fps,
                codec=self.video_settings.codec,
                bitrate=self.video_settings.bitrate,
                audio_codec=self.video_settings.audio_codec,
                audio_bitrate=self.video_settings.audio_bitrate,
                preset=self.video_settings.preset,
                threads=4,
                logger=None,
            )

            return True

        except Exception as e:
            print(f"   [ERROR] MoviePy composition failed: {e}")
            # Fallback to FFmpeg
            return self._compose_scene_ffmpeg_fallback(image_path, audio_path, text_overlay, output_path, scene_duration, scene_number)

    def _build_scene_filter_complex(
        self,
        width: int,
        height: int,
        duration: float,
        direction: str,
        text_overlay: str,
    ) -> str:
        """Full per-scene filter: half-res zoompan, upscale, text, cine.

        Full-res zoompan is ~4x slower (single-threaded) and times out on
        long scenes; text stays crisp because drawtext runs after upscale.
        """
        if self.kenburns_settings.enabled:
            if direction == "random":
                import random
                direction = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])
            kenburns = self._build_kenburns_filter(
                width // 2, height // 2, duration, direction,
            )
            text_filter = self._build_text_filter(text_overlay, width, height, duration)
            filter_complex = f"[0:v]{kenburns},scale={width}:{height}[kb];[kb]{text_filter}[v]"
        else:
            text_filter = self._build_text_filter(text_overlay, width, height, duration)
            filter_complex = f"[0:v]{text_filter}[v]"

        cine_filter = self._build_cine_filter()
        if cine_filter:
            filter_complex = filter_complex.replace("[v]", f",{cine_filter}[v]", 1)
        return filter_complex

    def _compose_scene_ffmpeg_fallback(self, image_path: Path, audio_path: Path, text_overlay: str,
                                        output_path: Path, scene_duration: float, scene_number: int) -> bool:
        """Compose a single scene using FFmpeg directly (fallback)."""
        # Build filter complex
        filter_complex = self._build_scene_filter_complex(
            self.video_settings.width,
            self.video_settings.height,
            scene_duration,
            self.kenburns_settings.direction,
            text_overlay,
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", self.video_settings.codec,
            "-preset", self.video_settings.preset,
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
                               output_path: Path, scene_duration: float, scene_number: int) -> bool:
        """Compose a single scene using FFmpeg directly."""
        # Build filter complex (shared helper: half-res zoompan + upscale)
        filter_complex = self._build_scene_filter_complex(
            self.video_settings.width,
            self.video_settings.height,
            scene_duration,
            self.kenburns_settings.direction,
            text_overlay,
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", self.video_settings.codec,
            "-preset", self.video_settings.preset,
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
            
            final.write_videofile(
                str(output_path),
                fps=self.video_settings.fps,
                codec=self.video_settings.codec,
                bitrate=self.video_settings.bitrate,
                audio_codec=self.video_settings.audio_codec,
                audio_bitrate=self.video_settings.audio_bitrate,
                preset=self.video_settings.preset,
                threads=4,
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
            cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
                    "-c:v", self.video_settings.codec, "-preset", self.video_settings.preset,
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

            scene_output = self._temp_dir / f"scene_{scene.scene_number:03d}.mp4"

            if self.video_settings.engine == "ffmpeg":
                success = self._compose_scene_ffmpeg(
                    Path(scene.image_path),
                    Path(scene.audio_path),
                    scene.text_overlay,
                    scene_output,
                    duration,
                    scene.scene_number,
                )
            else:
                success = self._apply_kenburns_and_text(
                    Path(scene.image_path),
                    Path(scene.audio_path),
                    scene.text_overlay,
                    scene_output,
                    duration,
                    scene.scene_number,
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
