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
        self._temp_dir = self.settings.paths.temp_dir / "video_composition"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

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

    def _apply_kenburns_and_text(self, image_path: Path, audio_path: Path, text_overlay: str, 
                                  output_path: Path, scene_duration: float, scene_number: int) -> bool:
        """Apply Ken Burns effect and text overlay using MoviePy (v2.x)."""
        try:
            from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip, VideoClip
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            import random
            import os

            # Load image
            img = Image.open(image_path)
            img_w, img_h = img.size
            target_w, target_h = self.video_settings.width, self.video_settings.height

            # Ken Burns effect parameters
            zoom = self.kenburns_settings.zoom_factor
            direction = self.kenburns_settings.direction
            if direction == "random":
                direction = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])

            fps = self.video_settings.fps
            total_frames = int(scene_duration * fps)

            # Pre-compute all frames for Ken Burns effect
            frames = []
            for frame_idx in range(total_frames):
                progress = frame_idx / max(1, total_frames - 1)
                
                # Calculate zoom and position based on direction
                if direction == "zoom_in":
                    curr_zoom = 1 + (zoom - 1) * progress
                    cx, cy = img_w / 2, img_h / 2
                elif direction == "zoom_out":
                    curr_zoom = zoom - (zoom - 1) * progress
                    cx, cy = img_w / 2, img_h / 2
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
                
                # Crop and resize
                cropped = img.crop((left, top, left + crop_w, top + crop_h))
                resized = cropped.resize((target_w, target_h), Image.LANCZOS)
                
                frames.append(np.array(resized))

            # Create video clip from frames using VideoClip with make_frame function
            def make_frame(t):
                frame_idx = min(int(t * self.video_settings.fps), total_frames - 1)
                return frames[frame_idx]
            
            clip = VideoClip(make_frame, duration=scene_duration)
            clip.fps = self.video_settings.fps

            # Add text overlay using MoviePy's TextClip
            try:
                txt_clip = TextClip(
                    text_overlay,
                    font_size=self.font_settings.size,
                    color=self.font_settings.color,
                    stroke_color=self.font_settings.stroke_color,
                    stroke_width=self.font_settings.stroke_width,
                    font=self.font_settings.windows_font_path if os.name == 'nt' else self.font_settings.linux_font_path,
                    method='caption',
                    size=(target_w - 100, None),
                ).with_duration(scene_duration)
                
                # Position at bottom
                if self.font_settings.position == "bottom":
                    txt_clip = txt_clip.with_position(('center', target_h - self.font_settings.margin_bottom))
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
                fps=fps,
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

    def _compose_scene_ffmpeg_fallback(self, image_path: Path, audio_path: Path, text_overlay: str, 
                                        output_path: Path, scene_duration: float, scene_number: int) -> bool:
        """Compose a single scene using FFmpeg directly (fallback)."""
        # Build filter complex
        kenburns = ""
        if self.kenburns_settings.enabled:
            direction = self.kenburns_settings.direction
            if direction == "random":
                import random
                directions = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
                direction = random.choice(directions)
            kenburns = self._build_kenburns_filter(
                self.video_settings.width,
                self.video_settings.height,
                scene_duration,
                direction,
            )

        text_filter = self._build_text_filter(
            text_overlay,
            self.video_settings.width,
            self.video_settings.height,
            scene_duration,
        )

        # Combine filters
        if kenburns:
            filter_complex = f"[0:v]{kenburns},{text_filter}[v]"
        else:
            filter_complex = f"[0:v]{text_filter}[v]"

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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
        # Build filter complex
        kenburns = ""
        if self.kenburns_settings.enabled:
            direction = self.kenburns_settings.direction
            if direction == "random":
                import random
                directions = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
                direction = random.choice(directions)
            kenburns = self._build_kenburns_filter(
                self.video_settings.width,
                self.video_settings.height,
                scene_duration,
                direction,
            )

        text_filter = self._build_text_filter(
            text_overlay,
            self.video_settings.width,
            self.video_settings.height,
            scene_duration,
        )

        # Combine filters
        if kenburns:
            filter_complex = f"[0:v]{kenburns},{text_filter}[v]"
        else:
            filter_complex = f"[0:v]{text_filter}[v]"

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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
            from moviepy import VideoFileClip, concatenate_videoclips, CompositeVideoClip
            
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
        from moviepy import CompositeVideoClip
        
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

        # Concatenate all scenes
        success = self._concatenate_videos(scene_videos, output_path)
        if not success:
            raise RuntimeError("Failed to concatenate scenes")

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