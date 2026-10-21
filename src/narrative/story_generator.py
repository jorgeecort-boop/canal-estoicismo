"""Story generator for stoic video narratives."""

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal, Optional
from datetime import datetime

from config import get_settings
from .templates import STOIC_TEMPLATES, get_template, list_themes


@dataclass
class Scene:
    """A single scene in the video narrative."""
    scene_number: int
    text_overlay: str
    voiceover_text: str
    image_prompt: str
    estimated_duration: float = 0.0
    audio_path: Optional[str] = None
    image_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        return cls(**data)


@dataclass
class Story:
    """Complete story structure for a video."""
    theme: str
    title: str
    philosopher: str
    scenes: list[Scene]
    total_estimated_duration: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Story":
        scenes = [Scene.from_dict(s) for s in data.get("scenes", [])]
        return cls(
            theme=data["theme"],
            title=data["title"],
            philosopher=data["philosopher"],
            scenes=scenes,
            total_estimated_duration=data.get("total_estimated_duration", 0.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Story":
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Story":
        return cls.from_json(path.read_text(encoding="utf-8"))


class StoryGenerator:
    """Generates stoic video narratives divided into scenes."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.narrative_settings = self.settings.narrative
        self._rng = random.Random()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible generation."""
        self._rng.seed(seed)

    def generate(
        self,
        theme: Optional[str] = None,
        target_duration: Optional[float] = None,
        num_scenes: Optional[int] = None,
    ) -> Story:
        """Generate a complete stoic story."""

        # Select theme
        if theme is None:
            theme = self._rng.choice(list_themes())

        template = get_template(theme)

        # Determine target duration
        if target_duration is None:
            target_duration = self._rng.uniform(
                self.narrative_settings.target_duration_min,
                self.narrative_settings.target_duration_max,
            )

        # Determine number of scenes
        if num_scenes is None:
            num_scenes = self._rng.randint(
                self.narrative_settings.scenes_min,
                self.narrative_settings.scenes_max,
            )

        # Generate scenes
        scenes = self._generate_scenes(template, num_scenes, target_duration)

        # Calculate total duration
        total_duration = sum(s.estimated_duration for s in scenes)

        # Create story
        story = Story(
            theme=theme,
            title=template.title,
            philosopher=template.philosopher,
            scenes=scenes,
            total_estimated_duration=total_duration,
            metadata={
                "target_duration": target_duration,
                "actual_scenes": num_scenes,
                "words_per_minute": self.narrative_settings.words_per_minute,
            },
        )

        return story

    def _generate_scenes(
        self,
        template,
        num_scenes: int,
        target_duration: float,
    ) -> list[Scene]:
        """Generate individual scenes from template."""

        # Narrative arc structure
        arc_sections = [
            ("hook", template.hook, 1),
            ("core_teaching", template.core_teaching, 2),
            ("practical_application", template.practical_application, 2),
            ("closing_reflection", template.closing_reflection, 1),
        ]

        # Distribute scenes across sections
        scenes = []
        scene_num = 1
        duration_per_scene = target_duration * 60 / num_scenes  # seconds per scene

        for section_name, content, min_scenes in arc_sections:
            if scene_num > num_scenes:
                break

            # Determine scenes for this section
            remaining_sections = len([s for s in arc_sections if s[0] != section_name and s[2] > 0])
            remaining_scenes = num_scenes - scene_num + 1
            section_scenes = min(
                max(min_scenes, remaining_scenes // max(1, remaining_sections + 1)),
                remaining_scenes,
            )

            # Split content into scene chunks
            chunks = self._split_content(content, section_scenes, section_name)

            for i, chunk in enumerate(chunks):
                if scene_num > num_scenes:
                    break

                # Generate image prompt
                image_prompt = self._build_image_prompt(
                    template,
                    section_name,
                    chunk,
                    i,
                    section_scenes,
                    global_scene=scene_num,
                )

                # Create text overlay (short version)
                text_overlay = self._create_text_overlay(chunk, section_name)

                # Estimate duration based on word count
                word_count = len(chunk.split())
                estimated_duration = max(
                    word_count / (self.narrative_settings.words_per_minute / 60),
                    duration_per_scene * 0.5,
                )

                scene = Scene(
                    scene_number=scene_num,
                    text_overlay=text_overlay,
                    voiceover_text=chunk,
                    image_prompt=image_prompt,
                    estimated_duration=estimated_duration,
                )
                scenes.append(scene)
                scene_num += 1

        return scenes

    def _split_content(self, content: str, num_chunks: int, section: str) -> list[str]:
        """Split content into narrative chunks."""
        sentences = [s.strip() + "." for s in content.split(".") if s.strip()]

        if num_chunks >= len(sentences):
            # Each sentence gets its own scene, pad if needed
            chunks = sentences[:]
            while len(chunks) < num_chunks:
                chunks.append(sentences[-1] if sentences else "Reflexiona sobre esto.")
            return chunks[:num_chunks]

        # Distribute sentences across chunks
        chunks = []
        sentences_per_chunk = len(sentences) / num_chunks

        for i in range(num_chunks):
            start = int(i * sentences_per_chunk)
            end = int((i + 1) * sentences_per_chunk) if i < num_chunks - 1 else len(sentences)
            chunk_sentences = sentences[start:end]
            if not chunk_sentences and sentences:
                chunk_sentences = [sentences[-1]]
            chunks.append(" ".join(chunk_sentences))

        return chunks

    def _create_text_overlay(self, voiceover: str, section: str) -> str:
        """Create a short text overlay (max 2 lines x 45 chars = 90)."""
        max_total = 90

        # Extract key phrase - first sentence or key clause
        sentences = [s.strip() for s in voiceover.split(".") if s.strip()]
        if not sentences:
            return "Reflexiona..."

        # Take first sentence, truncate to two lines max
        overlay = sentences[0]
        if len(overlay) > max_total:
            # Try to cut at a natural break
            words = overlay.split()
            result = []
            for word in words:
                if len(" ".join(result + [word])) <= max_total:
                    result.append(word)
                else:
                    break
            overlay = " ".join(result) + "..."

        return overlay

    def _build_image_prompt(
        self,
        template,
        section: str,
        chunk: str,
        scene_index: int,
        total_in_section: int,
        global_scene: int | None = None,
    ) -> str:
        """Build an image generation prompt for the scene.

        The SDXL render truncates the prompt to its first 200 characters
        (image_manager appends the 35mm style block afterwards), so every
        part below is chosen to be short: motif, framing, theme palette,
        one motion beat and the section atmosphere, in that order.
        """

        # Section-specific visual emphasis (rotated to avoid repetitive statues)
        section_visuals = {
            "hook": "dramatic opening, mysterious atmosphere",
            "core_teaching": "dark chiaroscuro, volumetric light shafts",
            "practical_application": "ethereal golden light, warm focus",
            "closing_reflection": "ancient fresco texture, transcendent calm",
        }

        # Rotate the template's visual motifs scene by scene. Using the first
        # three keywords for every scene made all memento_mori scenes contain
        # a skull/candle/tomb and all present_moment scenes contain the same
        # drop/stone/garden composition.
        visual_index = (global_scene - 1) if global_scene else scene_index
        keywords = template.image_keywords[visual_index % len(template.image_keywords)]

        # Framing variety cycling by global scene (close-up / wide / detail / human / cosmic / fresco)
        framings = (
            "extreme close-up, shallow depth",
            "wide shot, epic architecture",
            "macro detail, carved relief",
            "solitary philosopher in contemplation",
            "aerial view, vast horizon",
            "fresco texture, mural style",
        )
        framing = framings[(visual_index) % len(framings)]

        # One subtle motion beat per scene, offset so it never changes in
        # lockstep with the framing cycle.
        actions = (
            "candle smoke curling upward",
            "dust drifting in a light shaft",
            "shadow sliding across stone",
            "embers rising from a low flame",
            "cloak moving in a faint breeze",
            "frost forming on worn edges",
        )
        action = actions[(visual_index + 2) % len(actions)]

        # Per-theme palette so each reel keeps its own colour world while the
        # composition, light and motif rotate scene by scene. Living in a dict
        # (not on the template dataclass) keeps templates.py untouched.
        palettes = {
            "memento_mori": "cold bone-white marble, ember orange candlelight, charcoal shadows",
            "control_dichotomy": "steel blue moonlight, torch amber accents, slate stone",
            "inner_fortress": "warm sheltered amber, storm indigo beyond the walls, dark bronze",
            "present_moment": "soft morning mist greys, one warm sunrise gold accent",
            "impermanence": "fading dawn peach, dust-rose decay, pale stone",
            "virtue_ethics": "ivory daylight, terracotta and olive tones",
            "adversity_growth": "stormy indigo sky, a shaft of breaking gold, dark iron",
            "ego_death": "black void, silver mist, cool teal dissolve",
            "cosmic_perspective": "deep space navy, nebula teal and gold dust",
            "amor_fati": "cosmic violet, star silver, warm bronze, midnight blue",
        }
        palette = palettes.get(
            template.theme, "warm amber on marble, deep charcoal shadows"
        )

        # Combine (include theme + global scene to avoid cache collisions).
        # Theme keywords lead so SDXL weights the motif; the shared style
        # block stays out (image_manager appends its own richer 35mm style).
        prompt_parts = [
            keywords,
            framing,
            palette,
            action,
            section_visuals.get(section, ""),
            f"theme={template.theme}",
            f"global_scene={global_scene if global_scene is not None else scene_index + 1}",
            f"section={section}",
            f"scene {scene_index + 1} of {total_in_section}",
        ]

        return ", ".join(filter(None, prompt_parts))

    def generate_multiple(
        self,
        count: int,
        themes: Optional[list[str]] = None,
    ) -> list[Story]:
        """Generate multiple stories."""
        stories = []
        available_themes = themes or list_themes()

        for _ in range(count):
            theme = self._rng.choice(available_themes)
            story = self.generate(theme=theme)
            stories.append(story)

        return stories


def create_story_from_outline(
    theme: str,
    custom_scenes: list[dict],
) -> Story:
    """Create a story from a custom outline (for manual scripting)."""
    template = get_template(theme)
    scenes = []

    for i, scene_data in enumerate(custom_scenes, 1):
        scene = Scene(
            scene_number=i,
            text_overlay=scene_data.get("text_overlay", ""),
            voiceover_text=scene_data.get("voiceover_text", ""),
            image_prompt=scene_data.get("image_prompt", ""),
            estimated_duration=scene_data.get("estimated_duration", 0.0),
        )
        scenes.append(scene)

    total_duration = sum(s.estimated_duration for s in scenes)

    return Story(
        theme=theme,
        title=template.title,
        philosopher=template.philosopher,
        scenes=scenes,
        total_estimated_duration=total_duration,
    )
