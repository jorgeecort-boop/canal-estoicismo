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
        """Create a short text overlay from voiceover text."""
        max_chars = self.narrative_settings.text_overlay_max_chars

        # Extract key phrase - first sentence or key clause
        sentences = [s.strip() for s in voiceover.split(".") if s.strip()]
        if not sentences:
            return "Reflexiona..."

        # Take first sentence, truncate if needed
        overlay = sentences[0]
        if len(overlay) > max_chars:
            # Try to cut at a natural break
            words = overlay.split()
            result = []
            for word in words:
                if len(" ".join(result + [word])) <= max_chars:
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
    ) -> str:
        """Build an image generation prompt for the scene."""

        # Base style
        base_style = self.settings.image.style_prompt

        # Section-specific visual emphasis
        section_visuals = {
            "hook": "dramatic opening, mysterious atmosphere, cinematic lighting",
            "core_teaching": "ancient wisdom visualization, philosophical symbols, marble",
            "practical_application": "person applying wisdom, daily life stoic, serene focus",
            "closing_reflection": "transcendent, peaceful resolution, golden light, eternal",
        }

        # Keywords from template
        keywords = " ".join(template.image_keywords[:3])

        # Combine
        prompt_parts = [
            base_style,
            section_visuals.get(section, ""),
            keywords,
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