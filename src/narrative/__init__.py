"""Narrative package for story generation."""
from .story_generator import StoryGenerator, Scene, Story
from .templates import STOIC_TEMPLATES, get_template

__all__ = ["StoryGenerator", "Scene", "Story", "STOIC_TEMPLATES", "get_template"]