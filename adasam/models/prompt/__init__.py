"""Prompt modules for label-efficient industrial segmentation."""

from adasam.models.prompt.defect_prompt_generator import DefectPromptGenerator
from adasam.models.prompt.defect_prompt_generator_v2 import DefectAwarePromptGeneratorV2

__all__ = ["DefectPromptGenerator", "DefectAwarePromptGeneratorV2"]
