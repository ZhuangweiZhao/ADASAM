"""adasam.prompt — 密集提示生成 | Dense prompt generation (instance queries)."""

from adasam.prompt.coarse_prior import CoarsePriorModule
from adasam.prompt.dense_prompt_generator import (
    DensePromptGenerator,
    DensePromptGeneratorConfig,
    DPGOutput,
)

__all__ = [
    "CoarsePriorModule",
    "DensePromptGenerator",
    "DensePromptGeneratorConfig",
    "DPGOutput",
]
