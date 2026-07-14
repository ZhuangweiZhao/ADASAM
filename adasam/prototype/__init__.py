"""adasam.prototype — 小样本原型构建/记忆/匹配 | Few-shot prototype build, memory & match."""

from adasam.prototype.builder import PrototypeBuilder
from adasam.prototype.memory import PrototypeMemory
from adasam.prototype.matcher import Matcher, PromptPoints, similarity_map

__all__ = [
    "PrototypeBuilder",
    "PrototypeMemory",
    "Matcher",
    "PromptPoints",
    "similarity_map",
]
