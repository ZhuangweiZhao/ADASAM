"""adasam.prototype — 小样本原型构建/记忆/匹配 | Few-shot prototype build, memory & match."""

from adasam.prototype.builder import PrototypeBuilder
from adasam.prototype.memory import PrototypeMemory
from adasam.prototype.matcher import Matcher, PromptPoints, similarity_map
from adasam.prototype.support_features import extract_support_features
from adasam.prototype.correlation import CorrelationBuilder, similarity_tensor

__all__ = [
    "PrototypeBuilder",
    "PrototypeMemory",
    "Matcher",
    "PromptPoints",
    "similarity_map",
    "extract_support_features",
    "CorrelationBuilder",
    "similarity_tensor",
]
