"""
语义分割协议项目级审计守卫 | Project-wide Semantic Segmentation Protocol Audit.
================================================================================

扫描全项目, 确保废弃的实例分割模式 (COCOeval, Hungarian matching, greedy_match)
不出现在活跃代码中。

Scans the whole project so deprecated instance-segmentation patterns never re-appear
in active code. AdaSAM is now unified as a semantic segmentation framework.

[DEPRECATED] Legacy instance segmentation evaluator (adasam/evaluator/) and trainer
(adasam/trainer/) are marked deprecated and exempt from this audit.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Deprecated files (exempt from audit) ──
DEPRECATED = {
    "adasam/evaluator/evaluate.py",
    "adasam/trainer/trainer.py",
    "tools/visualize.py",                # legacy COCO-based visualization
    "tools/deep_diagnose.py",            # legacy diagnostic tool
    "tools/diag_prompt_chain.py",        # legacy diagnostic tool
    "tools/post_train_check.py",         # legacy diagnostic tool
}

# ── Forbidden patterns (never allowed in active code) ──
FORBIDDEN = [
    # Hungarian matching (instance-specific, removed)
    ("hungarian", re.compile(r"\bHungarianMatcher\b|\bhungarian_match\b|\bhungarian_matcher\b")),
    # Old criterion with objectness
    ("set_criterion", re.compile(r"\bSetCriterion\b|\bCriterionConfig\b")),
    # Direct pycocotools usage
    ("pycocotools", re.compile(r"\bCOCOeval\b|\bpycocotools\b")),
    # Instance matching functions
    ("greedy_match", re.compile(r"\bgreedy_match\b|\binstance_miou\b")),
    # Old DPGOutput field names
    ("old_dpg_fields", re.compile(r"\binstance_queries\b|\bobjectness_logits\b")),
    # Built-in hash() for sampling (non-deterministic)
    ("builtin_hash_sampling", re.compile(r"\+\s*hash\(")),
]


def _iter_py():
    for base in ("adasam", "tools"):
        base_dir = _REPO_ROOT / base
        if not base_dir.exists():
            continue
        for p in base_dir.rglob("*.py"):
            rel = p.relative_to(_REPO_ROOT).as_posix()
            yield rel, p


def test_no_instance_patterns_in_active_code():
    """Active code must not contain deprecated instance-segmentation patterns."""
    leaks = []
    for rel, path in _iter_py():
        if rel in DEPRECATED:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, rx in FORBIDDEN:
            m = rx.search(text)
            if m:
                line = text[:m.start()].count("\n") + 1
                leaks.append(f"{rel}:{line}  [{name}]  '{m.group(0)}'")
    assert not leaks, (
        "\n\n  ===== DEPRECATED INSTANCE-SEGMENTATION PATTERN FOUND =====\n"
        + "".join(f"    - {p}\n" for p in leaks)
        + "  AdaSAM is now a semantic segmentation framework.\n"
        + "  Remove instance-specific code or mark the file as DEPRECATED.\n")
