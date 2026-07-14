"""
评估协议 V3 项目级审计守卫 | Project-wide Evaluation Protocol V3 Audit Guard.
=============================================================================

扫描全项目, 确保旧协议定义 (oracle zero-shot / union-mask IoU / 自造 AP / 非确定采样)
不出现在**官方 V3 评估面**, 也不被任何新文件引入。AdaSAM 是全新仓库, 无历史包袱,
因此隔离区 (QUARANTINE) 为空 —— 任何禁用模式出现在官方面或仓库任意位置即 FAIL。

Scans the whole project so pre-V3 definitions never appear on the official V3 surface
and are not introduced anywhere. AdaSAM is a clean-slate repo with no legacy, so the
QUARANTINE set is empty — any forbidden pattern on the surface or anywhere → FAIL.

语义与 AdaTile-FastSAM/tests/test_protocol_audit.py 一致, 仅重定向到 adasam.* 路径。
Same semantics as AdaTile-FastSAM's audit, only re-pointed at adasam.* paths.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ── 官方 V3 评估面 (必须绝对干净) | Official V3 surface — must be spotless ──
#    evaluate.py 在 M7 加入; 此处对不存在的文件宽容跳过。
#    evaluate.py lands in M7; non-existent surface files are tolerantly skipped.
OFFICIAL_V3 = {
    "adasam/metrics/instance_match.py",
    "adasam/metrics/coco_eval.py",   # 唯一允许直接调用 pycocotools COCOeval 的文件
    "adasam/evaluator/evaluate.py",
}

# ── 隔离区: 全新仓库无历史脚本 | Quarantine: clean-slate repo, none ──
QUARANTINE: set[str] = set()

# ── 禁用模式 | Forbidden patterns (name → regex, per-pattern file exceptions) ──
FORBIDDEN = [
    # oracle zero-shot: 用 GT 选最优 mask / GT bbox 提示
    ("oracle_zero_shot", re.compile(r"def\s+zero_shot_bbox_iou\b|Select best mask by GT IoU"), set()),
    # 自造 AP (禁止重复实现 COCO AP)
    ("self_rolled_ap", re.compile(r"def\s+compute_ap\b|def\s+compute_instance_ap\b|def\s+voc_ap\b"), set()),
    # 原始 COCOeval (必须走官方封装 COCOInstanceEvaluator); coco_eval.py 为封装本体, 豁免
    ("raw_cocoeval", re.compile(r"COCOeval\("), {"adasam/metrics/coco_eval.py"}),
    # 内置 hash() 参与采样 (非确定性); V3 用 _det_hash 不会命中
    ("builtin_hash_sampling", re.compile(r"\+\s*hash\("), set()),
]


def _iter_py():
    for base in ("adasam", "tools"):
        base_dir = _REPO_ROOT / base
        if not base_dir.exists():
            continue
        for p in base_dir.rglob("*.py"):
            rel = p.relative_to(_REPO_ROOT).as_posix()
            yield rel, p


def test_official_v3_surface_is_clean():
    """官方 V3 评估面不得含任何旧协议模式 | Official V3 surface must contain no legacy pattern."""
    problems = []
    for rel in sorted(OFFICIAL_V3):
        path = _REPO_ROOT / rel
        if not path.exists():
            continue  # 尚未实现的评估面文件宽容跳过 | tolerate not-yet-created surface files
        text = path.read_text(encoding="utf-8")
        for name, rx, exceptions in FORBIDDEN:
            if rel in exceptions:
                continue
            for m in rx.finditer(text):
                line = text[:m.start()].count("\n") + 1
                problems.append(f"{rel}:{line}  [{name}]  '{m.group(0)}'")
    assert not problems, (
        "\n\n  ===== OFFICIAL V3 SURFACE CONTAMINATED =====\n"
        + "".join(f"    - {p}\n" for p in problems)
        + "  The V3 evaluator/metrics must never contain oracle / self-AP / raw-COCOeval / hash().\n")


def test_no_legacy_protocol_outside_quarantine():
    """旧协议模式只允许出现在隔离区 (本仓库为空) | Legacy patterns forbidden everywhere."""
    allowed = OFFICIAL_V3 | QUARANTINE
    leaks = []
    for rel, path in _iter_py():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, rx, exceptions in FORBIDDEN:
            if rel in exceptions:
                continue
            m = rx.search(text)
            if m and rel not in allowed:
                line = text[:m.start()].count("\n") + 1
                leaks.append(f"{rel}:{line}  [{name}]  '{m.group(0)}'")
    assert not leaks, (
        "\n\n  ===== LEGACY-PROTOCOL LEAK =====\n"
        + "".join(f"    - {p}\n" for p in leaks)
        + "  Route all metrics through adasam.metrics + adasam/evaluator/evaluate.py.\n")
