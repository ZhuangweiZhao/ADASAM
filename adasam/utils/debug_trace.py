"""
数据流调试追踪器 | Data-flow Debug Tracer.
============================================

轻量级张量追踪器, 在模型前向传播的每个关键节点记录张量的形状/数值范围/
空间统计, 帮助定位信息链断裂位置。同时输出到终端和文件。
Lightweight tensor tracer that records shape/value-range/spatial-stats
at each key forward node. Outputs to both console and file.

用法 | Usage::

    from adasam.utils.debug_trace import tracer

    tracer.enable(level=2, log_dir="runs/xxx/")  # 自动创建 trace.jsonl

    tracer.section("Backbone")
    tracer.tensor("image_embedding", emb)

    tracer.summary()
    tracer.close()  # 训练结束时关闭文件

配置驱动 | Config-driven::

    # yaml
    debug:
      enabled: true
      level: 2         # 0=off, 1=shapes, 2=stats, 3=stats+gradients
      log_every: 10    # 每 N 个 episode 打印一次
      log_dir: null    # 自动使用 output_dir; 也可手动指定

输出文件格式 | Output format:
    runs/xxx/trace.jsonl  — 每行一个 JSON 事件 (step, section, tensor name, stats)
    runs/xxx/trace.log    — 人类可读的纯文本 (与终端输出一致)
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

import torch
import torch.nn as nn


class _Tracer:
    """全局单例调试追踪器 | Global singleton debug tracer.

    同时输出到 stdout 和 trace.log / trace.jsonl 文件。
    Writes to both stdout and trace.log / trace.jsonl files.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._level = 0          # 0=off, 1=shapes, 2=stats, 3=stats+grads
        self._step = 0           # global step counter
        self._log_every = 1      # log every N steps
        self._section_name = ""
        self._section_count = 0
        self._stats: dict[str, list[float]] = defaultdict(list)  # running stats for summary
        self._indent = 0
        # File output
        self._log_file: TextIO | None = None   # trace.log (human-readable)
        self._jsonl_file: TextIO | None = None  # trace.jsonl (machine-readable)
        self._log_dir: Path | None = None

    # ── Public API ──

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def level(self) -> int:
        return self._level

    def enable(
        self, level: int = 2, log_every: int = 1, log_dir: str | Path | None = None
    ) -> None:
        """启用调试追踪 | Enable debug tracing.

        :param level: 0=off, 1=shape only, 2=+value stats, 3=+gradients.
        :param log_every: 每 N 步打印一次 | print every N steps.
        :param log_dir: 日志输出目录 (自动创建 trace.log + trace.jsonl).
            None 时仅输出到终端。| Output directory; console-only if None.
        """
        self._enabled = True
        self._level = level
        self._log_every = max(1, log_every)
        self._open_files(log_dir)
        self._emit(f"[trace] enabled level={level} log_every={log_every}"
                   f"{' file=' + str(self._log_dir) if self._log_dir else ''}")

    def disable(self) -> None:
        self._enabled = False
        self._close_files()

    def close(self) -> None:
        """关闭文件句柄 | Close file handles (call at end of training)."""
        self._close_files()

    def step(self) -> None:
        """推进步数计数器 | Advance the step counter."""
        self._step += 1

    @property
    def should_log(self) -> bool:
        """当前步是否应打印 | Whether to print at current step."""
        return self._enabled and self._step % self._log_every == 0

    def section(self, title: str) -> None:
        """开始一个命名节 | Begin a named section."""
        if not self.should_log:
            return
        self._section_name = title
        self._section_count += 1
        line = "─" * 50
        self._emit("")
        self._emit(line)
        self._emit(f"  [{self._section_count}] {title}  (step={self._step})")
        self._emit(line)
        self._indent = 2
        # JSONL event
        self._json_event("section", {"title": title, "section_idx": self._section_count})

    def tensor(
        self,
        name: str,
        t: torch.Tensor | None,
        spatial: bool = False,
        detail: bool = False,
    ) -> dict[str, Any] | None:
        """记录一个张量 | Log a tensor.

        :param name: 张量名 | tensor name.
        :param t: 张量; None 时打印 "None".
        :param spatial: 是否做空间诊断 (计算空间位置相关性).
        :param detail: 是否打印详细直方图分位数.
        :return: dict of stats, or None if skipped.
        """
        if not self.should_log:
            return None
        if t is None:
            self._emit(f"{' ' * self._indent}{name}: None")
            self._json_event("tensor", {"name": name, "value": None})
            return None

        shape = tuple(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        device = str(t.device)
        record: dict[str, Any] = {"name": name, "shape": list(shape), "dtype": dtype, "device": device}

        if self._level >= 2:
            t_f = t.float()
            v_min = float(t_f.min())
            v_max = float(t_f.max())
            v_mean = float(t_f.mean())
            v_std = float(t_f.std())
            v_norm = float(t_f.norm(p="fro"))

            self._emit(
                f"{' ' * self._indent}{name}:  "
                f"shape={shape}  {dtype}  {device}"
            )
            self._emit(
                f"{' ' * self._indent}  min={v_min:.5f}  max={v_max:.5f}  "
                f"mean={v_mean:.5f}  std={v_std:.5f}  |F|={v_norm:.2f}"
            )
            record.update({"min": v_min, "max": v_max, "mean": v_mean, "std": v_std, "norm": v_norm})

            if spatial and len(shape) == 4 and shape[2] > 1 and shape[3] > 1:
                sp = self._spatial_diag(name, t_f)
                record.update(sp)

            if detail and t_f.numel() > 0:
                self._quantiles(name, t_f)

            self._stats[f"{name}.std"].append(v_std)
            self._stats[f"{name}.norm"].append(v_norm)

        elif self._level >= 1:
            grad_flag = " GRAD" if t.requires_grad else ""
            self._emit(f"{' ' * self._indent}{name}:  shape={shape}  {dtype}  {device}{grad_flag}")

        self._json_event("tensor", record)
        return record

    def tensor_dict(
        self,
        prefix: str,
        d: dict[str, torch.Tensor],
    ) -> None:
        """记录字典中的每个标量张量 | Log each scalar tensor in a dict."""
        if not self.should_log:
            return
        records = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                val = float(v)
                self._emit(f"{' ' * self._indent}{prefix}.{k}:  {val:.6f}")
                records[k] = val
        self._json_event("tensor_dict", {"prefix": prefix, "values": records})

    def grad(
        self,
        name: str,
        param: nn.Parameter | torch.Tensor,
    ) -> None:
        """记录参数的梯度信息 | Log gradient info for a parameter."""
        if not self.should_log or self._level < 3:
            return
        if not isinstance(param, torch.Tensor):
            return
        if param.grad is None:
            self._emit(f"{' ' * self._indent}∇{name}:  NO GRAD")
            self._json_event("grad", {"name": name, "has_grad": False})
        else:
            g = param.grad.float()
            gmin, gmax = float(g.min()), float(g.max())
            gmean, gstd = float(g.mean()), float(g.std())
            gnorm = float(g.norm())
            self._emit(
                f"{' ' * self._indent}∇{name}:  "
                f"min={gmin:.6f}  max={gmax:.6f}  "
                f"mean={gmean:.6f}  std={gstd:.6f}  "
                f"|grad|={gnorm:.4f}"
            )
            self._json_event("grad", {
                "name": name, "has_grad": True,
                "min": gmin, "max": gmax, "mean": gmean, "std": gstd, "norm": gnorm,
            })

    def grad_summary(self, module: nn.Module, prefix: str = "") -> None:
        """打印模块所有参数的梯度概况 | Print gradient summary for a module."""
        if not self.should_log or self._level < 3:
            return
        label = prefix or type(module).__name__
        self._emit(f"\n{' ' * self._indent}── ∇ 梯度概况: {label} ──")
        total = 0
        with_grad = 0
        max_gn = 0.0
        max_name = ""
        grads_list = []
        for name, p in module.named_parameters():
            total += 1
            if p.grad is not None:
                gn = float(p.grad.norm())
                with_grad += 1
                grads_list.append({"name": name, "grad_norm": gn})
                if gn > max_gn:
                    max_gn = gn
                    max_name = name
        self._emit(f"{' ' * self._indent}  params: {total} total, {with_grad} with grad")
        if max_name:
            self._emit(f"{' ' * self._indent}  max grad norm: {max_name} = {max_gn:.6f}")
        grads_list.sort(key=lambda x: -x["grad_norm"])
        if grads_list:
            self._emit(f"{' ' * self._indent}  top-5:")
            for item in grads_list[:5]:
                self._emit(f"{' ' * self._indent}    {item['name']}: {item['grad_norm']:.6f}")
        self._json_event("grad_summary", {
            "module": label, "total": total, "with_grad": with_grad,
            "max_name": max_name, "max_norm": max_gn, "top5": grads_list[:5],
        })

    def summary(self) -> None:
        """打印汇总统计 | Print summary since tracing started."""
        if not self._stats:
            return
        self._emit("")
        self._emit("=" * 60)
        self._emit(f"  TRACE SUMMARY ({len(self._stats)} metrics, {self._section_count} sections)")
        self._emit("=" * 60)
        summary_data = {}
        for key, vals in sorted(self._stats.items()):
            if len(vals) > 1:
                s_mean = sum(vals) / len(vals)
                s_min, s_max = min(vals), max(vals)
                self._emit(f"  {key}:  mean={s_mean:.6f}  "
                           f"min={s_min:.6f}  max={s_max:.6f}  n={len(vals)}")
                summary_data[key] = {"mean": s_mean, "min": s_min, "max": s_max, "n": len(vals)}
            else:
                self._emit(f"  {key}:  {vals[0]:.6f}")
                summary_data[key] = vals[0]
        self._json_event("summary", summary_data)

    def reset(self) -> None:
        """重置累积统计 | Reset accumulated stats."""
        self._stats.clear()
        self._section_count = 0

    # ── Internal: output ──

    def _emit(self, msg: str) -> None:
        """同时输出到终端和日志文件 | Emit to both console and log file."""
        print(msg)
        if self._log_file is not None:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

    def _json_event(self, etype: str, data: dict | list | None) -> None:
        """写入 JSONL 事件 | Write a JSONL event."""
        if self._jsonl_file is None:
            return
        event = {"step": self._step, "section": self._section_name, "type": etype}
        if isinstance(data, dict):
            event.update(data)
        elif isinstance(data, list):
            event["data"] = data
        self._jsonl_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def _open_files(self, log_dir: str | Path | None) -> None:
        """打开日志文件 | Open log files."""
        self._close_files()
        if log_dir is not None:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = (self._log_dir / "trace.log").open("w", encoding="utf-8")
            self._jsonl_file = (self._log_dir / "trace.jsonl").open("w", encoding="utf-8")

    def _close_files(self) -> None:
        """关闭日志文件 | Close log files."""
        for fh in (self._log_file, self._jsonl_file):
            if fh is not None:
                fh.close()
        self._log_file = None
        self._jsonl_file = None

    # ── Internal: diagnostics ──

    def _spatial_diag(self, name: str, t: torch.Tensor) -> dict[str, float]:
        """空间诊断: 随机位置对余弦相似度 + per-channel 空间变化."""
        if t.ndim == 3:
            t = t.unsqueeze(0)  # [1, C, H, W]
        B, C, H, W = t.shape
        if B == 0 or C == 0:
            return {}

        ch_std = t.reshape(B, C, H * W).std(dim=2)  # [B, C]
        avg_ch_std = float(ch_std.mean())
        self._stats[f"{name}.ch_spatial_std"].append(avg_ch_std)

        n_pairs = min(200, H * W)
        flat = t[0].reshape(C, H * W).T  # [H*W, C]
        idx1 = torch.randint(0, H * W, (n_pairs,))
        idx2 = torch.randint(0, H * W, (n_pairs,))
        same = idx1 == idx2
        idx2[same] = (idx2[same] + 1) % (H * W)

        v1 = torch.nn.functional.normalize(flat[idx1].float(), dim=1)
        v2 = torch.nn.functional.normalize(flat[idx2].float(), dim=1)
        cos_sim = float((v1 * v2).sum(dim=1).mean())
        self._stats[f"{name}.spatial_corr"].append(cos_sim)

        status = "FLAT(广播)" if cos_sim > 0.95 else ("STRUCTURED" if cos_sim < 0.8 else "WEAK")
        self._emit(f"{' ' * self._indent}  spatial: ch_std={avg_ch_std:.5f}  "
                   f"pos_corr={cos_sim:.4f}  [{status}]")
        return {"ch_spatial_std": avg_ch_std, "pos_corr": cos_sim, "spatial_status": status}

    def _quantiles(self, name: str, t: torch.Tensor) -> None:
        """打印分位数 | Print quantiles."""
        flat = t.flatten().float()
        qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        vals = [float(torch.quantile(flat, q)) for q in qs]
        q_str = "  ".join(f"Q{q:.0%}={vals[i]:.4f}" for i, q in enumerate(qs))
        self._emit(f"{' ' * self._indent}  quantiles: {q_str}")


# 全局单例 | Global singleton
tracer = _Tracer()


def configure_from_config(cfg: dict, output_dir: str | Path | None = None) -> None:
    """从 yaml 配置字典配置 tracer | Configure tracer from a yaml config dict.

    :param cfg: 完整配置字典 | full config dict.
    :param output_dir: 训练输出目录 (trace 文件保存位置).
        Training output directory (where trace files are saved).
    """
    dbg = cfg.get("debug", {})
    if isinstance(dbg, bool):
        if dbg:
            tracer.enable(level=2, log_every=1, log_dir=output_dir)
        return
    if not isinstance(dbg, dict):
        return
    enabled = dbg.get("enabled", False)
    if enabled:
        log_dir = dbg.get("log_dir") or output_dir
        tracer.enable(
            level=int(dbg.get("level", 2)),
            log_every=int(dbg.get("log_every", 1)),
            log_dir=str(log_dir) if log_dir else None,
        )
    else:
        tracer.disable()
