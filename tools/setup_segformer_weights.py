"""Download and convert ImageNet-1k SegFormer (MIT) backbone weights.

Source: HuggingFace ``nvidia/mit-b{0,1,2}`` checkpoints (backbone-only,
``SegformerModel``). The conversion remaps the HuggingFace naming onto the
official NVIDIA/SegFormer ``util/mit.py`` naming used by
``adasam.models.baselines.MixVisionTransformer``:

- HF separate query/key/value projections are concatenated into the official
  single ``kv`` projection;
- HF ``encoder.layer_norm`` and per-block norms map onto ``norm{i}``/``norm1/2``;
- the classifier/head is discarded.

Usage:
    python tools/setup_segformer_weights.py --variant b0   # -> weights/mit_b0.pth
    python tools/setup_segformer_weights.py --all          # b0, b1, b2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ENDPOINT = "https://huggingface.co"
HF_REPOS = {
    "b0": "nvidia/mit-b0",
    "b1": "nvidia/mit-b1",
    "b2": "nvidia/mit-b2",
}


def resolve_endpoint(explicit: str | None) -> str:
    """Endpoint order: --hf-endpoint > HF_ENDPOINT env > huggingface.co."""
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("HF_ENDPOINT", "").strip()
    return env.rstrip("/") if env else DEFAULT_ENDPOINT


def convert_state_dict(hf_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HuggingFace SegformerModel keys onto official mit.py naming.

    HF naming (``nvidia/mit-b{0,1,2}``):
      segformer.encoder.patch_embeddings.{i}.proj / layer_norm
      segformer.encoder.block.{s}.{j}.layer_norm_1 / layer_norm_2
      segformer.encoder.block.{s}.{j}.attention.self.{query,key,value,sr,layer_norm}
      segformer.encoder.block.{s}.{j}.attention.output.dense
      segformer.encoder.block.{s}.{j}.mlp.{dense1,dwconv.dwconv,dense2}
      segformer.encoder.layer_norm.{i}
    Official mit.py naming:
      patch_embed{i+1}.proj / norm
      block{s+1}.{j}.norm1 / norm2
      block{s+1}.{j}.attn.{q, kv, sr, norm, proj}
      block{s+1}.{j}.mlp.{fc1, dwconv.dwconv, fc2}
      norm{i+1}
    """
    converted: dict[str, torch.Tensor] = {}
    kv_parts: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def put(key: str, tensor: torch.Tensor) -> None:
        converted[key] = tensor

    for key, tensor in hf_state.items():
        if key.startswith("classifier"):
            continue
        if key.startswith("segformer."):
            key = key[len("segformer."):]
        # patch embeddings
        m = re.match(r"^encoder\.patch_embeddings\.(\d)\.(proj|layer_norm)\.(weight|bias)$", key)
        if m:
            stage, kind, suffix = m.groups()
            name = "proj" if kind == "proj" else "norm"
            put(f"patch_embed{int(stage) + 1}.{name}.{suffix}", tensor)
            continue
        # block norms
        m = re.match(r"^encoder\.block\.(\d)\.(\d+)\.layer_norm_(\d)\.(weight|bias)$", key)
        if m:
            stage, block, which, suffix = m.groups()
            put(f"block{int(stage) + 1}.{block}.norm{which}.{suffix}", tensor)
            continue
        # attention projections
        m = re.match(
            r"^encoder\.block\.(\d)\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)$", key
        )
        if m:
            stage, block, proj, suffix = m.groups()
            kv_parts.setdefault((stage, block, suffix), {})[proj] = tensor
            continue
        m = re.match(
            r"^encoder\.block\.(\d)\.(\d+)\.attention\.self\.(sr|layer_norm)\.(weight|bias)$", key
        )
        if m:
            stage, block, kind, suffix = m.groups()
            name = "sr" if kind == "sr" else "norm"
            put(f"block{int(stage) + 1}.{block}.attn.{name}.{suffix}", tensor)
            continue
        m = re.match(r"^encoder\.block\.(\d)\.(\d+)\.attention\.output\.dense\.(weight|bias)$", key)
        if m:
            stage, block, suffix = m.groups()
            put(f"block{int(stage) + 1}.{block}.attn.proj.{suffix}", tensor)
            continue
        # mlp
        m = re.match(r"^encoder\.block\.(\d)\.(\d+)\.mlp\.(dense1|dense2|dwconv\.dwconv)\.(weight|bias)$", key)
        if m:
            stage, block, kind, suffix = m.groups()
            name = {"dense1": "mlp.fc1", "dense2": "mlp.fc2", "dwconv.dwconv": "mlp.dwconv.dwconv"}[kind]
            put(f"block{int(stage) + 1}.{block}.{name}.{suffix}", tensor)
            continue
        # stage-final norms
        m = re.match(r"^encoder\.layer_norm\.(\d)\.(weight|bias)$", key)
        if m:
            stage, suffix = m.groups()
            put(f"norm{int(stage) + 1}.{suffix}", tensor)
            continue
        raise RuntimeError(f"unhandled HF key: {key}")

    # concatenate HF key/value projections into the official single kv projection
    for (stage, block, suffix), parts in kv_parts.items():
        missing = {"query", "key", "value"} - set(parts)
        if missing:
            raise RuntimeError(f"incomplete qkv for block{stage}.{block}: missing {missing}")
        if "query" in parts:
            put(f"block{int(stage) + 1}.{block}.attn.q.{suffix}", parts["query"])
        put(f"block{int(stage) + 1}.{block}.attn.kv.{suffix}",
            torch.cat([parts["key"], parts["value"]], dim=0))
    return converted


def convert_one(variant: str, weights_root: Path, force: bool = False,
                endpoint: str | None = None) -> Path:
    out = weights_root / f"mit_{variant}.pth"
    if out.exists() and not force:
        print(f"[skip] {out} already exists (use --force to re-download)")
        return out
    url = f"{resolve_endpoint(endpoint)}/{HF_REPOS[variant]}/resolve/main/pytorch_model.bin"
    tmp = weights_root / f"mit_{variant}.hf.bin"
    if tmp.exists():
        # cached download may be truncated from an interrupted run; validate it
        try:
            torch.load(tmp, map_location="cpu", weights_only=False)
            print(f"[reuse] cached {tmp}")
        except Exception:  # noqa: BLE001 - truncated/invalid cache
            print(f"[reuse] cached {tmp} is invalid, re-downloading")
            tmp.unlink(missing_ok=True)
    if not tmp.exists():
        print(f"[download] {url}")
        urllib.request.urlretrieve(url, tmp)
    hf_state = torch.load(tmp, map_location="cpu", weights_only=False)
    if isinstance(hf_state, dict) and "state_dict" in hf_state:
        hf_state = hf_state["state_dict"]
    print(f"[convert] {len(hf_state)} HF keys -> official mit.py naming")
    converted = convert_state_dict(hf_state)
    torch.save(converted, out)
    tmp.unlink(missing_ok=True)
    print(f"[done] wrote {out} ({len(converted)} keys)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["b0", "b1", "b2"], default="b0")
    parser.add_argument("--all", action="store_true", help="convert b0, b1 and b2")
    parser.add_argument("--weights-root", type=Path, default=ROOT / "weights")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hf-endpoint", default=None,
        help="HuggingFace endpoint override (e.g. https://hf-mirror.com); "
             "falls back to the HF_ENDPOINT env var, then huggingface.co",
    )
    args = parser.parse_args()

    args.weights_root.mkdir(parents=True, exist_ok=True)
    variants = ["b0", "b1", "b2"] if args.all else [args.variant]
    for variant in variants:
        convert_one(variant, args.weights_root, force=args.force, endpoint=args.hf_endpoint)

    # validate against the model definition
    from adasam.models.baselines import MixVisionTransformer  # noqa: PLC0415

    for variant in variants:
        model = MixVisionTransformer(variant=variant)
        state = torch.load(args.weights_root / f"mit_{variant}.pth", map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"validation failed for mit_{variant}: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        print(f"[validate] mit_{variant} loads cleanly into MixVisionTransformer")


if __name__ == "__main__":
    main()
