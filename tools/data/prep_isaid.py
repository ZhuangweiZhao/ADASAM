#!/usr/bin/env python3
"""
iSAID 数据处理 — 解压 DOTA 图像 + 重组目录 + 修正标注
==========================================================

将原始 DOTA + iSAID 数据重组为 ISAIDDataset 兼容格式。

输入 | Input:
    data/DOTA/train/images/part{1,2,3}.zip   → train PNG patches
    data/DOTA/val/images/part1.zip           → val PNG patches
    data/iSAID/TrainData/train/Annotations/  → instance JSON
    data/iSAID/ValidationData/val/Annotations/ → instance JSON
    data/iSAID/TestData/TestData/            → test PNG patches

输出 | Output (data/iSAID_processed/):
    ├── train/
    │   ├── images/              # *.png
    │   └── annotations/
    │       └── instances_train.json  (fixed categories + height/width)
    ├── val/
    │   ├── images/
    │   └── annotations/
    │       └── instances_val.json
    └── test/
        └── images/              # *.png

用法 | Usage::
    python tools/prep_isaid.py
    python tools/prep_isaid.py --skip-extract  # 图像已解压
"""

import sys, argparse, zipfile, shutil, json
from pathlib import Path
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(description="iSAID Data Preparation | iSAID 数据准备")
    p.add_argument("--dota-root", type=str, default="data/DOTA",
                   help="DOTA 原始数据目录 | DOTA raw data directory")
    p.add_argument("--isaid-root", type=str, default="data/iSAID",
                   help="iSAID 原始数据目录 | iSAID raw data directory")
    p.add_argument("--output-root", type=str, default="data/iSAID_processed",
                   help="处理后输出目录 | Processed output directory")
    p.add_argument("--skip-extract", action="store_true",
                   help="跳过 zip 解压 | Skip zip extraction")
    p.add_argument("--force", action="store_true",
                   help="强制重新生成 JSON 标注 (覆盖已有) | Force regenerate JSON annotations")
    p.add_argument("--dry-run", action="store_true",
                   help="只检查不执行 | Only check, don't execute")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Step 1: Extract / copy images
# ═══════════════════════════════════════════════════════════════════

def extract_zips(zip_dir: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """
    解压目录下所有 zip → output_dir/images/ | Extract all zips → output_dir/images/.

    增量检测 | Incremental: 跳过已存在的 PNG | Skip already-existing PNGs.

    :return: {"total": N, "exist": N, "extracted": N}
    :rtype: dict
    """
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(zip_dir.glob("*.zip"))
    if not zips:
        return {"total": 0, "exist": 0, "extracted": 0, "msg": "no zip files"}

    # 统计 zip 内 png 列表 | List all PNGs in zips
    zip_pngs: list[tuple[Path, str]] = []  # (zip_path, internal_path)
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            for m in zf.namelist():
                if m.lower().endswith(".png"):
                    zip_pngs.append((zp, m))

    # 检查已存在的 | Check existing
    exist_count = sum(1 for _, m in zip_pngs if (img_dir / Path(m).name).exists())
    need_count = len(zip_pngs) - exist_count

    if need_count == 0:
        return {"total": len(zip_pngs), "exist": exist_count, "extracted": 0,
                "msg": f"all {exist_count} already extracted"}

    if dry_run:
        return {"total": len(zip_pngs), "exist": exist_count,
                "extracted": 0, "need": need_count,
                "msg": f"would extract {need_count} (dry-run)"}

    # 按 zip 分组解压 | Group by zip for batch extract
    import tempfile
    pbar = tqdm(total=need_count, desc=f"  Extracting {len(zips)} zip(s)", unit="img")
    extracted = 0

    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            pngs = [m for m in zf.namelist() if m.lower().endswith(".png")
                    and not (img_dir / Path(m).name).exists()]
            if not pngs:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp, members=pngs)
                for m in pngs:
                    src = Path(tmp) / m
                    dest = img_dir / Path(m).name
                    if src.exists():
                        shutil.move(str(src), str(dest))
                        extracted += 1
                        pbar.update(1)
    pbar.close()
    return {"total": len(zip_pngs), "exist": exist_count, "extracted": extracted}


def copy_images(src_dir: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """
    批量复制 PNG → output_dir/images/ | Copy PNGs → output_dir/images/.
    增量: 跳过已存在文件 | Incremental: skip existing files.
    """
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    pngs = list(src_dir.rglob("*.png"))
    exist = sum(1 for p in pngs if (img_dir / p.name).exists())
    need = len(pngs) - exist

    if need == 0:
        return {"total": len(pngs), "exist": exist, "copied": 0,
                "msg": f"all {exist} already present"}

    if dry_run:
        return {"total": len(pngs), "exist": exist, "need": need,
                "msg": f"would copy {need} (dry-run)"}

    copied = 0
    pbar = tqdm(total=need, desc="  Copying", unit="img")
    for p in pngs:
        dest = img_dir / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
            copied += 1
            pbar.update(1)
    pbar.close()
    return {"total": len(pngs), "exist": exist, "copied": copied}


# ═══════════════════════════════════════════════════════════════════
# Step 2: Fix JSON annotations
# ═══════════════════════════════════════════════════════════════════

def fix_annotations(json_path: Path, output_path: Path, images_dir: Path,
                    force: bool = False, dry_run: bool = False) -> dict:
    """
    修正 JSON | Fix JSON annotations:
    1. 映射 category IDs → ISAIDDataset.ISAID_CATEGORIES
    2. 补全 image height/width (并行读取 | parallel read)

    :return: {"fixed_cat": N, "fixed_hw": N, "skipped": bool, "msg": str}
    :rtype: dict
    """
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    from adatile.utils.label_mapping import build_mapping, ISAID_CATEGORIES

    # 检查是否已处理 | Check if already processed
    if output_path.exists() and not force:
        return {"fixed_cat": 0, "fixed_hw": 0, "skipped": True,
                "msg": f"already exists ({output_path}), use --force to regenerate"}

    if dry_run:
        return {"fixed_cat": 0, "fixed_hw": 0, "skipped": False,
                "msg": f"would generate {output_path} (dry-run)"}

    with open(json_path) as f:
        data = json.load(f)

    # 根据原始 JSON 的 categories 构建 per-split 映射
    # Build per-split mapping from original JSON categories (train/val have different numbering!)
    orig_categories = data.get("categories", [])
    code_mapping, unmatched = build_mapping(orig_categories)
    if unmatched:
        print(f"  ⚠️  Warning: {len(unmatched)} categories could not be mapped: {unmatched}")

    # 修正 annotation 中的 category_id | Fix category_id in annotations
    fixed_cat = 0
    for ann in data.get("annotations", []):
        old_id = ann.get("category_id", 0)
        if old_id in code_mapping:
            ann["category_id"] = code_mapping[old_id]
            fixed_cat += 1

    # 写入代码定义的 categories | Write code-defined categories
    from adatile.datasets.isaid import ISAID_CATEGORIES
    data["categories"] = ISAID_CATEGORIES

    # 补全 height/width (并行读图头) | Fill height/width (parallel image header read)
    imgs_needing_hw = [(i, img) for i, img in enumerate(data.get("images", []))
                       if "height" not in img or "width" not in img]

    def _get_size(idx_img):
        idx, img_info = idx_img
        img_path = images_dir / img_info["file_name"]
        if img_path.exists():
            with Image.open(img_path) as pil:
                return idx, pil.height, pil.width
        return idx, 0, 0

    fixed_hw = 0
    if imgs_needing_hw:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(tqdm(
                ex.map(_get_size, imgs_needing_hw),
                total=len(imgs_needing_hw),
                desc="  Reading image sizes",
                unit="img",
                leave=False,
            ))
        for idx, h, w in results:
            if h > 0:
                data["images"][idx]["height"] = h
                data["images"][idx]["width"] = w
                fixed_hw += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    return {"fixed_cat": fixed_cat, "fixed_hw": fixed_hw, "skipped": False,
            "msg": f"{fixed_cat} cats + {fixed_hw} h/w → {output_path}"}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    dota = Path(args.dota_root)
    isaid = Path(args.isaid_root)
    out = Path(args.output_root)

    print("=" * 70)
    print("  iSAID Data Preparation")
    print(f"  DOTA:   {dota}")
    print(f"  iSAID:  {isaid}")
    print(f"  Output: {out}")
    if args.dry_run:
        print("  MODE:   DRY-RUN (检查不执行 | check only)")
    if args.force:
        print("  MODE:   FORCE (覆盖已有 JSON | overwrite existing JSON)")
    print("=" * 70)

    # ── 预检查：报告所有状态后再执行 | Pre-check: report all status first ──
    steps = []
    all_ok = True

    # Step 1-2: 图像提取/复制 | Images: extract or copy
    # train/val 从 DOTA zip 提取 | train/val from DOTA zips
    # test 直接从 iSAID 复制 | test copied directly from iSAID
    for label, src_zip_dir, dst_dir, is_copy in [
        ("Train images", dota / "train" / "images", out / "train", False),
        ("Val images",   dota / "val" / "images",   out / "val",   False),
        ("Test images",  isaid / "TestData" / "TestData", out / "test", True),
    ]:
        img_dir = dst_dir / "images"
        if is_copy:
            src_pngs = list(src_zip_dir.rglob("*.png")) if src_zip_dir.exists() else []
            exist_n = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
            total_n = len(src_pngs)
            need_n = total_n - exist_n
            src_type = "copy"
        else:
            zips = sorted(src_zip_dir.glob("*.zip")) if src_zip_dir.exists() else []
            total_n = 0
            exist_n = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
            for zp in zips:
                try:
                    with zipfile.ZipFile(zp) as zf:
                        total_n += sum(1 for m in zf.namelist() if m.lower().endswith(".png"))
                except Exception:
                    pass
            need_n = total_n - exist_n
            src_type = "extract"

        status = "✅" if need_n == 0 and total_n > 0 else ("⬜" if need_n > 0 else "❌")
        if need_n > 0:
            all_ok = False
        steps.append((f"[{status}] {label}", src_type, src_zip_dir, dst_dir, is_copy, need_n, total_n, exist_n))

    # Step 3-4: 标注修正 | Annotations: fix category IDs + height/width
    for label, src_json, dst_json, img_dir in [
        ("Train annotations",
         isaid / "TrainData" / "train" / "Annotations" / "iSAID_train.json",
         out / "train" / "annotations" / "instances_train.json",
         out / "train" / "images"),
        ("Val annotations",
         isaid / "ValidationData" / "val" / "Annotations" / "iSAID_val.json",
         out / "val" / "annotations" / "instances_val.json",
         out / "val" / "images"),
    ]:
        if dst_json.exists() and not args.force:
            status = "✅"
            need_ann = 0
        elif src_json.exists():
            status = "⬜"
            need_ann = 1
            all_ok = False
        else:
            status = "❌"
            need_ann = 0
        steps.append((f"[{status}] {label}", "json", src_json, dst_json, img_dir, need_ann, 1, 1 if dst_json.exists() else 0))

    # ── 打印状态报告 | Print status report ──
    print("\n  Status Report | 状态报告:")
    print(f"  {'─'*60}")
    for step_label, *_ in steps:
        print(f"  {step_label}")
    print(f"  {'─'*60}")

    if all_ok:
        print(f"\n  ✅ All steps complete! Nothing to do.")
        print(f"  Use: ISAIDDataset(root_dir='{out}', split='train')")
        return

    if args.dry_run:
        print(f"\n  Dry-run complete. Run without --dry-run to execute.")
        return

    # ── 执行待处理步骤 | Execute pending steps ──
    total_new = 0
    for step_label, step_type, src, dst, extra, need_n, total_n, exist_n in steps:
        if need_n == 0:
            continue

        print(f"\n  {step_label} ({exist_n}/{total_n} exist, {need_n} to {step_type})")

        if step_type in ("extract", "copy"):
            # 图像提取/复制 | Image extraction or copy
            if step_type == "extract" and not args.skip_extract:
                result = extract_zips(src, dst, dry_run=False)
                n = result.get("extracted", 0)
            elif step_type == "copy":
                result = copy_images(src, dst, dry_run=False)
                n = result.get("copied", 0)
            else:
                n = 0
            total_new += n
            if n > 0:
                print(f"    → {step_type}ed {n} images")
        elif step_type == "json":
            # JSON 标注修正 | JSON annotation fix (category mapping + height/width)
            result = fix_annotations(src, dst, extra, force=args.force)
            n = result.get("fixed_cat", 0) + result.get("fixed_hw", 0)
            total_new += n
            print(f"    → {result['msg']}")

    # ── 最终摘要：验证每个 split 的完整性 | Final summary: verify completeness per split ──
    print(f"\n{'=' * 70}")
    print(f"  Final Summary | 最终状态:")
    print(f"  {'─'*50}")
    all_done = True
    for split in ["train", "val", "test"]:
        img_dir = out / split / "images"
        ann_dir = out / split / "annotations"
        n_img = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
        n_ann = len(list(ann_dir.glob("*.json"))) if ann_dir.exists() else 0
        ok = "✅" if n_img > 0 and (split == "test" or n_ann > 0) else "❌"
        if "❌" in ok:
            all_done = False
        print(f"  {ok} {split:6s}: {n_img:5d} images, {n_ann} annotations")
    print(f"  {'─'*50}")

    if all_done:
        print(f"\n  ✅ All ready! Use: ISAIDDataset(root_dir='{out}', split='train')")
    else:
        print(f"\n  ⚠️  Some steps incomplete. Re-run to fix.")


if __name__ == "__main__":
    main()
