"""Run U-Net at the same label ratios and emit the same summary schema."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--ratios",nargs="+",type=int,default=[1,5,10,25,100]); p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--base-channels",type=int,default=32); p.add_argument("--data-root",default="data/NEU_Seg"); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42); p.add_argument("--augmentation",choices=["none","basic","defect"],default="none"); p.add_argument("--output-dir",default="runs/unet_label_ratio"); a=p.parse_args()
    out=Path(a.output_dir); out=out if out.is_absolute() else ROOT/out
    for r in a.ratios: subprocess.run([sys.executable,str(ROOT/"tools/train_unet.py"),"--label_ratio",str(r),"--epochs",str(a.epochs),"--batch-size",str(a.batch_size),"--base-channels",str(a.base_channels),"--data-root",a.data_root,"--augmentation",a.augmentation,"--device",a.device,"--seed",str(a.seed),"--output-dir",str(out)],cwd=ROOT,check=True)
    rows=[]
    for r in a.ratios:
        m=json.loads((out/f"neu_seg_ratio{r}_seed{a.seed}"/"metrics.json").read_text(encoding="utf-8")); t=m["test"]; rows.append({"ratio":r,"labeled_images":m["label_pool_samples"],"mIoU":t["mIoU"],"mIoU_fg":t["mIoU_fg"],"Dice":t["Dice"],"Dice_fg":t["Dice_fg"],"pixel_accuracy":t["pixel_accuracy"],"train_time_seconds":sum(x["seconds"] for x in m["history"]),"FPS":t["FPS"],"best_epoch":m["best_epoch"]})
    summary={"protocol":{"dataset":"NEU_Seg","model":"LabelEfficientUNet","seed":a.seed,"epochs":a.epochs,"validation_fraction":.2,"test_images":840,"augmentation":a.augmentation},"results":rows}; out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
