"""Train the fair U-Net baseline under the label-ratio benchmark protocol."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from adasam.datasets.industrial import LabelRatioSubset, NEUSegSemanticDataset  # noqa: E402
from adasam.datasets.augmentation import build_augmentation  # noqa: E402
from adasam.losses import LabelEfficientSegmentationLoss  # noqa: E402
from adasam.models import LabelEfficientUNet  # noqa: E402
from adasam.utils import set_seed  # noqa: E402
from tools.train_segmentation import evaluate  # noqa: E402

def args():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=["neu_seg"],default="neu_seg")
    p.add_argument("--label_ratio",type=int,choices=[1,5,10,25,100],required=True)
    p.add_argument("--data-root",default="data/NEU_Seg"); p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--base-channels",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="cuda")
    p.add_argument("--val-fraction",type=float,default=.2); p.add_argument("--output-dir",default="runs/unet_label_ratio")
    p.add_argument("--augmentation",choices=["none","basic","defect"],default="none")
    return p.parse_args()

def main():
    a=args(); set_seed(a.seed); device=torch.device(a.device if torch.cuda.is_available() else "cpu")
    root=Path(a.data_root); root=root if root.is_absolute() else ROOT/root
    base=NEUSegSemanticDataset(root,"train"); pool=LabelRatioSubset(base,a.label_ratio,a.seed)
    train_base=NEUSegSemanticDataset(root,"train",transforms=build_augmentation(a.augmentation))
    validation_base=NEUSegSemanticDataset(root,"train")
    n=max(1,round(len(pool)*a.val_fraction)); val=Subset(validation_base,pool.indices[:n]); train=Subset(train_base,pool.indices[n:])
    test=NEUSegSemanticDataset(root,"test"); kw={"batch_size":a.batch_size,"shuffle":False,"num_workers":0}
    train_loader=DataLoader(train,shuffle=True,**{k:v for k,v in kw.items() if k!="shuffle"}); val_loader=DataLoader(val,**kw); test_loader=DataLoader(test,**kw)
    model=LabelEfficientUNet(base.NUM_CLASSES,a.base_channels).to(device); criterion=LabelEfficientSegmentationLoss(); opt=AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    counts=model.parameter_counts(); out=Path(a.output_dir); out=out if out.is_absolute() else ROOT/out; out=out/f"neu_seg_ratio{a.label_ratio}_seed{a.seed}"; out.mkdir(parents=True,exist_ok=True)
    history=[]; best=-1; best_path=out/"best_model.pt"
    print(f"parameters total={counts['total']:,} trainable={counts['trainable']:,} frozen=0 ratio=100.00%")
    for epoch in range(1,a.epochs+1):
        model.train(); start=time.perf_counter(); losses=[]
        for batch in tqdm(train_loader,desc=f"epoch {epoch}/{a.epochs}"):
            image=batch["image"].to(device); target=batch["mask"].to(device); opt.zero_grad(set_to_none=True)
            loss=criterion(model(image),target); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        val_metrics=evaluate(model,val_loader,device,base.NUM_CLASSES); record={"epoch":epoch,"mean_loss":sum(losses)/len(losses),"first_loss":losses[0],"last_loss":losses[-1],"seconds":time.perf_counter()-start,"validation":val_metrics}; history.append(record)
        print(json.dumps(record))
        if val_metrics["mIoU_fg"]>best: best=val_metrics["mIoU_fg"]; torch.save({"model":model.state_dict(),"epoch":epoch},best_path)
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)["model"]); test_metrics=evaluate(model,test_loader,device,base.NUM_CLASSES)
    metrics={"parameters":counts,"label_pool_samples":len(pool),"train_samples":len(train),"validation_samples":len(val),"history":history,"best_epoch":torch.load(best_path,map_location="cpu",weights_only=False)["epoch"],"test":test_metrics,"augmentation":a.augmentation}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); torch.save({"model":model.state_dict(),"metrics":metrics},out/"last_model.pt"); print(f"test={json.dumps(test_metrics)}\nsaved={out}")
if __name__=="__main__": main()
