# NEU_Seg Full-Supervision U-Net Baseline

This baseline imports the unmodified U-Net architecture from
`thirdparty/Pytorch-UNet/unet`.

Protocol:

- Split the 3,630 official `images/training` samples into 90% train and 10%
  validation by dominant foreground class, with a fixed seed.
- Evaluate every epoch on validation only. The 840 official `images/test` samples
  remain unseen until final evaluation of the validation-selected checkpoint.
- Count every test image exactly once.
- Report background IoU, Inclusion IoU, Patch IoU, Scratch IoU, and foreground
  macro mIoU.
- Save `best_model.pt` by validation foreground macro mIoU and `last_model.pt`
  every epoch. Save the final held-out result as `test_evaluation.json`.

This is a full-supervision reference model. Its result is an upper-bound reference
for the K-shot support-conditioned study, not a directly label-budget-matched
comparison.

Run from the repository root:

```powershell
conda activate pytorch
python tools/U-Net/train_neu_seg.py --device cuda
```
