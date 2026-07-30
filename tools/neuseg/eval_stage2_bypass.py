from pathlib import Path
import json
import torch
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stage2 import Stage2Trainer

ckpt_path = Path(r'E:\A_postgraduate_stude\AdaSAM\runs\neuseg_stage2_k5_seed42\best_model.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
if ckpt.get('stage') != 'neuseg_stage2':
    raise SystemExit('not a neuseg_stage2 checkpoint')
cfg = ckpt['config']
# enable bypass decoder via ablation flag
cfg.setdefault('ablation', {})['bypass_decoder'] = True
# set small val samples to speed up
cfg.setdefault('train', {})['val_samples'] = 200
# set eval threshold
cfg.setdefault('eval', {})['foreground_threshold'] = 0.3
# build trainer
trainer_args = type('A', (), {'stage1_ckpt': ckpt.get('stage1_checkpoint'), 'steps':None,'epochs':None,'episodes':None,'support_shot':None,'seed':None,'device':None,'data_root':None,'output_dir':None,'val_samples':None})
trainer = Stage2Trainer(cfg, trainer_args)
missing, unexpected = trainer.model.load_state_dict(ckpt['model'], strict=False)
print('missing', missing)
print('unexpected', unexpected)
metrics = trainer.validate()
out = Path(r'E:\A_postgraduate_stude\AdaSAM\runs\eval_bypass.json')
out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics, indent=2))
print('saved to', out)
