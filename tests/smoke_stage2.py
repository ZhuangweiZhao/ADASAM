"""Smoke test: verify Trainer initialization with Stage 2 architecture."""
import sys
sys.path.insert(0, 'thirdparty/MobileSAM')
from pathlib import Path

# Quick check: can we import all components without error?
print("1. Testing imports...")
from adasam.support_encoder import SupportEncoder, SupportEncoderConfig
print("   SupportEncoder: OK")

from adasam.prompt import DensePromptGenerator, DensePromptGeneratorConfig, DPGOutput
print("   DensePromptGenerator + DPGOutput: OK")

from adasam.decoder import QueryMaskDecoder, QueryMaskDecoderConfig
print("   QueryMaskDecoder: OK")

from adasam.model import AdaSAMModel, AdaSAMModelConfig
print("   AdaSAMModel: OK")

# Check config round-trip
print("\n2. Testing config round-trip...")
cfg_dict = {
    'prompt_generator': {'num_queries': 64, 'num_layers': 3},
    'decoder': {'train_mask_decoder': True},
    'support_encoder': {
        'n_support_tokens': 16, 'n_memory_tokens': 64,
        'n_encoder_layers': 2, 'n_heads': 8, 'ffn_dim': 1024,
    },
}
model_cfg = AdaSAMModelConfig.from_dict(cfg_dict)
print(f"   dpg.num_queries={model_cfg.dpg.num_queries}")
print(f"   support_encoder.n_support_tokens={model_cfg.support_encoder.n_support_tokens}")
print(f"   support_encoder.n_encoder_layers={model_cfg.support_encoder.n_encoder_layers}")
print(f"   support_encoder.is_stage2={model_cfg.support_encoder.is_stage2}")

# Verify weights exist
weights = Path('weights/mobile_sam.pt')
if not weights.exists():
    print(f"\n3. SKIP: {weights} not found")
    exit(0)

print("\n3. Testing model construction...")
import torch
from adasam.backbone import build_mobile_sam

sam = build_mobile_sam(str(weights), 'vit_t', 'cpu')
model = AdaSAMModel(sam, model_cfg)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Model built: {total_params/1e6:.2f}M params")

# Verify forward works
print("\n4. Testing forward shapes...")
K, C, H, W = 5, 256, 64, 64
emb = torch.randn(1, C, H, W)
sf = torch.randn(K, C, H, W)
sm = torch.rand(K, H, W) > 0.3

with torch.no_grad():
    dpg_out, low_res, iou_pred = model.forward_train(emb, sf, sm.float())

assert dpg_out.instance_queries.shape == (64, 256)
assert low_res.shape == (64, 1, 256, 256)
assert iou_pred.shape == (64, 1)
assert dpg_out.dense_prompt is not None
assert dpg_out.dense_prompt.shape == (1, 256, 1, 1)
assert len(dpg_out.aux) == 3
print("   All shapes OK!")

# Verify predict
print("\n5. Testing predict...")
masks, scores = model.predict(emb, sf, sm.float(), (896, 896), (896, 896))
print(f"   {masks.shape[0]} predicted instances")

# Verify gradient flow
print("\n6. Testing gradient flow...")
dpg_out, low_res, iou_pred = model.forward_train(emb, sf, sm.float())
loss = low_res.sum() + dpg_out.objectness_logits.sum()
loss.backward()

grad_count = 0
no_grad_modules = []
for name, p in model.named_parameters():
    if p.requires_grad:
        if p.grad is None:
            no_grad_modules.append(name)
        else:
            grad_count += 1

if no_grad_modules:
    print(f"   WARNING: {len(no_grad_modules)} trainable params without grad:")
    for n in no_grad_modules[:5]:
        print(f"     - {n}")
else:
    print(f"   All {grad_count} trainable parameters received gradients!")

# Verify dense_prompt_gen gets gradient (dense_alpha removed in v2 fix)
assert model.dpg.dense_prompt_gen[-1].weight.grad is not None
assert model.dpg.dense_prompt_gen[-1].bias.grad is not None
print(f"   dense_prompt_gen.2.weight.grad_sum = {model.dpg.dense_prompt_gen[-1].weight.grad.abs().sum():.2f}")
print(f"   dense_prompt_gen.2.bias.grad_sum = {model.dpg.dense_prompt_gen[-1].bias.grad.abs().sum():.2f}")

# Verify support_encoder gradients
se_has_grad = sum(1 for p in model.support_encoder.parameters() if p.grad is not None)
print(f"   support_encoder params with grad: {se_has_grad}")

print("\n=== ALL STAGE 2 TESTS PASSED ===")
