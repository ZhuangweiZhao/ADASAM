"""Smoke test: verify AdaSAM Stage 2 architecture (SPG + GeometricPrior + PromptFusion)."""
import sys
sys.path.insert(0, 'thirdparty/MobileSAM')
from pathlib import Path

# Quick check: can we import all components without error?
print("1. Testing imports...")
from adasam.support_encoder import SupportEncoder, SupportEncoderConfig
print("   SupportEncoder: OK")

from adasam.prompt import SemanticPriorGenerator, SemanticPriorGeneratorConfig, SPGOutput
from adasam.prompt import GeometricPriorModule, PromptFusion
print("   SPG + GeometricPrior + PromptFusion: OK")

from adasam.decoder import SemanticMaskDecoder, SemanticMaskDecoderConfig
print("   SemanticMaskDecoder: OK")

from adasam.model import AdaSAMModel, AdaSAMModelConfig
print("   AdaSAMModel: OK")

# Check config round-trip
print("\n2. Testing config round-trip...")
cfg_dict = {
    'semantic_prior': {'num_probes': 64, 'num_layers': 3},
    'decoder': {'train_mask_decoder': True},
    'support_encoder': {
        'n_support_tokens': 16, 'n_memory_tokens': 64,
        'n_encoder_layers': 2, 'n_heads': 8, 'ffn_dim': 1024,
    },
    'geometric_prior': {'enabled': True},
    'prompt_fusion': {'enabled': True, 'mode': 'concat'},
}
model_cfg = AdaSAMModelConfig.from_dict(cfg_dict)
print(f"   spg.num_probes={model_cfg.spg.num_probes}")
print(f"   use_geometric_prior={model_cfg.use_geometric_prior}")
print(f"   use_prompt_fusion={model_cfg.use_prompt_fusion}")

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
print(f"   geometric_prior: {'ON' if model.geometric_prior else 'OFF'}")
print(f"   prompt_fusion: {'ON' if model.prompt_fusion else 'OFF'}")
print(f"   num_probes: {model.num_probes}")

# Verify forward works
print("\n4. Testing forward shapes (SPG unified output, no dense_prompt/sparse_token)...")
K, C, H, W = 5, 256, 64, 64
emb = torch.randn(1, C, H, W)
sf = torch.randn(K, C, H, W)
sm = torch.rand(K, H, W) > 0.3

with torch.no_grad():
    spg_out, low_res, iou_pred = model.forward_train(emb, sf, sm.float())

# SPG: unified outputs only (no dense_prompt/sparse_token)
assert spg_out.semantic_prior.shape == (1, 256, 64, 64)
assert spg_out.prior_mask.shape == (1, 1, 64, 64)
# Single mask from SAM decoder
assert low_res.shape == (1, 1, 256, 256)
assert iou_pred.shape == (1, 1)
# prior_aux: unified [1, gh, gw] per layer (NOT per-probe [N, gh, gw])
assert len(spg_out.prior_aux) == 3
for a in spg_out.prior_aux:
    assert a["prior_mask"].shape == (1, 64, 64)
print("   All shapes OK!")
print(f"   semantic_prior: {spg_out.semantic_prior.shape}")
print(f"   low_res: {low_res.shape}")

# Verify predict
print("\n5. Testing predict...")
masks, scores = model.predict(emb, sf, sm.float(), (896, 896), (896, 896))
print(f"   masks: {masks.shape}, scores: {scores}")

# Verify gradient flow
print("\n6. Testing gradient flow...")
spg_out, low_res, iou_pred = model.forward_train(emb, sf, sm.float())
# Include all SPG outputs in loss: semantic_prior + prior_mask + prior_aux
loss = low_res.sum() + spg_out.semantic_prior.sum() + spg_out.prior_mask.sum()
for a in spg_out.prior_aux:
    loss = loss + a["prior_mask"].sum()
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

# Verify SPG probe gradients
assert model.spg.probe_feat.weight.grad is not None
print(f"   probe_feat grad = {model.spg.probe_feat.weight.grad.abs().sum():.2f}")

# Verify support_encoder gradients
se_has_grad = sum(1 for p in model.support_encoder.parameters() if p.grad is not None)
print(f"   support_encoder params with grad: {se_has_grad}")

# Verify geometric_prior gradients
if model.geometric_prior is not None:
    gp_grad = sum(1 for p in model.geometric_prior.parameters() if p.grad is not None)
    print(f"   geometric_prior params with grad: {gp_grad}")

# Verify prompt_fusion gradients
if model.prompt_fusion is not None:
    pf_grad = sum(1 for p in model.prompt_fusion.parameters() if p.grad is not None)
    print(f"   prompt_fusion params with grad: {pf_grad}")

# Verify spatial_prompt_proj gradients (now in AdaSAMModel)
sp_has_grad = sum(1 for p in model.spatial_prompt_proj.parameters() if p.grad is not None)
print(f"   spatial_prompt_proj params with grad: {sp_has_grad}")

print("\n=== ALL STAGE 2 TESTS PASSED ===")
