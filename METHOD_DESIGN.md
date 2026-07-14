# METHOD_DESIGN — AdaSAM V2: Dense Prompt Generation

## Status: Phase 0 — Architecture Design (Revision 2, ✅ Approved — Phases 1-4 implemented)

---

## 1. Paper Narrative

**Title direction**: *Automatic Prompt Generation for SAM in Remote Sensing Few-shot Segmentation*

**One-sentence pitch**: Instead of compressing K support similarity maps into a single scalar prototype and then running top-K peak picking, we keep the full **Similarity Tensor [K, H, W]** alive through region proposal, and introduce a **learnable Prompt Generator** that consumes per-region multi-support features to produce rich SAM prompts (point + box + residual prompt token + confidence score).

**Claim**: The bottleneck in few-shot SAM-based segmentation is not the decoder, not the backbone — it is the **information flow break** between support features and prompt generation. Current methods compress K×64×64 = 4096K similarity values into K point coordinates. We are the first to preserve the similarity tensor through region proposal and let a learned module decide which regions are worth prompting.

---

## 2. Architecture Overview

```
                    Support Images (K=5)              Query Image
                         │                                │
                         ▼                                ▼
              MobileSAM Image Encoder          MobileSAM Image Encoder
              (frozen, shared weights)         (frozen, shared weights)
                         │                                │
           ┌─────────────┴─────────────┐                  │
           │                           │                  │
           ▼                           ▼                  ▼
   Global Prototype [256]     Dense Support Features   Query Feature
   (masked avg pool + L2)     [K, 256, 64, 64]        [1, 256, 64, 64]
           │                           │                  │
           │    ┌──────────────────────┘                  │
           │    │                                         │
           ▼    ▼                                         ▼
        ┌─────────────────────────────────────────────────┐
        │         Correlation Builder                      │
        │  Per-support k:                                  │
        │    sim_k[h,w] = cosine(support_k[:,h,w],         │
        │                        query[:,h,w])             │
        │    gate_k = sigmoid(cosine(support_pooled[k],    │
        │                        prototype))               │
        │    sim_k *= gate_k                               │
        │  → Similarity Tensor [K, 64, 64]                 │
        │  (NO fusion here — keep K channels alive)        │
        └──────────────────────┬──────────────────────────┘
                               │  [K, 64, 64]
                               ▼
        ┌─────────────────────────────────────────────────┐
        │         Region Proposal Module                   │
        │                                                  │
        │  Per support k:                                  │
        │    threshold τ_k = μ_k + α·σ_k  (relative!)      │
        │    binary_k = sim_k > τ_k                        │
        │                                                  │
        │  Union across supports:                          │
        │    binary_union = any(binary_k) over k           │
        │                                                  │
        │  connected_components(binary_union, 8-connect)   │
        │  → N regions                                     │
        │                                                  │
        │  Per region:                                     │
        │    - centroid_xy, bbox_xywh (geometry)           │
        │    - pooled query feature [256]                  │
        │    - per-support mean_sim [K] (NOT fused!)       │
        │    - region_score = f(mean, max, area, …)        │
        └──────────────────────┬──────────────────────────┘
                               │  N regions, each with [K] support signals
                               ▼
        ┌─────────────────────────────────────────────────┐
        │         Learnable Prompt Generator               │
        │                                                  │
        │  Input per region:                               │
        │    prototype [256]                                │
        │    region_query_feature [256]                     │
        │    per_support_mean_sim [K]                       │
        │    geometric_feats [~8]  (area, aspect, …)       │
        │                                                  │
        │  Output per region:                              │
        │    point_xy      [2]    (from region, not learned)│
        │    box_xyxy      [4]    (from region, not learned)│
        │    Δprompt_token [256]  (residual, zero-init)    │
        │    region_score  [1]    (learnable confidence)   │
        │                                                  │
        │  prompt_token = prototype + Δprompt_token        │
        │  (~100K params, 2-layer MLP)                     │
        └──────────────────────┬──────────────────────────┘
                               │  N × (point, box, prompt_token, region_score)
                               ▼
        ┌─────────────────────────────────────────────────┐
        │         SAM Prompt Encoder (frozen)              │
        │  prompt_encoder(points, boxes, masks=None)       │
        │  → sparse_embeddings [N, 1+2, 256]              │
        │    (1 point token + 2 box corner tokens)         │
        │  → dense_embeddings [N, 256, 64, 64]            │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼  inject prompt_token
        ┌─────────────────────────────────────────────────┐
        │  sparse = concat([prompt_token[N,1,256],         │
        │                   sparse_emb], dim=1)             │
        │  → [N, 1+3, 256]  =  [N, 4, 256]                │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────────┐
        │         MobileSAM Mask Decoder (trainable)       │
        │  mask_decoder(image_emb, image_pe,               │
        │               sparse, dense, multimask=False)    │
        │  → low_res_masks [N, 1, 256, 256]               │
        │  → iou_pred [N, 1]                               │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────────────┐
        │         Post-processing                          │
        │  upscale → tile resolution → threshold           │
        │  → masks [N, 896, 896] bool                      │
        │  → scores = iou_pred × region_score              │
        │  → Mask IoU NMS (threshold=0.6)                  │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
                    Instance Predictions
```

**Key difference from V1 design**: Fusion does NOT happen before region proposal. The similarity tensor [K, 64, 64] flows directly into region proposal. Per-support statistics are preserved through the region descriptor and into the Prompt Generator. Fusion happens *implicitly* inside the Prompt Generator MLP, which can learn to weight supports differently per region.

---

## 3. Module Contracts

### 3.1 Global Prototype (REUSE existing `PrototypeBuilder`)

No changes. Existing module is correct for its single responsibility.

```
Input:  K × embedding [256, 64, 64] + K × FG mask [896, 896]
Output: prototype [256]  (L2-normalized)
```

### 3.2 Dense Support Features (NEW — thin wrapper)

Purpose: extract and store per-support image embeddings WITHOUT averaging.

```python
# New file: adasam/prototype/support_features.py

def extract_support_features(
    backbone: MobileSAMBackbone,
    support_images: list[torch.Tensor],     # K × [3, 896, 896] float [0,1]
    support_fg_masks: list[torch.Tensor],    # K × [896, 896] float {0,1}
) -> tuple[
    torch.Tensor,                            # [K, 256, 64, 64] support embeddings
    torch.Tensor,                            # [256] global prototype
]:
```

Implementation: call backbone on each support image, collect embeddings, also call PrototypeBuilder to get global prototype. Pure plumbing — no new logic.

### 3.3 Correlation Builder (NEW)

```
Input:
  - support_features: [K, 256, 64, 64]
  - prototype: [256]
  - query_feature: [1, 256, 64, 64]

Output:
  - sim_tensor: [K, 64, 64]  (one similarity map per support, NO fusion)
```

**Algorithm — Simple cosine + prototype gate (v1):**

```python
# For each support k:
for k in range(K):
    # Cosine similarity at every spatial location
    sim_k = cosine_similarity(support_features[k], query_feature[0])  # [64, 64]

    # Prototype gate: how relevant is this support to the class prototype?
    support_pooled_k = support_features[k].mean(dim=(1,2))  # [256]
    gate_k = sigmoid(cosine_similarity(support_pooled_k, prototype))  # scalar

    sim_tensor[k] = sim_k * gate_k
```

Complexity: O(K × 64 × 64) — negligible.

**Why no fusion here**: Each support may highlight different parts of the same object (e.g., Support1→bow, Support2→stern). Fusing before region proposal loses this diversity. Instead, region proposal operates on all K maps, and the Prompt Generator learns to combine per-support signals per region.

### 3.4 Region Proposal Module (NEW — replaces `Matcher.select`)

```
Input:
  - sim_tensor: [K, 64, 64]  (K similarity maps, NOT fused)
  - query_feature_3d: [256, 64, 64]  (for region feature pooling)
  - alpha: float = 1.0  (relative threshold coefficient)
  - min_area: int = 1   (grid cells, ≈ 1 cell → 16×16 px at tile res)

Output:
  - regions: list[Region] where:
      Region = {
          "centroid_xy":       [2] float,    # input-frame (1024²) coords
          "bbox_xyxy":         [4] float,    # input-frame bbox (SAM format: two corners)
          "per_support_mean_sim": [K] float, # mean similarity per support within region
          "max_sim":           float,        # max similarity across all supports in region
          "area_cells":        int,          # region area in grid cells
          "aspect_ratio":      float,        # bbox height/width ratio
          "query_feature":     [256] float,  # pooled query feature at region
          "grid_mask":         [64, 64] bool,# region in grid space
      }
  - N: int  (number of regions, can be 0)
```

**Algorithm:**

```python
# Step 1: Relative threshold per support
#   τ_k = μ_k + α·σ_k
# where μ_k = sim_tensor[k].mean(), σ_k = sim_tensor[k].std()
# α is a hyperparameter (default 1.0)

binary_union = torch.zeros(64, 64, dtype=torch.bool)
for k in range(K):
    mu_k = sim_tensor[k].mean()
    sigma_k = sim_tensor[k].std()
    tau_k = mu_k + alpha * sigma_k
    binary_union |= (sim_tensor[k] > tau_k)

# Alternative relative strategies (ablation candidates):
#   - tau_k = 0.8 * sim_tensor[k].max()
#   - top 20% pixels per support map, then union

# Step 2: Connected components on union
labels = connected_components(binary_union, connectivity=8)  # [64, 64] int

# Step 3: Per-region descriptor
for i in range(1, labels.max() + 1):
    grid_mask = (labels == i)
    if grid_mask.sum() < min_area:
        continue

    # --- Geometry (grid → input frame) ---
    gy, gx = center_of_mass(grid_mask)
    centroid_xy = ((gx + 0.5) * 16, (gy + 0.5) * 16)

    rows, cols = where(grid_mask)
    bbox_xyxy = (min(cols) * 16, min(rows) * 16,
                 (max(cols) + 1) * 16, (max(rows) + 1) * 16)

    w = bbox_xyxy[2] - bbox_xyxy[0]
    h = bbox_xyxy[3] - bbox_xyxy[1]

    # --- Per-support similarity statistics (K values, NOT fused!) ---
    per_support_mean = []
    for k in range(K):
        per_support_mean.append(sim_tensor[k][grid_mask].mean().item())

    # --- Confidence from similarity ---
    max_sim = max(per_support_mean)
    mean_sim = sum(per_support_mean) / K
    area_cells = int(grid_mask.sum().item())
    # Region score: combines mean, max, area (learnable weights in Prompt Generator)
    region_score_raw = 0.5 * mean_sim + 0.5 * max_sim

    # --- Pooled query feature ---
    query_feature = query_feature_3d[:, grid_mask].mean(dim=1)  # [256]

    regions.append({
        "centroid_xy": (float(centroid_xy[0]), float(centroid_xy[1])),
        "bbox_xyxy": (float(bbox_xyxy[0]), float(bbox_xyxy[1]),
                       float(bbox_xyxy[2]), float(bbox_xyxy[3])),
        "per_support_mean_sim": per_support_mean,
        "max_sim": float(max_sim),
        "area_cells": area_cells,
        "aspect_ratio": float(max(w, h) / max(min(w, h), 1)),
        "query_feature": query_feature.tolist(),
        "grid_mask": grid_mask,
        "region_score_raw": float(region_score_raw),
    })
```

**Key differences from current Matcher:**
- Current: `argmax → suppress neighbors → argmax → ...` (top-K peaks, greedy NMS)
- New: `relative threshold per support → union → CC → per-region stats`
- Operates on **similarity tensor [K,64,64]**, not a fused map
- Each region carries **K** support similarity values, not a single scalar

**Fallback when N=0:** If no region passes any threshold, fall back to the global-max peak of the best support map (select support k* with max max(sim_k), take global argmax of sim_k*). This ensures every class gets at least 1 prediction attempt, important for recall.

### 3.5 Learnable Prompt Generator (NEW — core innovation)

```
Input per region:
  - prototype:              [256]   global class prototype
  - region_query_feature:   [256]   pooled query feature at region
  - per_support_mean_sim:   [K]     per-support similarity signal
  - geometric_feats:        [6]     area_cells, aspect_ratio,
                                    max_sim, mean_sim_across_supports,
                                    region_score_raw, n_supports_active

  Total input dim: 256 + 256 + K + 6 = 518 + K
  → projected to 512 via first Linear layer

Output per region:
  - point_xy:        [2]     centroid point (from region, NOT learned)
  - box_xyxy:        [4]     bounding box (from region, NOT learned)
  - prompt_token:    [256]   prototype + residual (zero-init residual)
  - region_score:    [1]     learnable confidence
```

**Design rationale:**
- **Point and box are geometric** — directly from region proposal, no learning needed.
- **Prompt token is residual** — `prompt_token = prototype + Δprompt`. At initialization (Δprompt=0), prompt_token = prototype, which means the model starts from a known-good baseline (prototype-as-prompt). Training learns a per-region *adjustment* to this prototype. This is strictly more stable than zero-init.
- **Region score is learned** — the Prompt Generator learns which regions produce good masks. `raw_score = f(mean, max, area)` is a heuristic; the MLP can learn a better mapping from region features → confidence.

**Architecture:**

```python
class PromptGenerator(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256, K=5):
        super().__init__()

        # Input: [prototype, region_feat, per_support_sim(K), geometric(6)]
        input_dim = embed_dim * 2 + K + 6  # 256+256+5+6 = 523

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual prompt token branch
        self.delta_prompt = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Zero-init → at epoch 0, prompt_token = prototype
        nn.init.zeros_(self.delta_prompt[-1].weight)
        nn.init.zeros_(self.delta_prompt[-1].bias)

        # Region score branch
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, prototype, region_query_feature,
                per_support_mean_sim, geometric_feats,
                centroid_xy, bbox_xyxy):
        """
        prototype:             [N, 256]
        region_query_feature:  [N, 256]
        per_support_mean_sim:  [N, K]
        geometric_feats:       [N, 6]
        centroid_xy:           [N, 2]
        bbox_xyxy:             [N, 4]

        Returns:
            point_xy:     [N, 2]
            box_xyxy:     [N, 4]
            prompt_token: [N, 256]  (= prototype + residual)
            region_score: [N, 1]
        """
        # Fuse all inputs
        x = torch.cat([prototype, region_query_feature,
                       per_support_mean_sim, geometric_feats], dim=-1)
        x = F.gelu(self.input_proj(x))  # [N, hidden_dim]

        # Residual prompt token
        delta = self.delta_prompt(x)     # [N, 256]
        prompt_token = prototype + delta  # residual connection

        # Learnable region confidence
        region_score = self.score_head(x)  # [N, 1]

        return centroid_xy, bbox_xyxy, prompt_token, region_score
```

**~100K parameters** (523×256 + 256×256 + 256×256 + 256×64 + 64×1 ≈ 270K, most in input projection).

### 3.6 SAM Interface (MODIFIED from current `decode()`)

Current `PromptMaskDecoder.decode()`:
```python
# Current: prototype_tokens + point_embeddings → sparse
sparse, dense = self.prompt_encoder(
    points=(coords, labels), boxes=None, masks=None)
proto_tokens = self.proto_adapter(prototype)
sparse = torch.cat([proto_tokens, sparse], dim=1)
```

New `PromptMaskDecoder.decode_v2()`:
```python
# New: prompt_token + box_embeddings + point_embeddings → sparse
sparse, dense = self.prompt_encoder(
    points=(coords, labels), boxes=boxes_xyxy, masks=None)
# sparse shape: [N, 1+2, 256]
#   _embed_points(pad=False, single positive point) → [N, 1, 256]
#   _embed_boxes(boxes) → [N, 2, 256] (two corners)
#   concatenated → [N, 3, 256]

prompt_tokens = prompt_token.unsqueeze(1)  # [N, 1, 256]
sparse = torch.cat([prompt_tokens, sparse], dim=1)  # [N, 4, 256]

low_res, iou_pred = self.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=self.prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse,
    dense_prompt_embeddings=dense,
    multimask_output=False,
)
# → low_res [N, 1, 256, 256], iou_pred [N, 1]
```

### 3.7 Mask IoU NMS (NEW)

```
Input:
  - masks:  [N, 896, 896] bool
  - scores: [N] float  (= iou_pred × region_score)

Output:
  - keep_indices: list[int]

Algorithm:
  Standard mask IoU NMS:
  1. Sort predictions by descending score
  2. For each prediction, if its mask IoU with any higher-score kept prediction > 0.6, discard
  3. Otherwise, keep

Implementation: use existing pairwise_iou() from adasam.metrics.instance_match
```

---

## 4. Training

### 4.1 Teacher Forcing with Sim-Peak Replacement (NOT Gaussian jitter)

Current teacher forcing uses exact GT interior points. This creates a train-test gap: training sees perfect GT points, inference sees similarity-peak points.

**Fix: directly bridge the gap.**

```python
# trainer.py _query_targets() modification:

# GT interior point (distance transform peak)
xy_exact = self._interior_point(inst["mask"])  # tile frame coords

# With 30% probability, replace GT point with similarity-peak point
# This directly trains the decoder to handle imperfect prompts
if random.random() < 0.3:
    # Use current prototype → similarity map → argmax peak
    xy = self._sim_peak_point(prototype, query_embedding, inst["mask"])
else:
    xy = xy_exact  # exact GT point (no jitter needed)
```

**Why this over Gaussian jitter:**
- Gaussian noise is arbitrary — it doesn't model the real test-time error distribution
- Sim-peak replacement trains on the *actual* points the model will see at inference
- This is a standard technique in prompt-based methods (PerSAM uses it)
- 30% is a balance: 70% clean signal for stable learning, 30% realistic noise for robustness

### 4.2 Training Targets

For each GT instance, we provide:
- **Point** (GT interior point or sim-peak, mapped to 1024² input frame)
- **Box** (GT bounding box from annotation, mapped to 1024² input frame)
- **Prompt Token** (from Prompt Generator: `prototype + Δprompt`, learned)
- **Region Score** (from Prompt Generator, supervised by mask IoU)

Loss decomposition:
```
L_total = L_focal(pred, GT_mask) + L_dice(pred, GT_mask)
        + L_iou_head(iou_pred, IoU(pred, GT_mask))
        + β × L_score(region_score, IoU(pred, GT_mask))
```

Where `L_score` is a simple MSE between predicted region_score and actual mask IoU. This teaches the Prompt Generator which regions are trustworthy. β starts at 0.1 and can be tuned.

### 4.3 Multi-Instance Training per Query

Each query tile may have N GT instances (capped at max_instances=32). Each instance gets its own point+box+prompt_token → decoder → mask. Identical to current training structure — only the prompt encoding is richer.

---

## 5. Inference

```
1. Build global prototype from K supports (same as current)

2. Extract dense support features [K, 256, 64, 64] (NEW)

3. Extract query image embedding [1, 256, 64, 64] (same as current)

4. Correlation Builder:
   sim_tensor [K, 64, 64] = correlate(support_feats, prototype, query_feat)
   (NO fusion — keep K channels)

5. Region Proposal on sim_tensor:
   per-support relative threshold → union → connected components → N regions
   Each region carries per_support_mean_sim [K] + geometric features

6. Prompt Generator (batched over N regions):
   point_xy, box_xyxy, prompt_token, region_score = prompt_gen(...)
   prompt_token = prototype + Δprompt (residual)

7. SAM Prompt Encoder (batched):
   sparse, dense = prompt_encoder(points, boxes, masks=None)
   sparse = concat([prompt_token, sparse], dim=1)  # [N, 4, 256]

8. SAM Mask Decoder (batched):
   low_res, iou_pred = mask_decoder(image_emb, image_pe, sparse, dense)

9. Upscale + threshold → masks [N, 896, 896]

10. Scores = iou_pred × region_score  (both learned signals)

11. Mask IoU NMS (threshold=0.6)

12. Return InstanceMasks(N', scores)
```

---

## 6. What Changes vs Current Code

| Component | Current | New | Action |
|-----------|---------|-----|--------|
| `PrototypeBuilder` | Masked avg pool → L2-norm [256] | Same | **Keep** |
| `PrototypeMemory` | Per-class running mean | Same | **Keep** |
| `MobileSAMBackbone` | Frozen, eval-only | Same | **Keep** |
| `Matcher` | Top-K peak + greedy NMS | Unused (replaced by Region Proposal) | **Archive** |
| `PrototypeAdapter` | MLP: proto → token | Unused (replaced by Prompt Generator) | **Archive** |
| `PromptMaskDecoder.decode()` | proto_token + point → mask | prompt_token + point + box → mask | **Extend** (add `decode_v2`) |
| `PromptMaskDecoder.forward()` | sim peak → point → masks | sim_tensor → region proposal → prompt gen → masks | **Rewrite** |
| Correlation Builder | N/A | `cosine_sim + prototype_gate` → [K,64,64] | **New file** |
| Region Proposal | N/A | `relative threshold + CC` on sim_tensor | **New file** |
| Prompt Generator | N/A | `2-layer MLP, residual token, score head` | **New file** |
| Mask IoU NMS | N/A | `pairwise_iou + greedy suppress` | **New file** |
| Trainer `_train_episode()` | GT point only | GT point/sim-peak(70/30) + box + prompt_token | **Modify** |
| Evaluator `_predict_tile()` | Sim peak → decode | Region proposal → prompt gen → decode | **Modify** |
| `adasam/losses/` | focal + dice + iou | Same + optional L_score | **Keep** (extend if needed) |
| `adasam/metrics/` | All V3 frozen | Same | **Keep** |
| `tools/train.py` | Same CLI | Same | **Keep** |
| `tools/evaluate.py` | Same CLI | Same | **Keep** |

---

## 7. File Plan

```
adasam/
├── prototype/
│   ├── builder.py          KEEP (unchanged)
│   ├── memory.py           KEEP (unchanged)
│   ├── matcher.py          ARCHIVE (replaced by region_proposal)
│   ├── support_features.py NEW    (extract_support_features)
│   ├── correlation.py      NEW    (CorrelationBuilder → sim_tensor [K,64,64])
│   └── __init__.py         MODIFY (add new exports)
│
├── decoder/
│   ├── mask_decoder.py     MODIFY (add decode_v2, rewrite forward)
│   ├── prompt_generator.py NEW    (PromptGenerator: residual token + score head)
│   └── __init__.py         MODIFY
│
├── utils/
│   ├── nms.py              NEW    (mask_iou_nms)
│   └── region_proposal.py  NEW    (relative-threshold CC on sim_tensor [K,64,64])
│
├── trainer/
│   └── trainer.py          MODIFY (_train_episode: sim-peak 30% + box + prompt_token)
│
├── evaluator/
│   └── evaluate.py         MODIFY (_predict_tile: use new pipeline)
│
(Everything else unchanged)
```

---

## 8. Ablation Experiment Plan

Goal: prove each component matters, and prove the prototype is necessary.

| ID | Name | Prompt Input | Key Change |
|----|------|-------------|------------|
| E0 | Baseline (current) | Point only | Top-K similarity peaks → points → SAM. Current `forward()`. |
| E1 | + Regions | Point from CC centroids | Replace peak-picking with connected-component regions on sim_tensor. Still point-only. |
| E2 | + Box | Point + Box | Add box prompt from region bbox (xyxy). |
| E3 | Full method | Point + Box + Prompt Token + Region Score | Add Prompt Generator with residual token + learned score. |
| **E3.5** | **No Prototype** | **Region Feature → MLP → Prompt Token** | **Remove prototype from Prompt Generator input. Region feature only. Answers: does prototype contribute?** |
| E4 | + Sim-peak training | Full + 30% sim-peak | Replace GT points with sim-peak 30% of the time during training. |
| E5 | + Mask NMS (final) | Full + sim-peak + NMS | Add mask IoU NMS post-processing. **Final model.** |

Expected narrative:
- **E0→E1**: Regions find more instances than top-K peaks (especially dense/small objects). `sim_tensor` union captures instances that different supports highlight. → AP↑
- **E1→E2**: Box prompts give SAM precise spatial extent (corner coordinates). SAM is trained with box prompts; this restores a capability the current method discards. → mask quality↑, AP↑
- **E2→E3**: Residual prompt token injects per-region class-conditional adjustment; learned region score improves confidence ranking. → AP↑ (core innovation)
- **E3→E3.5**: Removing prototype from Prompt Generator. If AP drops significantly, prototype is essential. If AP stays same, prototype is redundant. → critical for paper narrative
- **E3→E4**: Sim-peak replacement bridges train-test gap; the decoder learns to handle realistic (imperfect) prompt points. → AP↑
- **E4→E5**: Mask IoU NMS removes duplicate predictions that come from overlapping regions (adjacent CCs that produce similar masks). → precision↑, FP↓

### Zero-shot baseline (E_ZS)
- MobileSAM everything-mode (class-agnostic, no support, no prototype, no Prompt Generator)

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| sim_tensor union → one giant blob for dense scenes | Medium | α in τ=μ+ασ controls threshold strictness; sweep α∈[0.5, 2.0]. Can also try per-support CC then union-regions (not union-maps then CC). |
| Relative threshold too aggressive for low-sim classes (e.g., vehicle ~0.4 max sim) | Low | α is per-class tunable. Fallback to top-k% if μ+ασ misses everything. |
| Prompt token provides negligible gain over point+box (E2≈E3) | Low | E1+E2 already show improvement over E0. E3.5 tests whether any learnable token helps. Residual init ensures baseline behavior at epoch 0. |
| Sim-peak replacement at 30% → unstable early training | Low | Start with 10% in epochs 1-10, ramp to 30%. |
| MobileSAM decoder can't use box prompts effectively after fine-tuning | Low | SAM is trained with box prompts; few-shot fine-tuning shouldn't break this capability. |
| Correlation Builder (simple cosine) not discriminative enough → E1 poor | Medium | If E1 fails, escalate to local correlation volume with search radius (Choice B). |

---

## 10. Implementation Order

```
Phase 1 (Core pipeline — get it running end-to-end)
  ├── support_features.py      (thin wrapper, ~20 lines)
  ├── correlation.py           (cosine + prototype gate → sim_tensor, ~30 lines)
  ├── region_proposal.py       (relative threshold + CC on sim_tensor, ~50 lines)
  ├── prompt_generator.py      (2-layer MLP + residual + score head, ~50 lines)
  ├── mask_decoder.py          (add decode_v2 path, ~40 lines modified)
  └── Smoke test: can forward pass through entire pipeline on random data

Phase 2 (Training — verify loss decreases)
  ├── trainer.py               (add sim-peak 30% + box + prompt_token + L_score)
  └── Train E1 (regions only) for 1 epoch to verify pipeline, then full E3

Phase 3 (Full model + eval)
  ├── nms.py                   (mask IoU NMS, ~30 lines)
  ├── mask_decoder.py          (rewrite forward() for inference)
  ├── evaluator.py             (update _predict_tile)
  └── Train + eval E5 (full model), compare vs E0

Phase 4 (Ablations — produce the paper table)
  ├── E0  (baseline, retrain with same config)
  ├── E1  (regions only, point-only prompt)
  ├── E2  (+ box)
  ├── E3  (+ prompt token + region score = full method)
  ├── E3.5 (+ ablation: no prototype in Prompt Generator)
  ├── E4  (+ sim-peak training)
  └── E5  (+ NMS → final model)
```

---

## 11. Design Decisions Resolved

| # | Question | Resolution |
|---|----------|------------|
| 1 | Threshold type | **Relative**: τ_k = μ_k + α·σ_k per support map. α=1.0 default, sweep [0.5, 2.0]. |
| 2 | Min region area | **1 grid cell** (≈ 256 px² at tile res = 16×16 px). Small enough for small_vehicle (~10×10 px). |
| 3 | Multi-support fusion point | **Inside Prompt Generator**, not before region proposal. sim_tensor [K,64,64] preserved through region proposal. |
| 4 | Box format | **xyxy** (two corners). SAM `_embed_boxes` reshapes to [N,2,2]. Region proposal outputs bbox_xyxy directly. |
| 5 | 0-region fallback | Global-max peak of best support map → 1 point-only prompt (no box, prototype as prompt_token). |
| 6 | Prompt token init | **Residual**: `prompt_token = prototype + Δprompt`, Δprompt zero-init. At epoch 0, behaves like prototype-as-token. |
| 7 | Point training noise | **30% sim-peak replacement** (not Gaussian jitter). Directly bridges train-test gap. |
| 8 | Confidence scoring | **Learned region_score** from Prompt Generator, supervised by mask IoU. Final score = iou_pred × region_score. |
| 9 | Ablation for prototype necessity | **E3.5**: train Prompt Generator without prototype input. E3 vs E3.5 answers "does prototype matter?" |
