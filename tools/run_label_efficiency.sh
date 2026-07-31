#!/usr/bin/env bash
set -Eeuo pipefail

# Run from any directory after cloning the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
EPISODES="${EPISODES:-100}"
LOW_BATCH_SIZE="${LOW_BATCH_SIZE:-4}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-16}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/NEU_Seg}"
STAGE1_CKPT="${STAGE1_CKPT:-runs/neuseg_stage1_k5_seed42/best_adapter.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/label_efficiency}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
KS="${KS:-5 10 20 50 100}"

export CUDA_VISIBLE_DEVICES="${GPU}"
# Some container images export an invalid OMP_NUM_THREADS string. OpenMP aborts
# before Python starts in that case, so normalize it to a safe integer.
if [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]] && (( OMP_NUM_THREADS > 0 )); then
  export OMP_NUM_THREADS
else
  export OMP_NUM_THREADS=1
fi

# Auto-detect an extra extraction level, e.g. /root/auto-tmp/NEU_Seg/NEU_Seg.
if [[ ! -d "${DATA_ROOT}/images/training" || ! -d "${DATA_ROOT}/annotations/training" ]]; then
  SEARCH_ROOT="${DATA_SEARCH_ROOT:-/root/autodl-tmp}"
  DETECTED_IMAGE_DIR="$(find "${SEARCH_ROOT}" -type d -path '*/images/training' -print -quit 2>/dev/null || true)"
  if [[ -n "${DETECTED_IMAGE_DIR}" ]]; then
    DETECTED_ROOT="$(dirname "$(dirname "${DETECTED_IMAGE_DIR}")")"
    if [[ -d "${DETECTED_ROOT}/annotations/training" ]]; then
      DATA_ROOT="${DETECTED_ROOT}"
    fi
  fi
fi
if [[ ! -d "${DATA_ROOT}/images/training" || ! -d "${DATA_ROOT}/annotations/training" ]]; then
  echo "ERROR: NEU_Seg root not found. Expected images/training and annotations/training under: ${DATA_ROOT}" >&2
  echo "Set DATA_ROOT=/actual/path/to/NEU_Seg and rerun." >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/manifests"

# Stage1 is commonly stored under OUTPUT_ROOT on cloud machines. Resolve it
# automatically when the repository-relative default is unavailable.
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  CANDIDATE="$(find "${OUTPUT_ROOT}" -type f -name 'best_adapter.pt' -print -quit 2>/dev/null || true)"
  if [[ -n "${CANDIDATE}" ]]; then
    STAGE1_CKPT="${CANDIDATE}"
  else
    echo "ERROR: Stage1 checkpoint not found: ${STAGE1_CKPT}" >&2
    echo "Set STAGE1_CKPT=/path/to/best_adapter.pt and rerun." >&2
    exit 2
  fi
fi

echo "Repository: ${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "STAGE1_CKPT=${STAGE1_CKPT}"
echo "K values: ${KS}"

for K in ${KS}; do
  MANIFEST="${OUTPUT_ROOT}/manifests/manifest_k${K}_seed${SEED}.json"
  echo "=== Creating K=${K} manifest ==="
  "${PYTHON_BIN}" tools/neuseg/make_kshot_manifest.py \
    --data-root "${DATA_ROOT}" \
    --k "${K}" \
    --seed "${SEED}" \
    --output "${MANIFEST}"

  echo "=== Low-data U-Net K=${K} ==="
  "${PYTHON_BIN}" tools/U-Net/train_low_data_neu_seg.py \
    --stage1-ckpt "${STAGE1_CKPT}" \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --config configs/neu_seg_unet.yaml \
    --epochs "${EPOCHS}" \
    --batch-size "${LOW_BATCH_SIZE}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --device cuda \
    --output-dir "${OUTPUT_ROOT}/unet_k${K}_seed${SEED}"

  echo "=== AdaSAM Stage2 K=${K} ==="
  "${PYTHON_BIN}" tools/neuseg/train_stage2.py \
    --stage1-ckpt "${STAGE1_CKPT}" \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --config configs/neu_seg_stage2.yaml \
    --epochs "${EPOCHS}" \
    --episodes "${EPISODES}" \
    --support-shot "${K}" \
    --seed "${SEED}" \
    --device cuda \
    --output-dir "${OUTPUT_ROOT}/ours_k${K}_seed${SEED}"
done

echo "=== Full-supervision U-Net ==="
"${PYTHON_BIN}" tools/U-Net/train_neu_seg.py \
  --config configs/neu_seg_unet.yaml \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${FULL_BATCH_SIZE}" \
  --seed "${SEED}" \
  --device cuda \
  --output-dir "${OUTPUT_ROOT}/unet_full_seed${SEED}"

echo "Label-efficiency experiments completed."
