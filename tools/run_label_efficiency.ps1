param(
    [int[]]$K = @(5, 10, 20, 50, 100),
    [int]$Epochs = 100,
    [int]$Seed = 42,
    [switch]$SkipUnet,
    [switch]$SkipOurs,
    [switch]$IncludeOursFull
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$stage1 = Join-Path $repo "runs/neuseg_stage1_k5_seed42/best_adapter.pt"
$manifestRoot = Join-Path $repo "runs/label_efficiency/manifests"
$outputRoot = Join-Path $repo "runs/label_efficiency"

function Run-Command([string]$label, [string]$command) {
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    Write-Host $command
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
}

foreach ($k in $K) {
    $manifest = Join-Path $manifestRoot "manifest_k${k}_seed${Seed}.json"
    $out = Join-Path $outputRoot "unet_k${k}_seed${Seed}"
    Run-Command "manifest K=$k" "python tools/neuseg/make_kshot_manifest.py --data-root data/NEU_Seg --k $k --seed $Seed --output `"$manifest`""
    if (-not $SkipUnet) {
        Run-Command "Low-data U-Net K=$k" "python tools/U-Net/train_low_data_neu_seg.py --stage1-ckpt `"$stage1`" --manifest `"$manifest`" --config configs/neu_seg_unet.yaml --epochs $Epochs --device cuda --seed $Seed --val-fraction 0.2 --output-dir `"$out`""
    }
    if (-not $SkipOurs) {
        $oursOut = Join-Path $outputRoot "ours_k${k}_seed${Seed}"
        Run-Command "AdaSAM Stage2 K=$k" "python tools/neuseg/train_stage2.py --stage1-ckpt `"$stage1`" --manifest `"$manifest`" --config configs/neu_seg_stage2.yaml --epochs $Epochs --support-shot $k --seed $Seed --device cuda --output-dir `"$oursOut`""
    }
}

if (-not $SkipUnet) {
    $fullOut = Join-Path $outputRoot "unet_full_seed${Seed}"
    Run-Command "Full-supervision U-Net" "python tools/U-Net/train_neu_seg.py --config configs/neu_seg_unet.yaml --epochs $Epochs --device cuda --seed $Seed --output-dir `"$fullOut`""
}

if ($IncludeOursFull) {
    Write-Warning "Ours full-data support-conditioned run is intentionally not launched by default; define and review its support budget before enabling it."
}
