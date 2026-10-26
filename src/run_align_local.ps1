param(
    [ValidateSet('smoke', 'debug', 'all')][string]$Mode = 'all',
    [string]$Python = 'D:\anaconda3\envs\yolo\python.exe',
    [string]$Workspace = 'E:\HEU_4151\RGB_IR',
    [int]$ImageSize = 640,
    [int]$Batch = 1,
    [string]$Device = '0',
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$code = $PSScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path $Workspace 'align_local_debug_20261022' }
$b1 = Join-Path $Workspace 'fog_ship_experiments_20261019\runs\b1_s42\weights\best.pt'
$dataset = Join-Path $Workspace 'MTDSP_v4_labelreview_20261021\dataset'
$manifestDir = Join-Path $OutputRoot 'manifests'
$runs = Join-Path $OutputRoot 'runs_debug'

foreach ($file in @($Python, $b1, (Join-Path $dataset 'pairs.csv'))) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing required file: $file" }
}

Push-Location $code
try {
    & $Python -c "import torch, ultralytics; assert ultralytics.__version__ == '8.3.163'; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'ultralytics', ultralytics.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'Python environment check failed' }

    & $Python 'run_b2.py' smoke --variant control --weights $b1
    if ($LASTEXITCODE -ne 0) { throw 'Control smoke failed' }
    & $Python 'run_b2.py' smoke --variant align --weights $b1
    if ($LASTEXITCODE -ne 0) { throw 'Align smoke failed' }

    if ($Mode -eq 'smoke') { return }
    if ($Device -eq '0') {
        & $Python -c "import torch; assert torch.cuda.is_available(), 'Local GPU is required for debug'; print(torch.cuda.get_device_name(0))"
        if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed' }
    }
    if (Test-Path -LiteralPath $runs) { throw "Debug output already exists; use a new -OutputRoot: $runs" }
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    & $Python 'prepare_data.py' --dataset $dataset --output $manifestDir
    if ($LASTEXITCODE -ne 0) { throw 'Manifest preparation failed' }
    $data = Join-Path $manifestDir 'fusion_eligible.yaml'

    foreach ($variant in @('control', 'align')) {
        $name = "b2_${variant}_local_s42"
        & $Python 'run_b2.py' debug --variant $variant --weights $b1 --data $data `
            --project $runs --name $name --imgsz $ImageSize --batch $Batch `
            --device $Device --workers 0 --seed 42
        if ($LASTEXITCODE -ne 0) { throw "$variant debug training failed" }
        $run = Join-Path $runs $name
        & $Python 'check_b2_debug.py' --variant $variant --run $run --b1 $b1 --imgsz $ImageSize --batch $Batch
        if ($LASTEXITCODE -ne 0) { throw "$variant debug verification failed" }
    }
    Write-Output "Local smoke and debug completed: $OutputRoot"
}
finally {
    Pop-Location
}
