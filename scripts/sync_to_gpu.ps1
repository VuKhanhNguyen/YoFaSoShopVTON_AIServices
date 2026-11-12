# ==============================================================================
# YoFaSoShop_AIService - Script dong bo code tu Laptop len GPU ao qua SSH/SCP
# Chi dong bo code Python (api/core, api/services, api/vision, workers, scripts)
# TUYET DOI KHONG dong bo api/models/, weights/, .venv/, __pycache__/
# ==============================================================================
param(
    [switch]$Watch
)

$serviceRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $serviceRoot '.env'

if (-not (Test-Path $envPath)) {
    Write-Error "Khong tim thay file .env tai $envPath. Vui long kiem tra."
    exit 1
}

$gpuConfig = @{}
foreach ($line in Get-Content -LiteralPath $envPath) {
    if ($line -match '^\s*(GPU_[A-Z0-9_]+)\s*=(.*)$') {
        $gpuConfig[$matches[1]] = $matches[2].Trim().Trim('"').Trim("'")
    }
}

$sshHost = $gpuConfig.GPU_SSH_HOST
$sshPort = $gpuConfig.GPU_SSH_PORT
$sshUser = $gpuConfig.GPU_SSH_USER
$sshKey = $gpuConfig.GPU_SSH_KEY_PATH

if ([string]::IsNullOrWhiteSpace($sshHost) -or $sshHost -eq "CHANGE_ME_HOST") {
    Write-Error "Vui long dien GPU_SSH_HOST, GPU_SSH_PORT, GPU_SSH_KEY_PATH trong file .env truoc khi dong bo!"
    exit 1
}

$remoteDest = "$sshUser@$sshHost"
$remoteAppDir = "/workspace/YoFaSoShop_AIService"
$sshKeyArgs = if (-not [string]::IsNullOrWhiteSpace($sshKey)) { @('-i', $sshKey) } else { @() }

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "DONG BO CODE YO FASO SHOP AI SERVICE -> GPU CLOUD" -ForegroundColor Cyan
Write-Host "Dich den: $remoteDest`:$remoteAppDir (Port: $sshPort)" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

function Sync-CodeFiles {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Dang dong bo code Python (loai tru models/weights)..." -ForegroundColor Green

    $mkdirCmd = "mkdir -p $remoteAppDir/api/core $remoteAppDir/api/services $remoteAppDir/api/vision $remoteAppDir/api/models $remoteAppDir/workers $remoteAppDir/scripts $remoteAppDir/weights"
    & ssh @sshKeyArgs -p $sshPort $remoteDest $mkdirCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Khong the ket noi SSH toi $remoteDest`:$sshPort"
        return
    }

    # Dong bo tung nhom file code nhe, tranh api/models (~20GB) va __pycache__
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/api/*.py" "$remoteDest`:$remoteAppDir/api/"
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/api/core/*.py" "$remoteDest`:$remoteAppDir/api/core/"
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/api/services/*.py" "$remoteDest`:$remoteAppDir/api/services/"
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/api/vision/*.py" "$remoteDest`:$remoteAppDir/api/vision/"
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/workers/*.py" "$remoteDest`:$remoteAppDir/workers/"
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/scripts/*.py" "$serviceRoot/scripts/*.sh" "$remoteDest`:$remoteAppDir/scripts/"
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/requirements-api.txt" "$serviceRoot/requirements-gpu.txt" "$serviceRoot/.env.example" "$serviceRoot/pyproject.toml" "$remoteDest`:$remoteAppDir/"

    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Hoan tat dong bo code! (~1-2s)" -ForegroundColor Green
}

Sync-CodeFiles

if ($Watch) {
    Write-Host "Dang theo doi thay doi file (.py)... (Nhan Ctrl+C de dung)" -ForegroundColor Yellow
    $lastTime = Get-Date
    while ($true) {
        Start-Sleep -Seconds 3
        $changed = Get-ChildItem -Path "$serviceRoot/api/core", "$serviceRoot/api/services", "$serviceRoot/api/vision", "$serviceRoot/workers", "$serviceRoot/scripts" -Filter "*.py" |
                   Where-Object { $_.LastWriteTime -gt $lastTime }
        $rootChanged = Get-ChildItem -Path "$serviceRoot/api/*.py" | Where-Object { $_.LastWriteTime -gt $lastTime }
        $allChanged = @($changed) + @($rootChanged)
        if ($allChanged.Count -gt 0) {
            $lastTime = Get-Date
            Write-Host "Phat hien thay doi tai: $($allChanged.Name -join ', ')" -ForegroundColor Yellow
            Sync-CodeFiles
        }
    }
}
