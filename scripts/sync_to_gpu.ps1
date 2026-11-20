# ==============================================================================
# YoFaSoShop_AIService - Script dong bo code tu Laptop len GPU ao qua SSH/SCP
# Chi dong bo code Python (api/core, api/services, api/vision, workers, scripts)
# va cac file cu the trong $modelFiles (KHONG dong bo ca api/models/, weights/)
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

# Danh sach file cu the trong api/models/ can dong bo (KHONG dong bo OOTDiffusion)
$modelFiles = @()

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "DONG BO CODE YO FASO SHOP AI SERVICE -> GPU CLOUD" -ForegroundColor Cyan
Write-Host "Dich den: $remoteDest`:$remoteAppDir (Port: $sshPort)" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

function Sync-CodeFiles {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Dang dong bo code Python (loai tru models/weights)..." -ForegroundColor Green

    $remoteDirs = @(
        "$remoteAppDir/api/core",
        "$remoteAppDir/api/services",
        "$remoteAppDir/api/vision",
        "$remoteAppDir/api/models",
        "$remoteAppDir/workers",
        "$remoteAppDir/scripts",
        "$remoteAppDir/weights"
    )
    if ($modelFiles -and ($modelFiles.Count -gt 0)) {
        foreach ($relPath in $modelFiles) {
            $normalized = ($relPath -replace '\\', '/').TrimStart('/')
            $parentDir = (Split-Path -Parent $normalized) -replace '\\', '/'
            if ($parentDir) {
                $remoteDirs += "$remoteAppDir/$parentDir"
            }
        }
    }
    $quotedDirs = ($remoteDirs | Select-Object -Unique) | ForEach-Object { "'$_'" }
    $mkdirCmd = "mkdir -p " + ($quotedDirs -join ' ')
    & ssh @sshKeyArgs -p $sshPort $remoteDest $mkdirCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Khong the ket noi SSH toi $remoteDest`:$sshPort"
        return
    }

    foreach ($dir in 'api', 'api/core', 'api/services', 'api/vision', 'workers', 'scripts') {
        $sources = @("$serviceRoot/$dir/*.py")
        if ($dir -eq 'scripts') { $sources += "$serviceRoot/scripts/*.sh" }
        & scp @sshKeyArgs -P $sshPort @sources "$remoteDest`:$remoteAppDir/$dir/"
        if ($LASTEXITCODE -ne 0) { throw "SCP that bai khi dong bo $dir." }
    }
    & scp @sshKeyArgs -P $sshPort "$serviceRoot/requirements-api.txt" "$serviceRoot/requirements-gpu.txt" "$serviceRoot/.env.example" "$serviceRoot/pyproject.toml" "$remoteDest`:$remoteAppDir/"
    if ($LASTEXITCODE -ne 0) { throw "SCP that bai khi dong bo cau hinh mau." }

    if ($modelFiles -and ($modelFiles.Count -gt 0)) {
        foreach ($relPath in $modelFiles) {
            $normalized = ($relPath -replace '\\', '/').TrimStart('/')
            $localFile = Join-Path $serviceRoot $normalized
            if (-not (Test-Path -LiteralPath $localFile -PathType Leaf)) {
                throw "Khong tim thay file trong `$modelFiles: $localFile"
            }
            & scp @sshKeyArgs -P $sshPort $localFile "$remoteDest`:$remoteAppDir/$normalized"
            if ($LASTEXITCODE -ne 0) { throw "SCP that bai khi dong bo $normalized." }
        }
    }

    Write-Host "Da chep code. Cho job dang chay ket thuc, sau do khoi dong lai python -m workers.runner de nap code moi." -ForegroundColor Yellow
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
        $modelChanged = @(
            foreach ($relPath in $modelFiles) {
                $localFile = Join-Path $serviceRoot (($relPath -replace '\\', '/').TrimStart('/'))
                if (Test-Path -LiteralPath $localFile -PathType Leaf) {
                    Get-Item -LiteralPath $localFile | Where-Object { $_.LastWriteTime -gt $lastTime }
                }
            }
        )
        $allChanged = @($changed) + @($rootChanged) + $modelChanged
        if ($allChanged.Count -gt 0) {
            $lastTime = Get-Date
            Write-Host "Phat hien thay doi tai: $($allChanged.Name -join ', ')" -ForegroundColor Yellow
            Sync-CodeFiles
        }
    }
}
