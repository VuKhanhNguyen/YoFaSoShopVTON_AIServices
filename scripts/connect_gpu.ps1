# Lấy host/port/key từ .env laptop; -Tunnel giữ cửa sổ mở để forward API.
param([switch]$Tunnel)
$serviceRoot = Split-Path -Parent $PSScriptRoot
$gpuConfig = @{}
foreach ($line in Get-Content -LiteralPath (Join-Path $serviceRoot '.env')) {
    if ($line -match '^\s*(GPU_[A-Z0-9_]+)\s*=(.*)$') {
        $gpuConfig[$matches[1]] = $matches[2].Trim().Trim('"').Trim("'")
    }
}
$sshArgs = @('-i', $gpuConfig.GPU_SSH_KEY_PATH, '-p', $gpuConfig.GPU_SSH_PORT)
$destination = '{0}@{1}' -f $gpuConfig.GPU_SSH_USER, $gpuConfig.GPU_SSH_HOST
if ($Tunnel) {
    $forward = '{0}:127.0.0.1:{1}' -f $gpuConfig.GPU_LOCAL_FORWARD_PORT, $gpuConfig.GPU_REMOTE_API_PORT
    & ssh @sshArgs -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L $forward $destination
} else {
    & ssh @sshArgs $destination
}
