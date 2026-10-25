param(
    [string]$LocalDir = "",
    [string]$Revision = "main",
    [switch]$DryRun,
    [switch]$ForceLowDisk
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Arguments = @("$ScriptRoot/download_ootdiffusion.py", "--revision", $Revision)

if ($LocalDir) { $Arguments += @("--local-dir", $LocalDir) }
if ($DryRun) { $Arguments += "--dry-run" }
if ($ForceLowDisk) { $Arguments += "--force-low-disk" }

python @Arguments
exit $LASTEXITCODE
