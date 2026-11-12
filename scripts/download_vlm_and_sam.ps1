param(
    [ValidateSet("default", "tiny-only", "base-only", "all-ultralytics", "meta-official", "skip")]
    [string]$SamVariant = "default",

    [ValidateSet("qwen", "florence", "both", "skip")]
    [string]$VlmModel = "qwen",

    [string]$SamDir = "",
    [string]$VlmDir = "",
    [switch]$DryRun,
    [switch]$ForceLowDisk
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Arguments = @("$ScriptRoot/download_vlm_and_sam.py", "--sam-variant", $SamVariant, "--vlm-model", $VlmModel)

if ($SamDir) { $Arguments += @("--sam-dir", $SamDir) }
if ($VlmDir) { $Arguments += @("--vlm-dir", $VlmDir) }
if ($DryRun) { $Arguments += "--dry-run" }
if ($ForceLowDisk) { $Arguments += "--force-low-disk" }

python @Arguments
exit $LASTEXITCODE

