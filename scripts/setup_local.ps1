# Tạo venv riêng và .env có secret ngẫu nhiên; không ghi đè .env đã có.
param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
& $Python -m venv .venv-api
if ($LASTEXITCODE -ne 0) { throw 'Không tạo được venv; chỉ định -Python đường dẫn Python 3.10+.' }
& ./.venv-api/Scripts/python.exe -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Cài dependency thất bại.' }
if (-not (Test-Path -LiteralPath .env)) {
    $secret = & ./.venv-api/Scripts/python.exe -c 'import secrets; print(secrets.token_urlsafe(32))'
    $template = Get-Content -LiteralPath .env.example -Raw
    $template.Replace('CHANGE_ME_RANDOM_SECRET_AT_LEAST_32_CHARACTERS', $secret.Trim()) | Set-Content -LiteralPath .env -Encoding utf8
}
Write-Output 'Sẵn sàng. Chạy: ./.venv-api/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000'
