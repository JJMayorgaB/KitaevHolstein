# =============================================================================
#  Bateria completa (version PowerShell de run_all.sh, para Windows)
#  Uso, desde la carpeta variational_tests:
#    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 6           # 6 nucleos
#    powershell -ExecutionPolicy Bypass -File .\run_all.ps1 6 --quick   # prueba rapida
#  Memoria: ~0.5 GB por nucleo. Con 8 GB de RAM usa 6 nucleos (8 como maximo).
#  Si se interrumpe, vuelve a lanzarlo igual: cada punto ya calculado se salta.
#  (Archivo solo en ASCII a proposito: PowerShell 5 lee los .ps1 sin BOM como ANSI.)
# =============================================================================
param([int]$Cores = 6,
      [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra = @())

# 'Continue': en PowerShell 5 cualquier linea en stderr (avisos de joblib) abortaria con 'Stop'.
# El fallo real se detecta con $LASTEXITCODE.
$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

# Python escribe en UTF-8 y la consola lo interpreta como UTF-8 (simbolos griegos, signo menos).
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch {}

$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
New-Item -ItemType Directory -Force -Path 'results\logs' | Out-Null

function Run-Step([string]$name, [string[]]$cmd) {
    Write-Host ''
    Write-Host ("============ $name   " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    & $py @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "results\logs\$name.log"
    if ($LASTEXITCODE -ne 0) { throw "$name termino con codigo $LASTEXITCODE" }
}

$t0 = Get-Date
Run-Step 'selftest' @('common.py', '--selftest')
Run-Step 'level1'   (@('level1_local.py',      '--cores', "$Cores") + $Extra)
Run-Step 'level2'   (@('level2_global.py',     '--cores', "$Cores") + $Extra)
Run-Step 'level3'   (@('level3_stationary.py', '--cores', "$Cores") + $Extra)
Run-Step 'level4'   @('level4_analysis.py')
Write-Host ''
Write-Host ("Todo listo en {0:N0} min.   Reporte: results\level4\report.txt" -f ((Get-Date) - $t0).TotalMinutes)
