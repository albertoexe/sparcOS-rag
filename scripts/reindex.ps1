<#
.SYNOPSIS
  Keeps the sparcOS RAG index fresh. Brings up services if down, waits until
  Postgres + Ollama respond, then runs an incremental index. Logs to logs/.

  Wired to run "At log on" + "Daily" via Windows Task Scheduler (see scripts/install-schedule.ps1).
  Incremental: only changed notes are re-embedded, so a no-change run is cheap.
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:PYTHONIOENCODING = 'utf-8'   # avoid cp1252 crashes on accented headings
$logDir = Join-Path $repo 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("reindex-{0}.log" -f (Get-Date -Format 'yyyyMMdd'))

function Log($msg) {
  $line = "{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

function Wait-For($name, [scriptblock]$check, $tries = 30, $delay = 5) {
  for ($i = 1; $i -le $tries; $i++) {
    try { if (& $check) { Log "$name pronto"; return $true } } catch {}
    Log "$name non pronto (tentativo $i/$tries), attendo ${delay}s..."
    Start-Sleep -Seconds $delay
  }
  return $false
}

try {
  Log "=== reindex start ==="

  # 1. Postgres (Docker). up -d is idempotent.
  docker compose up -d db | Out-Null
  $pgOk = Wait-For 'Postgres' { docker compose exec -T db pg_isready -U sparcos -q; $LASTEXITCODE -eq 0 }
  if (-not $pgOk) { Log 'ERRORE: Postgres non risponde. Docker Desktop attivo?'; exit 2 }

  # 2. Ollama (embedding model host).
  $ollamaOk = Wait-For 'Ollama' {
    (Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -UseBasicParsing -TimeoutSec 4).StatusCode -eq 200
  }
  if (-not $ollamaOk) { Log 'ERRORE: Ollama non risponde su :11434.'; exit 3 }

  # 3. Incremental index.
  $out = & (Join-Path $repo '.venv\Scripts\sparcos-rag.exe') index 2>&1
  Log ("index -> " + ($out -join ' '))

  # 4. Freshness verdict (read-only). exit 1 from status == still stale.
  $st = & (Join-Path $repo '.venv\Scripts\sparcos-rag.exe') status 2>&1
  Log ("status -> " + ($st -join ' '))

  Log "=== reindex done ==="
}
catch {
  Log ("ECCEZIONE: " + $_.Exception.Message)
  exit 1
}
