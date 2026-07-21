<#
.SYNOPSIS
  Registers (or refreshes) the Windows Task Scheduler job that keeps the
  sparcOS RAG index fresh: runs reindex.ps1 daily at 16:30.

  If the PC is off at 16:30 the run is simply skipped (no catch-up):
  StartWhenAvailable is left OFF on purpose.

  Idempotent: re-running replaces the existing task. No admin required
  (runs as the current user, only while logged on).

  Uninstall:  Unregister-ScheduledTask -TaskName 'sparcOS-RAG-Reindex' -Confirm:$false
#>
$ErrorActionPreference = 'Stop'
$taskName = 'sparcOS-RAG-Reindex'
$script   = Join-Path $PSScriptRoot 'reindex.ps1'

$action = New-ScheduledTaskAction `
  -Execute 'powershell.exe' `
  -Argument ("-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"{0}`"" -f $script)

$daily = New-ScheduledTaskTrigger -Daily -At 16:30

# No -StartWhenAvailable: a missed run (PC off at 16:30) is skipped, not caught up later.
$settings = New-ScheduledTaskSettingsSet `
  -DontStopOnIdleEnd `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask `
  -TaskName $taskName `
  -Action $action `
  -Trigger $daily `
  -Settings $settings `
  -Description 'sparcOS RAG: incremental reindex to keep the vault index fresh (daily 16:30, skipped if PC off).' `
  -Force | Out-Null

Write-Output "Registrato task '$taskName' (Daily 16:30, nessun recupero se il PC e' spento)."
