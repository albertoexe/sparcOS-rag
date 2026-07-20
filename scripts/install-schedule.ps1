<#
.SYNOPSIS
  Registers (or refreshes) the Windows Task Scheduler job that keeps the
  sparcOS RAG index fresh: runs reindex.ps1 at logon and daily at 08:00.

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

$atLogon = New-ScheduledTaskTrigger -AtLogOn
$daily   = New-ScheduledTaskTrigger -Daily -At 08:00
$triggers = @($atLogon, $daily)

$settings = New-ScheduledTaskSettingsSet `
  -StartWhenAvailable `
  -DontStopOnIdleEnd `
  -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask `
  -TaskName $taskName `
  -Action $action `
  -Trigger $triggers `
  -Settings $settings `
  -Description 'sparcOS RAG: incremental reindex to keep the vault index fresh (login + daily 08:00).' `
  -Force | Out-Null

Write-Output "Registrato task '$taskName' (At log on + Daily 08:00)."
