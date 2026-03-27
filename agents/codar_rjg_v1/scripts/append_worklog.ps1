param(
  [string]$RoundId = "",
  [string]$JobId = "",
  [string]$State = "heartbeat",
  [string]$LastAction = "",
  [string]$NextAction = "",
  [string]$BlockedBy = "none",
  [string]$MetricsJson = "",
  [string]$DeltaJson = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logPath = Join-Path $root "logs/optimization_worklog.jsonl"
$ts = [DateTime]::UtcNow.ToString("o")

$metricsObj = $null
if (-not [string]::IsNullOrWhiteSpace($MetricsJson)) {
  try { $metricsObj = $MetricsJson | ConvertFrom-Json } catch { $metricsObj = $MetricsJson }
}
$deltaObj = $null
if (-not [string]::IsNullOrWhiteSpace($DeltaJson)) {
  try { $deltaObj = $DeltaJson | ConvertFrom-Json } catch { $deltaObj = $DeltaJson }
}

$row = [ordered]@{
  ts = $ts
  round_id = $RoundId
  job_id = $JobId
  state = $State
  current_metrics = $metricsObj
  delta_vs_baseline = $deltaObj
  last_action = $LastAction
  next_action = $NextAction
  blocked_by = $BlockedBy
}

New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null
Add-Content -Path $logPath -Value ($row | ConvertTo-Json -Depth 12 -Compress)
Write-Host ("[worklog] appended: {0}" -f $logPath)
