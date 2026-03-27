param(
  [string]$HostAlias = "nus_hopper",
  [string]$RemoteRoot = "/scratch/e1561245/cot_yz",
  [string]$ProjectName = "codar_v1",
  [string]$RemoteOutputRoot = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$bundle = Join-Path $env:TEMP ("{0}_{1}.tar.gz" -f $ProjectName, (Get-Date -Format "yyyyMMdd_HHmmss"))
$remoteBundle = "$RemoteRoot/${ProjectName}_bundle.tar.gz"
$remoteProject = "$RemoteRoot/$ProjectName"
if ([string]::IsNullOrWhiteSpace($RemoteOutputRoot)) {
  $RemoteOutputRoot = "$RemoteRoot/codar_output/$ProjectName"
}

Write-Host "[deploy] projectRoot=$projectRoot"
Write-Host "[deploy] bundle=$bundle"
Write-Host "[deploy] remoteProject=$remoteProject"
Write-Host "[deploy] remoteOutputRoot=$RemoteOutputRoot"

Push-Location $projectRoot
try {
  if (Test-Path $bundle) { Remove-Item -Force $bundle }
  $tarArgs = @(
    "-czf", $bundle,
    "--exclude=.venv",
    "--exclude=__pycache__",
    "--exclude=.pytest_cache",
    "--exclude=output",
    "--exclude=cache",
    "--exclude=*.pyc",
    "."
  )
  & tar @tarArgs

  ssh $HostAlias "mkdir -p $RemoteRoot $remoteProject $RemoteOutputRoot/pbs_logs"
  scp $bundle "$HostAlias`:$remoteBundle"
  ssh $HostAlias "mkdir -p $remoteProject && tar -xzf $remoteBundle -C $remoteProject --strip-components=1 && rm -f $remoteBundle && mkdir -p $RemoteOutputRoot/pbs_logs"
  Write-Host "[deploy] deployed to ${HostAlias}:$remoteProject"
}
finally {
  Pop-Location
  if (Test-Path $bundle) { Remove-Item -Force $bundle }
}
