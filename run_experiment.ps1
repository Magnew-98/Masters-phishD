#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir
& (Join-Path $ProjectDir '.venv\Scripts\python.exe') -m src.evaluation.run_experiment @args
