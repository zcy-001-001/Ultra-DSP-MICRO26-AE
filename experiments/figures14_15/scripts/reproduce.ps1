$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    python .\scripts\simulate_figures.py
    python .\tests\test_simulator.py
}
finally {
    Pop-Location
}
