param(
    [string]$VivadoBin = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$aeRoot = Split-Path -Parent $PSScriptRoot
$tbRoot = Join-Path $aeRoot "src\rtl_testbench"

if (-not $OutDir) {
    $OutDir = Join-Path $aeRoot "results\rtl\rerun"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if ($VivadoBin) {
    # Use the Windows batch launchers so PowerShell waits for each tool to
    # finish before inspecting its log.
    $xvlog = Join-Path $VivadoBin "xvlog.bat"
    $xelab = Join-Path $VivadoBin "xelab.bat"
    $xsim = Join-Path $VivadoBin "xsim.bat"
    $vivadoRoot = Split-Path -Parent $VivadoBin
} else {
    $xvlog = "xvlog"
    $xelab = "xelab"
    $xsim = "xsim"
    $vivadoRoot = $env:XILINX_VIVADO
}
$glbl = Join-Path $vivadoRoot "data\verilog\src\glbl.v"
if (-not (Test-Path -LiteralPath $glbl)) {
    throw "Vivado glbl.v not found; pass -VivadoBin or source the Vivado environment"
}

$cases = @(
    @{ Precision = "W4A4"; Stem = "W4A4_P";      Top = "INT4_INT4_P_tb" },
    @{ Precision = "W4A4"; Stem = "W4A4_D";      Top = "INT4_INT4_D_tb" },
    @{ Precision = "W4A4"; Stem = "W4A4_Hybrid"; Top = "Hybrid_INT4_INT4_PD_tb" },
    @{ Precision = "W3A4"; Stem = "W3A4_P";      Top = "INT4_INT3_P_tb" },
    @{ Precision = "W3A4"; Stem = "W3A4_D";      Top = "INT4_INT3_D_tb" },
    @{ Precision = "W3A4"; Stem = "W3A4_Hybrid"; Top = "W3A4_PD_tb" }
)

$failed = @()
foreach ($case in $cases) {
    $caseDir = Join-Path $OutDir $case.Stem
    New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
    Push-Location $caseDir
    try {
        $design = Join-Path $tbRoot ("{0}\{1}.v" -f $case.Precision, $case.Stem)
        $testbench = Join-Path $tbRoot ("{0}\{1}_tb.v" -f $case.Precision, $case.Stem)
        $snapshot = "sim_{0}" -f $case.Stem

        & $xvlog $design $testbench $glbl --log xvlog.log
        if ($LASTEXITCODE -ne 0) { throw "xvlog failed" }

        # DSP48E2's simulation model references the Vivado global module.
        # Elaborate glbl explicitly in addition to linking unisims_ver.
        & $xelab $case.Top glbl -L unisims_ver -timescale 1ns/1ps -s $snapshot --log xelab.log
        if (($LASTEXITCODE -ne 0) -or
            (Select-String -LiteralPath xelab.log -Pattern "ERROR:" -Quiet)) {
            throw "xelab failed"
        }

        & $xsim $snapshot -runall -log xsim.log
        if ($LASTEXITCODE -ne 0) { throw "xsim failed" }

        if (-not (Select-String -LiteralPath xsim.log -Pattern "ALL TESTS PASSED" -Quiet)) {
            throw "pass marker missing from xsim.log"
        }
        Write-Host "PASS $($case.Stem)"
    } catch {
        $failed += $case.Stem
        Write-Host "FAIL $($case.Stem): $($_.Exception.Message)"
    } finally {
        Pop-Location
    }
}

if ($failed.Count -gt 0) {
    throw "RTL simulation failures: $($failed -join ', ')"
}
Write-Host "RTL_SIM_PASS cases=$($cases.Count)"
