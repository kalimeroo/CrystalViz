<#
.SYNOPSIS
    Run the CrystalViz headless Blender test suite.

.DESCRIPTION
    Runs every tests/test_*.py with Blender in background mode, writes one log
    per test under tests/_logs/, and prints a PASS/FAIL summary.

    A test passes when its log reports "RESULT: ALL PASS", "RESULT: PASS" or a
    unittest "OK"; it fails on "RESULT: FAIL", a non-zero "TOTAL FAILURES", a
    "FAILED" unittest summary, or an uncaught traceback.

.PARAMETER Blender
    Path to blender.exe. Defaults to $env:BLENDER, then the Blender 5.2 install.

.PARAMETER Only
    Wildcard filter on the test file name, e.g. -Only "*polyhedra*".

.PARAMETER Timeout
    Per-test timeout in seconds (default 600).

.EXAMPLE
    pwsh tests/run_all.ps1
    pwsh tests/run_all.ps1 -Only "*gn_polyhedra*"
#>
[CmdletBinding()]
param(
    [string]$Blender = $(if ($env:BLENDER) { $env:BLENDER } else { "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" }),
    [string]$Only = "*",
    [int]$Timeout = 600
)

$ErrorActionPreference = "Stop"
$testsDir = $PSScriptRoot
$logsDir = Join-Path $testsDir "_logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

if (-not (Test-Path $Blender)) {
    Write-Error "blender.exe not found at '$Blender'. Pass -Blender or set `$env:BLENDER."
}

$tests = Get-ChildItem $testsDir -Filter "test_*.py" -File |
    Where-Object { $_.Name -like $Only } |
    Sort-Object Name

if (-not $tests) {
    Write-Error "No tests matched '$Only' in $testsDir."
}

$results = @()
foreach ($test in $tests) {
    $log = Join-Path $logsDir ($test.BaseName + ".log")
    $proc = Start-Process -FilePath $Blender `
        -ArgumentList @(
            "--background", "--factory-startup",
            "--python-exit-code", "1",
            "--python", "`"$($test.FullName)`""
        ) `
        -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError ($log + ".err")
    if (-not $proc.WaitForExit($Timeout * 1000)) {
        $proc.Kill()
        $results += [pscustomobject]@{ Test = $test.Name; Status = "TIMEOUT"; Seconds = $Timeout }
        continue
    }

    $text = if (Test-Path $log) { (Get-Content $log -Raw) } else { "" }
    $status = "UNKNOWN"
    if ($text -match "RESULT:\s*ALL PASS" -or $text -match "RESULT:\s*PASS") {
        $status = "PASS"
    } elseif ($text -match "TOTAL FAILURES:\s*[1-9]" -or $text -match "RESULT:\s*FAIL") {
        $status = "FAIL"
    } elseif ($text -match "(?m)^OK\b" -and $text -notmatch "FAILED") {
        $status = "PASS"
    } elseif ($text -match "(?m)^FAILED\b" -or $text -match "\nTraceback \(most recent call last\)") {
        $status = "FAIL"
    } elseif ($proc.ExitCode -ne 0) {
        $status = "FAIL"
    } else {
        $status = "PASS"   # clean exit with no failure marker
    }

    $results += [pscustomobject]@{
        Test    = $test.Name
        Status  = $status
        Seconds = [math]::Round(($proc.ExitTime - $proc.StartTime).TotalSeconds, 1)
    }
}

$results | Format-Table -AutoSize
$failed = @($results | Where-Object { $_.Status -ne "PASS" })
Write-Host ""
Write-Host ("{0} passed, {1} failed, {2} total" -f `
    (@($results | Where-Object { $_.Status -eq "PASS" }).Count), $failed.Count, $results.Count)
if ($failed.Count -gt 0) {
    Write-Host "Logs: $logsDir"
    exit 1
}
