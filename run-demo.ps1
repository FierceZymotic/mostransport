# One-command reviewer demo for Windows PowerShell: same flow as run-demo.sh. Builds and starts
# the existing Docker Compose stack, waits until it is actually usable (container health + real
# HTTP endpoints), then prints the dashboard URL and opens it. Safe to run repeatedly: volumes
# (database data) are never removed, an existing .env is never overwritten.
#
#   $env:DEMO_TIMEOUT = 300        seconds to wait for readiness after the stack is started
#   $env:DEMO_NO_BROWSER = 1       do not open a browser
#   $env:EMULATOR_TAR = 'C:\path\ndtp-telemetry-emulator.tar'   load the organizer emulator image if missing

Set-Location -LiteralPath $PSScriptRoot
$Timeout = if ($env:DEMO_TIMEOUT) { [int]$env:DEMO_TIMEOUT } else { 300 }
$EmulatorImage = 'ndtp-telemetry-emulator:1.0'   # docker-compose.yml, service "emulator"

function Step([string]$Message) { Write-Host "`n==> $Message" }
function Note([string]$Message) { Write-Host "    $Message" }
function Fail([string]$Message) { Write-Host "`nERROR: $Message" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------------ prerequisites
Step 'Checking Docker'
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail 'Docker is not installed. Install Docker Desktop and run .\run-demo.ps1 again.' }
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail 'Docker is installed but not responding. Start Docker Desktop and run .\run-demo.ps1 again.' }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { Fail "'docker compose' (Compose v2) is not available." }

Step 'Checking configuration'
if (-not (Test-Path -LiteralPath '.env')) {
    if (-not (Test-Path -LiteralPath '.env.example')) { Fail '.env and .env.example are both missing.' }
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Note 'Created .env from .env.example (existing .env files are never overwritten).'
} else {
    Note 'Using existing .env'
}

function Get-EnvValue([string]$Name) {  # shell environment first, then .env (as docker compose resolves it)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if (-not $value) {
        $line = Get-Content -LiteralPath '.env' | Where-Object { $_ -match "^$Name=" } | Select-Object -Last 1
        if ($line) { $value = ($line -replace "^$Name=", '').Trim().Trim('"').Trim("'") }
    }
    return $value
}

$mapsKey = Get-EnvValue 'VITE_YANDEX_MAPS_API_KEY'
if (-not $mapsKey -or $mapsKey -eq 'your_yandex_maps_js_api_key') {
    Write-Warning 'VITE_YANDEX_MAPS_API_KEY in .env is not set: the dashboard works, but map tiles will not load.'
}

$artifactDir = Get-EnvValue 'ML_ARTIFACT_DIR'
if (-not $artifactDir) { $artifactDir = './artifacts/hgb-h0-runtime-safe-v1-group-a-v1' }
$bundle = Join-Path $artifactDir 'bundle.json'
if (-not (Test-Path -LiteralPath $bundle)) { Fail "Model artifact not found: $bundle (ML_ARTIFACT_DIR in .env)." }
$modelFile = (Get-Content -Raw -LiteralPath $bundle | ConvertFrom-Json).model_filename
if (-not $modelFile -or -not (Test-Path -LiteralPath (Join-Path $artifactDir $modelFile))) { Fail "Model file named in $bundle is missing ($modelFile)." }
if (-not (Test-Path -LiteralPath (Join-Path $artifactDir 'manifest.json'))) { Fail "Model manifest missing: $(Join-Path $artifactDir 'manifest.json')" }
Note "Model artifact: $artifactDir ($modelFile)"

docker image inspect $EmulatorImage *> $null
if ($LASTEXITCODE -ne 0) {
    if ($env:EMULATOR_TAR) {
        if (-not (Test-Path -LiteralPath $env:EMULATOR_TAR)) { Fail "EMULATOR_TAR does not exist: $env:EMULATOR_TAR" }
        Note "Loading $EmulatorImage from $env:EMULATOR_TAR"
        docker load -i $env:EMULATOR_TAR | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail "docker load failed for $env:EMULATOR_TAR" }
    } else {
        Fail ("Docker image $EmulatorImage is missing. It ships with the organizer dataset as ndtp-telemetry-emulator.tar:`n" +
              "       docker load -i C:\path\to\ndtp-telemetry-emulator.tar`n" +
              "   or: `$env:EMULATOR_TAR = 'C:\path\to\ndtp-telemetry-emulator.tar'; .\run-demo.ps1")
    }
}
Note "Emulator image: $EmulatorImage"

# ------------------------------------------------------------------ build / start
Step 'Building and starting the stack (first build can take several minutes)'
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { docker compose ps -a; Fail 'docker compose up failed (see the output above).' }

# ------------------------------------------------------------------ readiness
function Get-PublishedUrl([string]$Service, [int]$Port) {
    $mapping = docker compose port $Service $Port 2>$null | Select-Object -First 1
    if (-not $mapping) { return $null }
    return 'http://localhost:' + ($mapping -split ':')[-1]
}

function Get-ContainerState([string]$Service) {
    $id = docker compose ps -q $Service 2>$null | Select-Object -First 1
    if (-not $id) { return 'missing' }
    $state = docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' $id 2>$null
    if ($state) { return $state } else { return 'unknown' }
}

function Get-HttpStatus([string]$Url) {
    try {
        return [int](Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5).StatusCode
    } catch {
        if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
        return 0
    }
}

Step "Waiting for the application to become ready (timeout ${Timeout}s)"
$backendUrl = Get-PublishedUrl 'backend' 3000
$dashboardUrl = Get-PublishedUrl 'frontend' 80
if (-not $backendUrl) { Fail 'backend has no published port for 3000 (docker compose port backend 3000).' }
if (-not $dashboardUrl) { Fail 'frontend has no published port for 80 (docker compose port frontend 80).' }

$deadline = (Get-Date).AddSeconds($Timeout)
$failed = @()
while ($true) {
    $ml = Get-ContainerState 'ml'
    $pg = Get-ContainerState 'postgres'
    $backend = Get-HttpStatus "$backendUrl/health"
    $frontend = Get-HttpStatus "$dashboardUrl/"
    Note "ml=$ml  postgres=$pg  backend /health=$backend  dashboard=$frontend"
    if ($ml -eq 'healthy' -and $pg -eq 'healthy' -and $backend -eq 200 -and $frontend -eq 200) { break }
    if ((Get-Date) -ge $deadline) {
        if ($ml -ne 'healthy') { $failed += 'ml' }
        if ($pg -ne 'healthy') { $failed += 'postgres' }
        if ($backend -ne 200) { $failed += 'backend' }
        if ($frontend -ne 200) { $failed += 'frontend' }
        break
    }
    Start-Sleep -Seconds 3
}

if ($failed.Count -gt 0) {
    Write-Host "`nNot ready after ${Timeout}s. Failed check(s): $($failed -join ' ')" -ForegroundColor Red
    if ($ml -eq 'unhealthy') { Note 'ml is unhealthy: /ready returns 503 until the model artifact loads (see logs below).' }
    Write-Host "`n--- docker compose ps ---"
    docker compose ps -a
    foreach ($service in $failed) {
        Write-Host "`n--- last log lines: $service ---"
        docker compose logs --no-color --tail=40 $service
    }
    exit 1
}

# The emulator is configured by the one-shot emulator-init service; report (do not fail) if it did not succeed.
$initId = docker compose ps -a -q emulator-init 2>$null | Select-Object -First 1
if ($initId) {
    $initState = docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' $initId 2>$null
    if ($initState -and $initState -ne 'exited 0' -and $initState -notlike 'running*' -and $initState -notlike 'created*') {
        Write-Warning "emulator-init: $initState - live telemetry may not start (docker compose logs emulator-init)."
    }
}

# ------------------------------------------------------------------ success
Write-Host "`n=================================================="
Write-Host "  Application is ready:"
Write-Host "  $dashboardUrl" -ForegroundColor Green
Write-Host "=================================================="
Write-Host 'Stop: docker compose down   (keeps database data)'

if (-not $env:DEMO_NO_BROWSER) {
    try { Start-Process $dashboardUrl } catch { Write-Host '(Open the URL above in your browser.)' }
}
