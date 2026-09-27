#!/usr/bin/env bash
# One-command reviewer demo: builds and starts the existing Docker Compose stack, waits until it
# is actually usable (container health + real HTTP endpoints), then prints the dashboard URL and
# opens it when a browser opener is available. Safe to run repeatedly: normal `docker compose up`
# semantics, volumes (database data) are never removed, an existing .env is never overwritten.
#
# Environment overrides:
#   DEMO_TIMEOUT=300          seconds to wait for readiness after the stack is started
#   DEMO_NO_BROWSER=1         do not try to open a browser
#   EMULATOR_TAR=/path/ndtp-telemetry-emulator.tar
#                             load the organizer NDTP emulator image if it is not present yet
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TIMEOUT="${DEMO_TIMEOUT:-300}"
EMULATOR_IMAGE="ndtp-telemetry-emulator:1.0"   # docker-compose.yml, service "emulator"

step() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ prerequisites
step "Checking Docker"
command -v docker >/dev/null 2>&1 || die "Docker is not installed. Install Docker (with Compose v2) and run ./run-demo.sh again."
docker info >/dev/null 2>&1 || die "Docker is installed but not responding. Start Docker (Docker Desktop on WSL: enable WSL integration for this distro) and run ./run-demo.sh again."
docker compose version >/dev/null 2>&1 || die "'docker compose' (Compose v2) is not available."
note "$(docker compose version)"

step "Checking configuration"
if [ ! -f .env ]; then
  [ -f .env.example ] || die ".env and .env.example are both missing."
  cp .env.example .env
  note "Created .env from .env.example (existing .env files are never overwritten)."
else
  note "Using existing .env"
fi

env_value() {  # value of NAME: shell environment first, then .env (as docker compose resolves it)
  local name="$1" value="${!1:-}"
  if [ -z "$value" ]; then
    value="$(grep -E "^${name}=" .env 2>/dev/null | tail -n 1 | cut -d= -f2- | tr -d '\r' | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
  fi
  printf '%s' "$value"
}

MAPS_KEY="$(env_value VITE_YANDEX_MAPS_API_KEY)"
if [ -z "$MAPS_KEY" ] || [ "$MAPS_KEY" = "your_yandex_maps_js_api_key" ]; then
  warn "VITE_YANDEX_MAPS_API_KEY in .env is not set: the dashboard works, but map tiles will not load."
fi

# Model artifact mounted into the ml service (docker-compose.yml: ${ML_ARTIFACT_DIR:-./artifacts/...}).
ARTIFACT_DIR="$(env_value ML_ARTIFACT_DIR)"
ARTIFACT_DIR="${ARTIFACT_DIR:-./artifacts/hgb-h0-runtime-safe-v1-group-a-v1}"
[ -f "$ARTIFACT_DIR/bundle.json" ] || die "Model artifact not found: $ARTIFACT_DIR/bundle.json (ML_ARTIFACT_DIR in .env)."
MODEL_FILE="$(sed -n 's/.*"model_filename"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$ARTIFACT_DIR/bundle.json")"
[ -n "$MODEL_FILE" ] && [ -f "$ARTIFACT_DIR/$MODEL_FILE" ] || die "Model file named in $ARTIFACT_DIR/bundle.json is missing (${MODEL_FILE:-unknown})."
[ -f "$ARTIFACT_DIR/manifest.json" ] || die "Model manifest missing: $ARTIFACT_DIR/manifest.json"
note "Model artifact: $ARTIFACT_DIR ($MODEL_FILE)"

# The organizer NDTP emulator image is not published in a registry; Docker cannot pull it.
if ! docker image inspect "$EMULATOR_IMAGE" >/dev/null 2>&1; then
  if [ -n "${EMULATOR_TAR:-}" ]; then
    [ -f "$EMULATOR_TAR" ] || die "EMULATOR_TAR does not exist: $EMULATOR_TAR"
    note "Loading $EMULATOR_IMAGE from $EMULATOR_TAR"
    docker load -i "$EMULATOR_TAR" >/dev/null || die "docker load failed for $EMULATOR_TAR"
  else
    die "Docker image $EMULATOR_IMAGE is missing. It ships with the organizer dataset as ndtp-telemetry-emulator.tar:
       docker load -i /path/to/ndtp-telemetry-emulator.tar
   or: EMULATOR_TAR=/path/to/ndtp-telemetry-emulator.tar ./run-demo.sh"
  fi
fi
note "Emulator image: $EMULATOR_IMAGE"

# ------------------------------------------------------------------ build / start
step "Building and starting the stack (first build can take several minutes)"
docker compose up -d --build || {
  docker compose ps -a || true
  die "docker compose up failed (see the output above)."
}

# ------------------------------------------------------------------ readiness
http_status() {  # HTTP status code of URL (000 if unreachable), no proxies for localhost
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    local code
    code="$(curl --noproxy '*' -s -o /dev/null -m 5 -w '%{http_code}' "$url" 2>/dev/null)" || true
    printf '%s' "${code:-000}"
  elif command -v wget >/dev/null 2>&1; then
    wget --no-proxy -q -t 1 -T 5 -S -O /dev/null "$url" 2>&1 | awk '/^  HTTP\//{code=$2} END{print code ? code : "000"}'
  else
    # Last resort without curl/wget: bash /dev/tcp, bounded (a connect can hang on filtered ports).
    local probe='rest="${1#http://}"; hp="${rest%%/*}"; p="/${rest#*/}"; [ "$rest" = "$hp" ] && p="/"
      exec 3<>"/dev/tcp/${hp%:*}/${hp##*:}" || exit 1
      printf "GET %s HTTP/1.0\r\nHost: %s\r\n\r\n" "$p" "$hp" >&3
      IFS=" " read -r _ code _ <&3; printf "%s" "$code"'
    local code
    if command -v timeout >/dev/null 2>&1; then
      code="$(timeout 5 bash -c "$probe" _ "$url" 2>/dev/null)" || true
    else
      code="$(bash -c "$probe" _ "$url" 2>/dev/null)" || true
    fi
    printf '%s' "${code:-000}"
  fi
}

published_url() {  # http URL of a service's published port, from the running compose project
  local service="$1" port="$2" mapping
  mapping="$(docker compose port "$service" "$port" 2>/dev/null | head -n 1 || true)"
  [ -n "$mapping" ] || return 1
  printf 'http://localhost:%s' "${mapping##*:}"
}

container_state() {  # health status if the service has a healthcheck, else the container state
  local id
  id="$(docker compose ps -q "$1" 2>/dev/null | head -n 1)" || true
  [ -n "$id" ] || { printf 'missing'; return; }
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$id" 2>/dev/null || printf 'unknown'
}

step "Waiting for the application to become ready (timeout ${TIMEOUT}s)"
BACKEND_URL="$(published_url backend 3000 || true)"
DASHBOARD_URL="$(published_url frontend 80 || true)"
[ -n "$BACKEND_URL" ] || die "backend has no published port for 3000 (docker compose port backend 3000)."
[ -n "$DASHBOARD_URL" ] || die "frontend has no published port for 80 (docker compose port frontend 80)."

deadline=$((SECONDS + TIMEOUT))
failed=""
while :; do
  ml="$(container_state ml)"
  pg="$(container_state postgres)"
  backend="$(http_status "$BACKEND_URL/health")"
  frontend="$(http_status "$DASHBOARD_URL/")"
  printf '    ml=%s  postgres=%s  backend /health=%s  dashboard=%s\n' "$ml" "$pg" "$backend" "$frontend"
  if [ "$ml" = healthy ] && [ "$pg" = healthy ] && [ "$backend" = 200 ] && [ "$frontend" = 200 ]; then
    break
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    [ "$ml" = healthy ] || failed="$failed ml"
    [ "$pg" = healthy ] || failed="$failed postgres"
    [ "$backend" = 200 ] || failed="$failed backend"
    [ "$frontend" = 200 ] || failed="$failed frontend"
    break
  fi
  sleep 3
done

if [ -n "$failed" ]; then
  printf '\nNot ready after %ss. Failed check(s):%s\n' "$TIMEOUT" "$failed" >&2
  [ "${ml:-}" = unhealthy ] && note "ml is unhealthy: /ready returns 503 until the model artifact loads (see logs below)." >&2
  printf '\n--- docker compose ps ---\n' >&2
  docker compose ps -a >&2 || true
  for service in $failed; do
    printf '\n--- last log lines: %s ---\n' "$service" >&2
    docker compose logs --no-color --tail=40 "$service" >&2 || true
  done
  exit 1
fi

# The emulator is configured by the one-shot emulator-init service; report (do not fail) if it did not succeed.
init_id="$(docker compose ps -a -q emulator-init 2>/dev/null | head -n 1 || true)"
if [ -n "$init_id" ]; then
  init_state="$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$init_id" 2>/dev/null || true)"
  case "$init_state" in
    "exited 0"|running*|created*) ;;
    *) warn "emulator-init: $init_state - live telemetry may not start (docker compose logs emulator-init)." ;;
  esac
fi

# ------------------------------------------------------------------ success
printf '\n==================================================\n'
printf '  Application is ready:\n  %s\n' "$DASHBOARD_URL"
printf '==================================================\n'
printf 'Stop: docker compose down   (keeps database data)\n'

open_browser() {  # convenience only: never fails the launcher
  local url="$1"
  [ -z "${DEMO_NO_BROWSER:-}" ] || return 0
  if grep -qi microsoft /proc/version 2>/dev/null; then
    if command -v wslview >/dev/null 2>&1; then wslview "$url" >/dev/null 2>&1 && return 0; fi
    if command -v powershell.exe >/dev/null 2>&1; then
      powershell.exe -NoProfile -NonInteractive -Command "Start-Process '$url'" >/dev/null 2>&1 && return 0
    fi
  fi
  if [ "$(uname -s)" = Darwin ] && command -v open >/dev/null 2>&1; then open "$url" >/dev/null 2>&1 && return 0; fi
  if command -v xdg-open >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    (xdg-open "$url" >/dev/null 2>&1 &) && return 0
  fi
  printf '(Open the URL above in your browser.)\n'
}
open_browser "$DASHBOARD_URL" || true
