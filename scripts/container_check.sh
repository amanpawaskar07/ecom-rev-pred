#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-reports}"
mkdir -p "$OUT_DIR"

cleanup() {
    docker compose -p revpred-p9 down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== CONTAINER CHECK START ==="

# a. Build docker compose images
echo "[step a] Building docker compose images..."
docker compose -p revpred-p9 build

# b. Record start time, up -d, wait until both services are healthy
echo "[step b] Starting services and waiting for healthy status..."
TIMEOUT_S=$(PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python -c "from gates import lib as gl; print(gl.thr('phase9.healthcheck_timeout_s'))")

START_TIME=$(PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python -c "import time; print(time.time())")
docker compose -p revpred-p9 up -d

PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python -c "
import subprocess
import time
import sys
import urllib.request

timeout = float('$TIMEOUT_S')
start = float('$START_TIME')

def check_healthy():
    try:
        r_api = subprocess.check_output(
            ['docker', 'inspect', '--format', '{{.State.Health.Status}}', 'revpred-p9-api-1'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        r_ui = subprocess.check_output(
            ['docker', 'inspect', '--format', '{{.State.Health.Status}}', 'revpred-p9-ui-1'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if r_api == 'healthy' and r_ui == 'healthy':
            with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1.0) as resp:
                if resp.status != 200:
                    return False
            with urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=1.0) as resp:
                if resp.status != 200:
                    return False
            return True
    except Exception:
        return False
    return False

while time.time() - start < timeout:
    if check_healthy():
        elapsed = time.time() - start
        print(f'containers_healthy_within_s={elapsed:.2f}')
        sys.exit(0)
    time.sleep(0.5)

print(f'ERROR: Timed out waiting for containers to become healthy within {timeout}s', file=sys.stderr)
sys.exit(1)
"

# c. Run parity check (500 dev rows)
echo "[step c] Running parity checks..."
PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python scripts/parity.py --url http://localhost:8000 --n 500

LOCAL_OPT=""
if [ -n "${LOCAL_API_URL:-}" ]; then
    LOCAL_OPT="--local-url $LOCAL_API_URL"
fi
PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python scripts/compare_container_local.py \
    --container-url http://localhost:8000 \
    $LOCAL_OPT \
    --n 500 \
    --out-dir "$OUT_DIR"

# d. Benchmark container API
echo "[step d] Benchmarking container API..."
PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python scripts/bench_api.py \
    --url http://localhost:8000 \
    --n 500 \
    --out "$OUT_DIR/bench_api_container.json"

P95_MS=$(PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python -c "
import json
with open('$OUT_DIR/bench_api_container.json') as f:
    data = json.load(f)
print(data['predict']['p95_ms'])
")
echo "predict_p95_ms_container=$P95_MS"

# e. Print container /health model_sha and curl UI health
echo "[step e] Verifying health endpoints..."
CONTAINER_SHA=$(PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python -c "
import urllib.request
import json
with urllib.request.urlopen('http://localhost:8000/health') as resp:
    data = json.loads(resp.read().decode())
    print(data.get('model_sha', ''))
")
echo "container_model_sha=$CONTAINER_SHA"

UI_OK=$(PYTHONPATH=. PYTHONHASHSEED=0 .venv/bin/python -c "
import urllib.request
try:
    with urllib.request.urlopen('http://localhost:8501/_stcore/health') as resp:
        print(1 if resp.read().decode().strip() == 'ok' else 0)
except Exception:
    print(0)
")
echo "ui_health_ok=$UI_OK"

# f. Compose down and verify no containers remain
echo "[step f] Bringing compose project down..."
docker compose -p revpred-p9 down -v
REMAINING=$(docker compose -p revpred-p9 ps -aq | wc -l | tr -d ' ')
echo "compose_down_remaining_containers=$REMAINING"
if [ "$REMAINING" -eq 0 ]; then
    echo "compose_down_leaves_no_containers=1"
else
    echo "compose_down_leaves_no_containers=0"
fi

echo "=== CONTAINER CHECK END ==="

