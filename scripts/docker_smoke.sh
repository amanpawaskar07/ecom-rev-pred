#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    docker compose -p revpred-p9 down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== DOCKER SMOKE START ==="

# 1. Build
echo "[1/5] Building images with project revpred-p9..."
docker compose -p revpred-p9 build

# 2. Up -d
echo "[2/5] Starting containers in detached mode..."
docker compose -p revpred-p9 up -d

# 3. Wait healthy (up to 60s)
echo "[3/5] Waiting for containers to become healthy..."
TIMEOUT=60
START=$(date +%s)
HEALTHY=0

while [ $(($(date +%s) - START)) -lt "$TIMEOUT" ]; do
    API_STATUS=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' revpred-p9-api-1 2>/dev/null || echo "missing")
    UI_STATUS=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' revpred-p9-ui-1 2>/dev/null || echo "missing")

    if [ "$API_STATUS" = "healthy" ] && [ "$UI_STATUS" = "healthy" ]; then
        HEALTHY=1
        break
    fi
    sleep 1
done

if [ "$HEALTHY" -ne 1 ]; then
    echo "ERROR: Timed out waiting for containers to become healthy after ${TIMEOUT}s" >&2
    docker compose -p revpred-p9 ps
    exit 1
fi
echo "Containers are healthy."

# 4. Curl /health
echo "[4/5] Checking API health..."
HEALTH_RESP=$(curl -s -f http://127.0.0.1:8000/health)
echo "GET /health -> $HEALTH_RESP"

# 5. POST /predict with an example from /meta
echo "[5/5] Querying /meta and sending POST /predict..."
META_RESP=$(curl -s -f http://127.0.0.1:8000/meta)
# Extract first example from /meta using python
PAYLOAD=$(python3 -c "import sys, json; data=json.loads(sys.stdin.read()); print(json.dumps(data['examples'][0]))" <<< "$META_RESP")
echo "Payload: $PAYLOAD"

PREDICT_RESP=$(curl -s -f -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")
echo "POST /predict -> $PREDICT_RESP"

# Down -v
echo "Tearing down containers..."
docker compose -p revpred-p9 down -v
echo "=== DOCKER SMOKE END: PASS ==="

