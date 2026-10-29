#!/usr/bin/env bash
set -euo pipefail

TMP_DIR=$(mktemp -d /tmp/clean_clone_XXXXXX)
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "=== CLEAN CLONE CHECK START ==="

# 1. Clone current repo into TMP_DIR
echo "[1/4] Cloning repository to $TMP_DIR..."
git clone . "$TMP_DIR"

CLONE_SHA=$(git -C "$TMP_DIR" rev-parse HEAD)
echo "clean_clone_sha=$CLONE_SHA"

# 2. Verify no data in clone
echo "[2/4] Verifying no data in clone..."
RAW_DATA=$(find "$TMP_DIR/data/raw" -name '*.csv' 2>/dev/null || true)
PROC_DATA=$(find "$TMP_DIR/data/processed" -name '*.parquet' 2>/dev/null || true)
if [ -n "$RAW_DATA" ] || [ -n "$PROC_DATA" ]; then
    echo "ERROR: clean clone contains data files!" >&2
    exit 1
fi
echo "Verified: clone has no data files."

# 3. Create venv and install dependencies
echo "[3/4] Creating virtual environment and installing dependencies..."
PYTHON_BIN=$(command -v python3.11 || command -v python3)
"$PYTHON_BIN" -m venv "$TMP_DIR/.venv"
"$TMP_DIR/.venv/bin/pip" install --upgrade pip
"$TMP_DIR/.venv/bin/pip" install -r "$TMP_DIR/requirements.txt"
"$TMP_DIR/.venv/bin/pip" install -e "$TMP_DIR"

# 4. Run make ci inside the clone
echo "[4/4] Running make ci inside clean clone..."
CI_EXIT_CODE=0
make -C "$TMP_DIR" ci || CI_EXIT_CODE=$?

echo "clean_clone_ci_exit_code=$CI_EXIT_CODE"
if [ "$CI_EXIT_CODE" -ne 0 ]; then
    echo "ERROR: make ci failed in clean clone with exit code $CI_EXIT_CODE" >&2
    exit "$CI_EXIT_CODE"
fi

echo "=== CLEAN CLONE CHECK END: PASS ==="

