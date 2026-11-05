#!/usr/bin/env bash
# ==============================================================================
# HydroPhysics AI: Local Startup & Execution Script
# ==============================================================================

set -e

PORT=${API_PORT:-8000}
HOST=${API_HOST:-"127.0.0.1"}

echo "----------------------------------------------------------------------"
echo "  Starting HydroPhysics AI Application"
echo "  Physics-Informed Machine Learning for Rainfall-Runoff Prediction"
echo "----------------------------------------------------------------------"

# 1. Check Python
PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3 is required but not found in PATH."
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# 2. Ensure benchmark data is generated
if [ ! -d "data/sample_camels_gb" ] || [ ! -f "data/sample_camels_gb/CAMELS_GB_catchment_attributes.csv" ]; then
    echo "Generating initial CAMELS-GB benchmark catchments..."
    $PYTHON_CMD -c "
from pathlib import Path
from src.data.downloader import generate_benchmark_camels_gb_sample
generate_benchmark_camels_gb_sample(Path('data/sample_camels_gb'))
"
fi

# 3. Launch FastAPI backend
echo "Starting FastAPI server on http://${HOST}:${PORT}..."
echo "Serving frontend at: http://${HOST}:${PORT}/"
echo "Interactive API docs: http://${HOST}:${PORT}/docs"
echo "Press Ctrl+C to terminate."
echo "----------------------------------------------------------------------"

exec $PYTHON_CMD -m uvicorn backend.main:app --host "${HOST}" --port "${PORT}"
