#!/bin/bash
cd "$(dirname "$0")/.."
source ml_env/bin/activate
python3 backend/predict_images.py "$1" 2>/dev/null