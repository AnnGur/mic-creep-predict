#!/usr/bin/env bash
# run_all.sh — Full pipeline runner for MIC Creep Prediction
#
# Prerequisites:
#   - Raw ATLAS data in data/raw/atlas_vivli_2004_2024.csv
#   - Python venv activated or use .venv/bin/python
#
# Usage:
#   bash run_all.sh
#   bash run_all.sh --skip-optuna   # faster training (no hyperparameter tuning)

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SKIP_OPTUNA=""

for arg in "$@"; do
  case "$arg" in
    --skip-optuna) SKIP_OPTUNA="--skip-optuna" ;;
  esac
done

echo "============================================"
echo " MIC Creep Prediction Pipeline"
echo " Vivli AMR Surveillance Challenge 2026"
echo "============================================"

# ── Step 1: EDA (both species, all charts) ──────────────────────────────────
echo ""
echo "[Step 1/4] Exploratory Data Analysis"
$PYTHON scripts/pipeline/1_run_atlas_eda.py

# ── Step 2: Feature engineering ─────────────────────────────────────────────
echo ""
echo "[Step 2/4] Feature Engineering"
for SPECIES in kpneumoniae abaumannii; do
  echo "  -> $SPECIES"
  $PYTHON scripts/pipeline/2_run_feature_engineering.py --species "$SPECIES"
done

# ── Step 3: Model training ───────────────────────────────────────────────────
echo ""
echo "[Step 3/4] Model Training (RF baseline + XGBoost tuned)"
for SPECIES in kpneumoniae abaumannii; do
  echo "  -> $SPECIES"
  $PYTHON scripts/pipeline/3_run_model_training.py --species "$SPECIES" $SKIP_OPTUNA
done

# ── Step 4: Export API artefacts ────────────────────────────────────────────
echo ""
echo "[Step 4/4] Export API Artefacts"
for SPECIES in kpneumoniae abaumannii; do
  echo "  -> $SPECIES"
  $PYTHON scripts/pipeline/4_run_export.py --species "$SPECIES"
done

echo ""
echo "============================================"
echo " Pipeline complete."
echo " Artefacts:"
echo "   models/           — .pkl model files"
echo "   reports/model/    — charts + model report"
echo "   reports/api/      — JSON artefacts for API"
echo "============================================"
