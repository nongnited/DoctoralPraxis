#!/bin/bash
# RAGShield — Train All 10 Classifiers
# Runs all training scripts in sequence.
# Classical models run in parallel where possible.
# BERT models run sequentially (GPU memory constraints).
#
# Expected total time: 12-24 hours depending on hardware
# GPU greatly reduces BERT/SecureBERT training time
#
# Usage:
#   bash 3_training/train_all.sh
#   bash 3_training/train_all.sh --classical-only
#   bash 3_training/train_all.sh --bert-only

set -e
cd "$(dirname "$0")/.."  # run from repo root

LOGDIR="logs/training"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M)

log() { echo "[$(date '+%H:%M:%S')] $*"; }
run_model() {
    local name=$1
    local script=$2
    log "Starting $name..."
    python -u "$script" 2>&1 | tee "$LOGDIR/${name}_${TS}.txt"
    log "$name complete"
}

CLASSICAL_ONLY=false
BERT_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--classical-only" ]] && CLASSICAL_ONLY=true
    [[ "$arg" == "--bert-only" ]]      && BERT_ONLY=true
done

log "======================================================"
log "RAGShield — Training Pipeline"
log "Started: $(date)"
log "======================================================"

if [[ "$BERT_ONLY" == false ]]; then
    log "── Classical Models ──────────────────────────────────"
    run_model "naive_bayes"          "3_training/train_naive_bayes.py"
    run_model "logistic_regression"  "3_training/train_logistic_regression.py"
    run_model "ridge"                "3_training/train_ridge.py"
    run_model "random_forest"        "3_training/train_random_forest.py"
    run_model "extra_trees"          "3_training/train_extra_trees.py"
    run_model "xgboost"              "3_training/train_xgboost.py"
    run_model "lightgbm"             "3_training/train_lightgbm.py"
    run_model "svm"                  "3_training/train_svm.py"
fi

if [[ "$CLASSICAL_ONLY" == false ]]; then
    log "── Deep Learning Models ──────────────────────────────"
    run_model "bert_base"   "3_training/train_bert_base.py"
    run_model "securebert"  "3_training/train_securebert.py"
fi

log "======================================================"
log "All training complete: $(date)"
log "Logs saved to: $LOGDIR"
log "Models saved to: ./models/"
log ""
log "Next: python 5_corpus/upload_corpus.py"
log "      then: run evaluation pipeline"
log "======================================================"
