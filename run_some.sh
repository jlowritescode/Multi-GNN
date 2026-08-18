#!/bin/bash

set -euo pipefail

DATA="Small_LI_patterns"
MODEL="gin"
EPOCHS=20


echo
echo "========================================"
echo "RUN: MULTI-GNN + ROLLING VELOCITY"
echo "========================================"

python main.py \
    --data "$DATA" \
    --model "$MODEL" \
    --n_epochs "$EPOCHS" \
    --emlps \
    --reverse_mp \
    --ports \
    --ego \
    --rolling_velocity \
    --tqdm \
    2>&1 | tee experiment_logs/05_multignn_rolling_velocity.log


echo
echo "========================================"
echo "RUN: MULTI-GNN + FLOW TDS"
echo "========================================"

python main.py \
    --data "$DATA" \
    --model "$MODEL" \
    --n_epochs "$EPOCHS" \
    --emlps \
    --reverse_mp \
    --ports \
    --ego \
    --flow_tds \
    --tqdm \
    2>&1 | tee experiment_logs/04_multignn_flow_tds2.log

echo
echo "========================================"
echo "RUN: MULTI-GNN + TDS + FLOW TDS"
echo "========================================"

python main.py \
    --data "$DATA" \
    --model "$MODEL" \
    --n_epochs "$EPOCHS" \
    --emlps \
    --reverse_mp \
    --ports \
    --ego \
    --flow_tds \
    --tqdm \
    2>&1 | tee experiment_logs/06_multignn_tds_flow_tds.log

echo
echo "========================================"
echo "ALL FIVE RUNS COMPLETE"
echo "========================================"