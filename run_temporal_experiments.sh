#!/bin/bash

set -euo pipefail

DATA="Small_LI"
MODEL="gin"
EPOCHS=20

mkdir -p experiment_logs

echo "========================================"
echo "RUN 1/5: BASELINE"
echo "========================================"

python main.py \
    --data "$DATA" \
    --model "$MODEL" \
    --n_epochs "$EPOCHS" \
    --tqdm \
    2>&1 | tee experiment_logs/01_baseline.log


echo
echo "========================================"
echo "RUN 2/5: MULTI-GNN"
echo "EMLPS + Reverse MP + Ports + Ego IDs"
echo "========================================"

python main.py \
    --data "$DATA" \
    --model "$MODEL" \
    --n_epochs "$EPOCHS" \
    --emlps \
    --reverse_mp \
    --ports \
    --ego \
    --tqdm \
    2>&1 | tee experiment_logs/02_multignn.log


echo
echo "========================================"
echo "RUN 3/5: MULTI-GNN + TDS"
echo "========================================"

python main.py \
    --data "$DATA" \
    --model "$MODEL" \
    --n_epochs "$EPOCHS" \
    --emlps \
    --reverse_mp \
    --ports \
    --ego \
    --tds \
    --tqdm \
    2>&1 | tee experiment_logs/03_multignn_tds.log


echo
echo "========================================"
echo "RUN 4/5: MULTI-GNN + FLOW TDS"
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
    2>&1 | tee experiment_logs/04_multignn_flow_tds.log


echo
echo "========================================"
echo "RUN 5/5: MULTI-GNN + ROLLING VELOCITY"
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
echo "ALL FIVE RUNS COMPLETE"
echo "========================================"