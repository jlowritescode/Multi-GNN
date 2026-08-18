#!/bin/bash

set -e

DATA="Small_HI_patterns"
EPOCHS=10

for SEED in 1 2 3 4 5
do
    echo ""
    echo "======================================================"
    echo "SEED $SEED — MULTI-GIN BASELINE"
    echo "======================================================"

    python main.py \
        --data "$DATA" \
        --model gin \
        --emlps \
        --reverse_mp \
        --ports \
        --ego \
        --seed "$SEED" \
        --n_epochs "$EPOCHS"

done

    echo ""
    echo "======================================================"
    echo "SEED $SEED — MULTI-GIN + FLOW_TDS + TDS"
    echo "======================================================"

    python main.py \
        --data "$DATA" \
        --model gin \
        --emlps \
        --reverse_mp \
        --ports \
        --ego \
        --tds \
        --flow_tds \
        --seed "$SEED" \
        --n_epochs "$EPOCHS"


    echo ""
    echo "======================================================"
    echo "SEED $SEED — MULTI-GIN + FLOW_TDS+ rolling velocity"
    echo "======================================================"

for SEED in 1 2 3 4 5
do
    python main.py \
        --data "$DATA" \
        --model gin \
        --emlps \
        --reverse_mp \
        --ports \
        --ego \
        -- tds \
        --flow_tds \
        --rolling_velocity

        --seed "$SEED" \
        --n_epochs "$EPOCHS"

done

echo ""
echo "======================================================"
echo "ALL 15 RUNS COMPLETE"
echo "======================================================"