#!/bin/bash

# Accept hostname as an argument
# If none provided, default to tempura
HOSTNAME=${1:-tempura}

LOG_DIR=../bench/optimized_ddp/$HOSTNAME
mkdir -p $LOG_DIR

CKPT_DIR=$LOG_DIR/ckpt
mkdir -p $CKPT_DIR

uv run verify_optimized_ddp.py --mode baseline --ckpt-dir $CKPT_DIR > $LOG_DIR/baseline.log 2>&1

# modes=("naive" "flat" "individual" "bucketed")
modes=("bucketed")
for mode in "${modes[@]}"; do
    uv run benchmark_optimized_ddp.py --mode $mode \
        --ckpt-dir $CKPT_DIR \
        --verbose > $LOG_DIR/$mode.log 2>&1
done

uv run verify_optimized_ddp.py --ckpt-dir $CKPT_DIR > $LOG_DIR/verify.log 2>&1
