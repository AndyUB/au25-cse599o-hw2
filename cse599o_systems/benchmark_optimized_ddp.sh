#!/bin/bash

# Accept hostname as an argument
# If none provided, default to tempura
HOSTNAME=${1:-tempura}

LOG_DIR=../bench/optimized_ddp/$HOSTNAME
mkdir -p $LOG_DIR

modes=("naive" "flat" "individual" "bucketed")
for mode in "${modes[@]}"; do
    # uv run nsys profile -o $LOG_DIR/$mode \
    #     python benchmark_optimized_ddp.py --mode $mode \
    #     --verbose > $LOG_DIR/$mode.log 2>&1

    # No nsys profiling
    uv run benchmark_optimized_ddp.py --mode $mode \
        --verbose > $LOG_DIR/$mode.log 2>&1
done
