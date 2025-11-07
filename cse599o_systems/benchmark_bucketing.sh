#!/bin/bash

# Accept hostname as an argument
# If none provided, default to tempura
HOSTNAME=${1:-tempura}

LOG_DIR=../bench/bucketing/$HOSTNAME
mkdir -p $LOG_DIR

bucket_size_mb_list=(1 10 100 1000)
for bucket_size_mb in "${bucket_size_mb_list[@]}"; do
    # uv run nsys profile -o $LOG_DIR/${bucket_size_mb}mb \
    #     python benchmark_optimized_ddp.py \
    #     --mode bucketed \
    #     --bucket-size-mb $bucket_size_mb \
    #     --verbose > $LOG_DIR/${bucket_size_mb}mb.log 2>&1

    # No nsys profiling
    uv run benchmark_optimized_ddp.py \
        --mode bucketed \
        --bucket-size-mb $bucket_size_mb \
        --verbose > $LOG_DIR/${bucket_size_mb}mb.log 2>&1
done
