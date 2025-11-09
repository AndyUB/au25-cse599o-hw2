#!/bin/bash

# Accept hostname as an argument
# If none provided, default to tempura
HOSTNAME=${1:-tempura}

LOG_DIR=../bench/sharding/$HOSTNAME
mkdir -p $LOG_DIR
MEMORY_DIR=$LOG_DIR/memory_profiles
mkdir -p $MEMORY_DIR

# uv run sharding_optimizer.py \
#     --mode memory \
#     --verbose \
#     --memory_output_dir $MEMORY_DIR > $LOG_DIR/memory.log 2>&1

uv run sharding_optimizer.py \
    --mode time \
    --verbose > $LOG_DIR/time.log 2>&1
