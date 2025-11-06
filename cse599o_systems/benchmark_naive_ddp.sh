#!/bin/bash

# Accept hostname as an argument
# If none provided, default to tempura
HOSTNAME=${1:-tempura}

TOY_DIR=../bench/naive_ddp_toy
mkdir -p $TOY_DIR

CKPT_PATH=$TOY_DIR/toy.pt

uv run benchmark_naive_ddp.py --verbose --toy_ckpt_path $CKPT_PATH > $TOY_DIR/toy.log 2>&1

TRANSFORMER_DIR=../bench/naive_ddp_transformer/$HOSTNAME
mkdir -p $TRANSFORMER_DIR

uv run benchmark_naive_ddp.py --model transformer \
    --verbose \
    --transformer_log_dir $TRANSFORMER_DIR > $TRANSFORMER_DIR/transformer.log 2>&1