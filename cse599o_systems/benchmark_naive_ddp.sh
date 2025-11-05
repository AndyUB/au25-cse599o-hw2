TOY_DIR=../bench/naive_ddp_toy
mkdir -p $TOY_DIR

CKPT_PATH=$TOY_DIR/toy.pt

uv run benchmark_naive_ddp.py --toy_ckpt_path $CKPT_PATH > $TOY_DIR/toy.log 2>&1
