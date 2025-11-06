import argparse
import os
import torch
from cse599o_basics.util import AdamW, cross_entropy_loss
from cse599o_basics.transformer import Transformer

from benchmark_naive_ddp import (
    SEED,
    TRANSFORMER_ARGS,
    ADAMW_ARGS,
    WORLD_SIZE,
    LOCAL_BATCH_SIZE,
    CONTEXT_LENGTH,
    set_seed,
)
from benchmark_optimized_ddp import NUM_ITERS, NUM_WARMUP


def run_baseline_transformer(
    baseline_path: str,
    start_path: str,
    data_path: str,
) -> None:
    device = torch.device("cuda:0")

    set_seed(SEED)
    global_seqs = torch.randint(
        low=0,
        high=TRANSFORMER_ARGS["vocab_size"],
        size=(
            WORLD_SIZE * LOCAL_BATCH_SIZE,
            CONTEXT_LENGTH + 1,
        ),
    ).cpu()
    torch.save(global_seqs, data_path)
    global_seqs = global_seqs.to(device)

    model = Transformer(
        context_length=CONTEXT_LENGTH,
        device=device,
        **TRANSFORMER_ARGS,
    ).to(device)
    torch.save(model.state_dict(), start_path)
    optimizer = AdamW(model.parameters(), **ADAMW_ARGS)

    input_ids = global_seqs[:, :CONTEXT_LENGTH]
    target_ids = global_seqs[:, 1 : CONTEXT_LENGTH + 1]

    for _ in range(NUM_WARMUP + NUM_ITERS):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = cross_entropy_loss(logits, target_ids)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.save(model.state_dict(), baseline_path)


def verify_optimized_ddp_main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "verify"],
        default="verify",
        help="baseline: Run baseline to generate `start.pt`, `data.pt`, "
        "and `baseline.pt` checkpoints; verify: Verify DDP modes.",
    )
    arg_parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory.",
    )
    arg_parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for verification.",
    )
    args = arg_parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    baseline_path = os.path.join(args.ckpt_dir, "baseline.pt")
    if args.mode == "baseline" or not os.path.exists(baseline_path):
        start_path = os.path.join(args.ckpt_dir, "start.pt")
        data_path = os.path.join(args.ckpt_dir, "data.pt")
        run_baseline_transformer(baseline_path, start_path, data_path)

    if args.mode == "baseline":
        return

    baseline_state_dict = torch.load(baseline_path, map_location="cpu")

    for mode in ["naive", "flat", "individual", "bucketed"]:
        print(f"=== Mode: {mode} ===")
        ckpt_path = os.path.join(args.ckpt_dir, f"{mode}.pt")
        if not os.path.exists(ckpt_path):
            print(f"No checkpoint for {mode} mode, skipping verification.")
            continue
        ddp_state_dict = torch.load(ckpt_path, map_location="cpu")

        failed = False
        overall_max_diff = args.atol
        for name in baseline_state_dict:
            if not torch.allclose(
                baseline_state_dict[name], ddp_state_dict[name], atol=args.atol
            ):
                max_diff = torch.max(
                    torch.abs(baseline_state_dict[name] - ddp_state_dict[name])
                ).item()
                print(f"Mode {mode}, parameter {name}, max difference: {max_diff}")
                failed = True
                overall_max_diff = max(overall_max_diff, max_diff)

        print(f"Verification {'failed' if failed else 'passed'} for mode {mode}.")
        print(f"Upper bound on max difference: {overall_max_diff}.")


if __name__ == "__main__":
    verify_optimized_ddp_main()
