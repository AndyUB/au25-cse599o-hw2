# benchmark_optimized_ddp.py
# -------------------------------------------------------------
# CSE 599O
#
# Extend your DDP benchmark to evaluate three optimized variants
# for the Transformer model:
#   (1) run_flat
#   (2) run_individual
#   (3) run_bucketed
#
# The TA will execute your script using commands like:
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode flat
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode individual
#     srun --gpus-per-node=2 uv run benchmark_optimized_ddp.py --mode bucketed --bucket-mb 10
#
# Each function should measure and print out the following statistics:
#   - iteration time per step  → append to iteration_times
#   - communication time per step → append to comm_times
# -------------------------------------------------------------

import argparse
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    get_model_dtype,
    compute_stats,
)
from ddp import DDPBucketed, DDPIndividualParameters

NUM_WARMUP = 2
NUM_ITERS = 5


def dist_setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12888"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def dist_cleanup() -> None:
    dist.destroy_process_group()


# ============================================================
# (0) Naive DDP
# ============================================================
def run_naive(
    model: torch.nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_iters: int,
    num_warmup: int,
    iteration_times: list[float],
    comm_times: list[float],
) -> dict:
    """A naive DDP training loop for reference."""
    input_ids, target_ids = data

    warmup_iter_times = []
    warmup_comm_times = []
    for _ in range(num_iters + num_warmup):
        iter_start_ev = torch.cuda.Event(enable_timing=True)
        iter_end_ev = torch.cuda.Event(enable_timing=True)
        comm_start_ev = torch.cuda.Event(enable_timing=True)
        comm_end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        iter_start_ev.record()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = cross_entropy_loss(logits, target_ids)
        loss.backward()

        comm_start_ev.record()
        for param in model.parameters():
            assert param.grad is not None
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= WORLD_SIZE
        comm_end_ev.record()

        optimizer.step()
        iter_end_ev.record()
        iter_end_ev.synchronize()
        torch.cuda.synchronize()

        # Times in milliseconds
        iter_time = iter_start_ev.elapsed_time(iter_end_ev)
        comm_time = comm_start_ev.elapsed_time(comm_end_ev)
        if _ >= num_warmup:
            iteration_times.append(iter_time)
            comm_times.append(comm_time)
        else:
            warmup_iter_times.append(iter_time)
            warmup_comm_times.append(comm_time)

    extra_stats = {
        "warmup_iteration_times": warmup_iter_times,
        "warmup_communication_times": warmup_comm_times,
    }
    return extra_stats


# ============================================================
# (1) Flat DDP
# ============================================================
def run_flat(
    model: torch.nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_iters: int,
    num_warmup: int,
    iteration_times: list[float],
    comm_times: list[float],
) -> dict:
    """All-reduce a single flattened gradient tensor."""
    input_ids, target_ids = data

    copy_times = []
    warmup_iter_times = []
    warmup_comm_times = []
    warmup_copy_times = []

    dtype = get_model_dtype(model)
    flat_tensor_size = sum(param.numel() for param in model.parameters())
    flat_grad = torch.empty(
        flat_tensor_size,
        dtype=dtype,
        device=input_ids.device,
        requires_grad=False,
    )

    for _ in range(num_iters + num_warmup):
        iter_start_ev = torch.cuda.Event(enable_timing=True)
        iter_end_ev = torch.cuda.Event(enable_timing=True)
        comm_start_ev = torch.cuda.Event(enable_timing=True)
        comm_end_ev = torch.cuda.Event(enable_timing=True)
        copy_start_ev = torch.cuda.Event(enable_timing=True)
        copy_end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        iter_start_ev.record()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = cross_entropy_loss(logits, target_ids)
        loss.backward()

        copy_start_ev.record()
        # Copy gradients into flat tensor
        offset = 0
        for param in model.parameters():
            assert param.grad is not None
            numel = param.grad.numel()
            flat_grad[offset : offset + numel].copy_(param.grad.reshape(-1))
            param.grad = flat_grad[offset : offset + numel].view_as(param)
            offset += numel
        copy_end_ev.record()

        comm_start_ev.record()
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad /= WORLD_SIZE
        comm_end_ev.record()

        optimizer.step()
        iter_end_ev.record()
        iter_end_ev.synchronize()
        torch.cuda.synchronize()

        # Times in milliseconds
        iter_time = iter_start_ev.elapsed_time(iter_end_ev)
        comm_time = comm_start_ev.elapsed_time(comm_end_ev)
        copy_time = copy_start_ev.elapsed_time(copy_end_ev)
        if _ >= num_warmup:
            iteration_times.append(iter_time)
            comm_times.append(comm_time)
            copy_times.append(copy_time)
        else:
            warmup_iter_times.append(iter_time)
            warmup_comm_times.append(comm_time)
            warmup_copy_times.append(copy_time)

    avg_copy_time = np.mean(copy_times)
    std_copy_time = np.std(copy_times)
    pct_copy_time = 100 * np.mean(
        [ct / it for ct, it in zip(copy_times, iteration_times)]
    )
    extra_stats = {
        "warmup_iteration_times": warmup_iter_times,
        "warmup_communication_times": warmup_comm_times,
        "warmup_copy_times": warmup_copy_times,
        "copy_times": copy_times,
        "copy_time_avg": avg_copy_time,
        "copy_time_std": std_copy_time,
        "copy_time_pct": pct_copy_time,
    }
    return extra_stats


# ============================================================
# (2) Individual DDP
# ============================================================
def run_individual(
    model: torch.nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_iters: int,
    num_warmup: int,
    iteration_times: list[float],
    comm_times: list[float],
) -> dict:
    """All-reduce each parameter's gradient individually."""
    input_ids, target_ids = data
    ddp_model = DDPIndividualParameters(model)

    warmup_iter_times = []
    warmup_comm_times = []
    if dist.get_rank() == 0:
        print(
            "[Note] All-reduce communication is overlapped with the backward pass "
            "so backward computation happens concurrently with the period that is "
            "measured as the communication time."
        )
    for _ in range(num_iters + num_warmup):
        iter_start_ev = torch.cuda.Event(enable_timing=True)
        iter_end_ev = torch.cuda.Event(enable_timing=True)
        comm_start_ev = torch.cuda.Event(enable_timing=True)
        comm_end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        iter_start_ev.record()
        optimizer.zero_grad()
        logits = ddp_model(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        comm_start_ev.record()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        comm_end_ev.record()

        optimizer.step()
        iter_end_ev.record()
        iter_end_ev.synchronize()
        torch.cuda.synchronize()

        # Times in milliseconds
        iter_time = iter_start_ev.elapsed_time(iter_end_ev)
        comm_time = comm_start_ev.elapsed_time(comm_end_ev)
        if _ >= num_warmup:
            iteration_times.append(iter_time)
            comm_times.append(comm_time)
        else:
            warmup_iter_times.append(iter_time)
            warmup_comm_times.append(comm_time)

    extra_stats = {
        "warmup_iteration_times": warmup_iter_times,
        "warmup_communication_times": warmup_comm_times,
    }
    return extra_stats


# ============================================================
# (3) Bucketed DDP
# ============================================================
def run_bucketed(
    model: torch.nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_iters: int,
    num_warmup: int,
    iteration_times: list[float],
    comm_times: list[float],
    bucket_size_mb: int,
) -> dict:
    """Group gradients into buckets and all-reduce each bucket."""
    input_ids, target_ids = data
    ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)

    warmup_iter_times = []
    warmup_comm_times = []
    if dist.get_rank() == 0:
        print(
            "[Note] All-reduce communication is overlapped with the backward pass "
            "so backward computation happens concurrently with the period that is "
            "measured as the communication time."
        )
    for _ in range(num_iters + num_warmup):
        iter_start_ev = torch.cuda.Event(enable_timing=True)
        iter_end_ev = torch.cuda.Event(enable_timing=True)
        comm_start_ev = torch.cuda.Event(enable_timing=True)
        comm_end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        iter_start_ev.record()
        optimizer.zero_grad()
        logits = ddp_model(input_ids)
        loss = cross_entropy_loss(logits, target_ids)

        comm_start_ev.record()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        comm_end_ev.record()

        optimizer.step()
        iter_end_ev.record()
        iter_end_ev.synchronize()
        torch.cuda.synchronize()

        # Times in milliseconds
        iter_time = iter_start_ev.elapsed_time(iter_end_ev)
        comm_time = comm_start_ev.elapsed_time(comm_end_ev)
        if _ >= num_warmup:
            iteration_times.append(iter_time)
            comm_times.append(comm_time)
        else:
            warmup_iter_times.append(iter_time)
            warmup_comm_times.append(comm_time)

    extra_stats = {
        "warmup_iteration_times": warmup_iter_times,
        "warmup_communication_times": warmup_comm_times,
    }
    if dist.get_rank() == 0:
        bucket_stats = ddp_model.get_bucket_stats()
        extra_stats["num_buckets"] = len(bucket_stats)
        extra_stats["bucket_stats"] = [bs.to_dict() for bs in bucket_stats]
    return extra_stats


# ============================================================
# Benchmark Function
# ============================================================
# You can change the function and variable names as needed.
def benchmark_optimized_ddp(
    rank: int,
    mode: str,
    global_seqs: torch.Tensor,
    local_batch_size: int,
    context_length: int,
    verbose: bool,
    ckpt_dir: str | None,
    bucket_size_mb: int,
) -> None:
    """Benchmark DDP variants on the Transformer model."""
    num_iters, num_warmup = NUM_ITERS, NUM_WARMUP
    # Times in milliseconds
    iter_times, comm_times = [], []

    # DDP setup
    # Initialize distributed process group
    assert global_seqs.size(0) % local_batch_size == 0
    world_size = global_seqs.size(0) // local_batch_size
    dist_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Construct model and move to GPU
    model = Transformer(
        context_length=context_length,
        device=device,
        **TRANSFORMER_ARGS,
    ).to(device)
    if ckpt_dir is None:
        # Sync params and buffers
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)
    else:
        start_ckpt_path = os.path.join(ckpt_dir, "start.pt")
        assert os.path.exists(start_ckpt_path)
        # Load checkpoint
        model.load_state_dict(torch.load(start_ckpt_path, map_location=device))

    # Construct optimizer
    optimizer = AdamW(model.parameters(), **ADAMW_ARGS)

    # Get input data
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size
    seqs = global_seqs[start_idx:end_idx]
    input_ids = seqs[:, :context_length].to(device)
    target_ids = seqs[:, 1 : context_length + 1].to(device)
    data = (input_ids, target_ids)

    if rank == 0:
        print(f"Mode: {mode}")
    run_fn = None
    extra_args = {}
    if mode == "naive":
        run_fn = run_naive
    elif mode == "flat":
        run_fn = run_flat
    elif mode == "individual":
        run_fn = run_individual
    elif mode == "bucketed":
        run_fn = run_bucketed
        extra_args["bucket_size_mb"] = bucket_size_mb
    assert run_fn is not None, f"Invalid mode: {mode}"

    extra_stats = run_fn(
        model,
        data,
        optimizer,
        num_iters,
        num_warmup,
        iter_times,
        comm_times,
        **extra_args,
    )

    dist_cleanup()

    compute_stats(
        rank,
        bench_step_times=iter_times,
        bench_comm_times=comm_times,
        print_basic_stats=True,
        compute_extra_stats=verbose,
        print_extra_stats=verbose,
        model=model,
        world_size=WORLD_SIZE,
        extra_stats=extra_stats,
    )

    if rank == 0 and ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{mode}.pt")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark optimized DDP variants.")
    parser.add_argument(
        "--mode",
        type=str,
        default="flat",
        choices=["naive", "flat", "individual", "bucketed"],
        help="Select which DDP variant to benchmark.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=WORLD_SIZE,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=LOCAL_BATCH_SIZE,
        help="Local batch size per GPU.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Context length for the Transformer model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory. "
        "Specifying this will load the `start.pt` checkpoint "
        "inside the directory as the initial model state, "
        "and the training data will be loaded from the "
        "`data.pt` file. After training, the model state is "
        "saved in the same directory in `<mode>.pt`.",
    )

    parser.add_argument(
        "--bucket-size-mb",
        type=int,
        default=10,
        help="Bucket size (in MB) for the bucketed DDP variant.",
    )
    args = parser.parse_args()
    if args.verbose:
        print(f"Arguments: {args}")

    set_seed(SEED)
    data_shape = (
        args.world_size * args.local_batch_size,
        args.context_length + 1,
    )
    if args.ckpt_dir is not None:
        assert os.path.exists(os.path.join(args.ckpt_dir, "start.pt"))
        data_path = os.path.join(args.ckpt_dir, "data.pt")
        assert os.path.exists(data_path)

        # Load data from checkpoint directory
        global_seqs: torch.Tensor = torch.load(data_path).cpu()
        assert global_seqs.shape == data_shape
    else:
        # Generate random data
        global_seqs = torch.randint(
            low=0,
            high=TRANSFORMER_ARGS["vocab_size"],
            size=data_shape,
        ).cpu()

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        benchmark_optimized_ddp,
        args=(
            args.mode,
            global_seqs,
            args.local_batch_size,
            args.context_length,
            args.verbose,
            args.ckpt_dir,
            args.bucket_size_mb,
        ),
        nprocs=args.world_size,
        join=True,
    )
