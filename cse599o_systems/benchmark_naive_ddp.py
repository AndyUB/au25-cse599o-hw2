# benchmark_naive_ddp.py
# -------------------------------------------------------------
# CSE 599O: Distributed Training Basics
#
# Implement a naive DDP version that reproduces the same model
# state as single-process training.
#
# The TA will test your implementation with the following commands:
#
# 1. To verify that DDP matches baseline (toy model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model toy
# Expected output: "Naive DDP matches baseline!"
#
# 2. To output communication and step time (transformer model):
#     srun --gpus-per-node=2 uv run benchmark_naive_ddp.py --model transformer
# Expected output: communication and step time statistics
#
# -------------------------------------------------------------

import argparse
from dataclasses import dataclass
import json
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cse599o_basics.util import AdamW, cross_entropy_loss
from cse599o_basics.transformer import Transformer
from tests.common import ToyModel

SEED = 599
ADAMW_ARGS = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
}
TRANSFORMER_ARGS = {
    "vocab_size": 50257,
    "d_model": 1280,
    "d_ff": 5120,
    "num_layers": 36,
    "num_heads": 20,
}
WORLD_SIZE = 2
LOCAL_BATCH_SIZE = 4
CONTEXT_LENGTH = 128


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((output - target) ** 2)


def save_ckpt(
    data: torch.Tensor,
    target_list: list[torch.Tensor],
    model: torch.nn.Module,
    ckpt_path: str,
) -> None:
    torch.save(
        {
            "data": data,
            "target_list": target_list,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )


def load_ckpt(
    model: torch.nn.Module,
    ckpt_path: str,
    expected_num_steps: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model_state_dict"])
    data, target_list = ckpt["data"], ckpt["target_list"]
    if len(target_list) != expected_num_steps:
        raise ValueError(
            f"Checkpoint target list has length {len(target_list)}, "
            f"expected {expected_num_steps}"
        )
    return data, target_list


def get_data_targets_shard(
    data: torch.Tensor,
    target_list: list[torch.Tensor],
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Get the data and target shard for the given rank."""
    batch_size = data.shape[0]
    if batch_size % world_size != 0:
        raise ValueError(
            f"Batch size {batch_size} not divisible by world size {world_size}"
        )
    for i, target in enumerate(target_list):
        if target.shape[0] != batch_size:
            raise ValueError(
                f"Target {i} batch size {target.shape[0]} does not match data batch size {batch_size}"
            )

    shard_size = batch_size // world_size
    data_shard = data[rank * shard_size : (rank + 1) * shard_size].to(device)
    target_shard_list = [
        target[rank * shard_size : (rank + 1) * shard_size].to(device)
        for target in target_list
    ]
    return data_shard, target_shard_list


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model's parameters."""
    dtype = None
    for param in model.parameters():
        if dtype is None:
            dtype = param.dtype
        else:
            assert (
                dtype == param.dtype
            ), f"Model has mixed dtypes {dtype} and {param.dtype}"
    assert dtype is not None, "Model has no parameters"
    return dtype


@dataclass
class BenchmarkStats:
    avg_step_time: float
    std_step_time: float
    avg_comm_time: float
    std_comm_time: float
    pct_comm_time: float


def compute_stats(
    rank: int,
    bench_step_times: list[float],
    bench_comm_times: list[float],
    print_basic_stats: bool = True,
    compute_extra_stats: bool = False,
    print_extra_stats: bool = True,
    warmup_step_times: list[float] | None = None,
    warmup_comm_times: list[float] | None = None,
    model: torch.nn.Module | None = None,
    world_size: int | None = None,
    extra_stats: dict | None = None,
) -> tuple[BenchmarkStats, dict]:
    """Compute benchmark statistics."""
    if extra_stats is None:
        extra_stats = {}

    # Per-iteration communication and step time
    avg_step_time = np.mean(bench_step_times)
    std_step_time = np.std(bench_step_times)
    avg_comm_time = np.mean(bench_comm_times)
    std_comm_time = np.std(bench_comm_times)
    pct_comm_time = 100 * np.mean(
        [c / s for c, s in zip(bench_comm_times, bench_step_times)]
    )
    basic_stats = BenchmarkStats(
        avg_step_time, std_step_time, avg_comm_time, std_comm_time, pct_comm_time
    )

    if print_basic_stats:
        print(
            f"=== Rank {rank} Results ===\n"
            f"(Rank {rank}) Iteration times (ms): {bench_step_times}\n"
            f"(Rank {rank}) Gradient communication times (ms): {bench_comm_times}\n"
            f"(Rank {rank}) Iteration time avg (std): "
            f"{avg_step_time:.2f} ({std_step_time:.2f}) ms\n"
            f"(Rank {rank}) Gradient communication time avg (std): "
            f"{avg_comm_time:.2f} ({std_comm_time:.2f}) ms\n"
            f"(Rank {rank}) Fraction of gradient communication time "
            f"in an iteration: {pct_comm_time:.2f}%"
        )
    if not compute_extra_stats:
        return basic_stats, extra_stats

    if model is not None and world_size is not None:
        dtype_size_bytes = get_model_dtype(model).itemsize
        total_allreduce_numel = 0
        for param in model.parameters():
            total_allreduce_numel += param.grad.data.numel()
        total_grad_bytes = total_allreduce_numel * dtype_size_bytes
        total_comm_vol_bytes = total_grad_bytes * 2 * (world_size - 1) / world_size
        extra_stats["total_communication_volume_in_bytes"] = total_comm_vol_bytes

        # Communication throughput in bytes per second
        bench_comm_tputs_bps = [
            total_comm_vol_bytes / (t / 1000) for t in bench_comm_times
        ]
        GB_SIZE = 1024**3
        avg_comm_tput_gbps = np.mean(bench_comm_tputs_bps) / GB_SIZE
        std_comm_tput_gbps = np.std(bench_comm_tputs_bps) / GB_SIZE
        extra_stats["communication_throughputs_in_bytes_per_second"] = (
            bench_comm_tputs_bps
        )
        extra_stats["avg_communication_throughput_in_gbps"] = avg_comm_tput_gbps
        extra_stats["std_communication_throughput_in_gbps"] = std_comm_tput_gbps

    if warmup_step_times is not None:
        extra_stats["warmup_iteration_times_in_ms"] = warmup_step_times
    if warmup_comm_times is not None:
        extra_stats["warmup_communication_times_in_ms"] = warmup_comm_times

    if print_extra_stats:
        extra_stats_str = json.dumps(extra_stats, indent=4)
        print(f"=== Rank {rank} Extra Stats ===\n{extra_stats_str}")

    return basic_stats, extra_stats


def run_naive_ddp_worker(
    rank: int,
    world_size: int,
    data: torch.Tensor,
    num_steps: int,
    ckpt_path: str,
    result_queue: mp.Queue,
    verbose: bool,
) -> None:
    """Run one DDP worker process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12888"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = ToyModel()
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint file {ckpt_path} not found.")
    _, target_list = load_ckpt(model, ckpt_path, num_steps)
    model.to(device)
    # No need to sync parameters as they are loaded from checkpoint
    optimizer = AdamW(model.parameters(), **ADAMW_ARGS)

    data_shard, target_shard_list = get_data_targets_shard(
        data, target_list, rank, world_size, device
    )

    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(data_shard)
        target = target_shard_list[step]
        loss = mse_loss(output, target)
        loss.backward()

        # Naively all-reduce each parameter's gradient
        # Parameters are iterated in the same order on all processes
        for param in model.parameters():
            if param.grad is None:
                if verbose:
                    print(
                        f"Rank {rank} found None grad for param "
                        f"with shape {param.shape}, skipping all-reduce"
                    )
                continue
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

        optimizer.step()

    dist.destroy_process_group()

    if rank == 0:
        # Collect and return the model state from rank 0
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            state_dict[name] = torch.clone(param).cpu()
        result_queue.put(state_dict)


def run_baseline(
    data: torch.Tensor, num_steps: int, ckpt_path: str
) -> dict[str, torch.Tensor]:
    """Run single-process baseline for comparison."""
    model = ToyModel()
    if os.path.exists(ckpt_path):
        _, target_list = load_ckpt(model, ckpt_path, num_steps)
    else:
        with torch.no_grad():
            example_output = model(data)
        target_list = [torch.randn_like(example_output) for _ in range(num_steps)]
        save_ckpt(data, target_list, model, ckpt_path)

    model = model.to("cpu")
    data = data.to("cpu")
    target_list = [target.to("cpu") for target in target_list]

    optimizer = AdamW(model.parameters(), **ADAMW_ARGS)
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(data)
        target = target_list[step]
        loss = mse_loss(output, target)
        loss.backward()
        optimizer.step()
    return model.state_dict()


def verify_naive_ddp(ckpt_path: str, verbose: bool) -> None:
    """Benchmark and verify naive DDP."""
    world_size = 2
    num_steps = 5
    num_gpus = torch.cuda.device_count()
    if num_gpus < world_size:
        raise RuntimeError(
            f"Not enough GPUs to run naive DDP: "
            f"required {world_size}, available {num_gpus}"
        )

    set_seed(SEED)
    data = torch.randn(10, 10)
    batch_size = data.shape[0]
    assert batch_size % world_size == 0

    # Run baseline
    no_ddp_state = run_baseline(data, num_steps, ckpt_path)

    # Set up multiprocessing for DDP
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    mp.spawn(
        run_naive_ddp_worker,
        args=(
            world_size,
            data,
            num_steps,
            ckpt_path,
            result_queue,
            verbose,
        ),
        nprocs=world_size,
        join=True,
    )

    # Get model state from DDP
    ddp_state = result_queue.get()

    assert len(no_ddp_state) > 0, "model state from baseline is empty"
    for name in no_ddp_state:
        assert torch.allclose(no_ddp_state[name], ddp_state[name], atol=1e-6)
    print("Naive DDP matches baseline!")


def run_transformer_worker(
    rank: int,
    world_size: int,
    global_input_ids: torch.Tensor,
    global_target_ids: torch.Tensor,
    context_length: int,
    rope_theta: float,
    warmup_steps: int,
    benchmark_steps: int,
    log_dir: str | None,
    verbose: bool,
) -> None:
    """Run one transformer DDP worker process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12888"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    model = Transformer(
        context_length=context_length,
        theta=rope_theta,
        device=device,
        **TRANSFORMER_ARGS,
    ).to(device)

    # Sync parameters: broadcast from rank 0 to all other ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)
    if verbose and rank == 0:
        param_count = sum(1 for _ in model.parameters())
        buffer_count = sum(1 for _ in model.buffers())
        broadcast_count = param_count + buffer_count
        print(
            f"{broadcast_count} broadcasts to sync "
            f"{param_count} params and {buffer_count} buffers"
        )

    optimizer = AdamW(model.parameters(), **ADAMW_ARGS)

    local_batch_size = global_input_ids.shape[0] // world_size
    input_ids_shard = global_input_ids[
        rank * local_batch_size : (rank + 1) * local_batch_size
    ].to(device)
    target_ids_shard = global_target_ids[
        rank * local_batch_size : (rank + 1) * local_batch_size
    ].to(device)

    step_times = []
    comm_times = []
    for _ in range(warmup_steps + benchmark_steps):
        step_start_ev = torch.cuda.Event(enable_timing=True)
        step_end_ev = torch.cuda.Event(enable_timing=True)
        comm_start_ev = torch.cuda.Event(enable_timing=True)
        comm_end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        step_start_ev.record()
        # Forward and backward pass
        optimizer.zero_grad()
        output = model(input_ids_shard)
        loss = cross_entropy_loss(output, target_ids_shard)
        loss.backward()

        comm_start_ev.record()
        # Naively all-reduce each parameter's gradient
        # Parameters are iterated in the same order on all processes
        for param in model.parameters():
            assert param.grad is not None
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        comm_end_ev.record()

        optimizer.step()
        step_end_ev.record()
        step_end_ev.synchronize()
        torch.cuda.synchronize()

        step_time = step_start_ev.elapsed_time(step_end_ev)
        comm_time = comm_start_ev.elapsed_time(comm_end_ev)
        step_times.append(step_time)  # in milliseconds
        comm_times.append(comm_time)

    dist.destroy_process_group()

    stats, _ = compute_stats(
        rank,
        step_times[warmup_steps:],
        comm_times[warmup_steps:],
        print_basic_stats=True,
        compute_extra_stats=verbose,
        print_extra_stats=verbose,
        warmup_step_times=step_times[:warmup_steps],
        warmup_comm_times=comm_times[:warmup_steps],
        model=model,
        world_size=world_size,
    )

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"rank{rank}.csv"), "w") as f:
            f.write("iter,step_time(ms),comm_time(ms)\n")

            for i in range(warmup_steps):
                f.write(f"warmup{i},{step_times[i]},{comm_times[i]}\n")
            for i in range(benchmark_steps):
                f.write(
                    f"bench{i},{step_times[warmup_steps + i]},"
                    f"{comm_times[warmup_steps + i]}\n"
                )

            f.write(
                f"avg,{stats.avg_step_time},{stats.avg_comm_time}\n"
                f"std,{stats.std_step_time},{stats.std_comm_time}\n"
            )


def time_naive_ddp(
    world_size: int,
    global_batch_size: int,
    context_length: int,
    rope_theta: float,
    warmup_steps: int,
    benchmark_steps: int,
    log_dir: str | None,
    verbose: bool,
):
    """Timing benchmark for naive DDP with transformer model."""
    set_seed(SEED)
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} not divisible by world size {world_size}"
        )

    # Use same batch of random data in all iterations
    global_seqs = torch.randint(
        low=0,
        high=TRANSFORMER_ARGS["vocab_size"],
        size=(global_batch_size, context_length + 1),
    )
    global_input_ids = global_seqs[..., :context_length]
    global_target_ids = global_seqs[..., 1 : context_length + 1]

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        run_transformer_worker,
        args=(
            world_size,
            global_input_ids,
            global_target_ids,
            context_length,
            rope_theta,
            warmup_steps,
            benchmark_steps,
            log_dir,
            verbose,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["toy", "transformer"],
        default="toy",
        help="Verify with toy model or benchmark with transformer model.",
    )
    parser.add_argument("--verbose", action="store_true")

    # Toy model arguments
    parser.add_argument(
        "--toy_ckpt_path",
        type=str,
        default="toy_model_ckpt.pt",
        help="Path to checkpoint file for toy model.",
    )

    # Transformer model arguments
    parser.add_argument(
        "--transformer_world_size",
        type=int,
        default=WORLD_SIZE,
        help="Number of processes / GPUs for transformer benchmark.",
    )
    parser.add_argument(
        "--transformer_global_batch_size",
        type=int,
        default=LOCAL_BATCH_SIZE * WORLD_SIZE,
        help="Global batch size for transformer benchmark.",
    )
    parser.add_argument(
        "--transformer_context_length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Context length for transformer benchmark.",
    )
    parser.add_argument(
        "--transformer_rope_theta",
        type=float,
        default=10000,
        help="RoPE theta for transformer benchmark.",
    )
    parser.add_argument(
        "--transformer_warmup_steps",
        type=int,
        default=2,
        help="Number of warmup steps for transformer benchmark.",
    )
    parser.add_argument(
        "--transformer_benchmark_steps",
        type=int,
        default=5,
        help="Number of benchmark steps for transformer benchmark.",
    )
    parser.add_argument(
        "--transformer_log_dir",
        type=str,
        default=None,
        help="Log directory to save results for transformer benchmark.",
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"Args: {args}")

    if args.model == "toy":
        verify_naive_ddp(args.toy_ckpt_path, args.verbose)
    elif args.model == "transformer":
        time_naive_ddp(
            args.transformer_world_size,
            args.transformer_global_batch_size,
            args.transformer_context_length,
            args.transformer_rope_theta,
            args.transformer_warmup_steps,
            args.transformer_benchmark_steps,
            args.transformer_log_dir,
            args.verbose,
        )
