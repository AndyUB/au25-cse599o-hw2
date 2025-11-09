# sharding_optimizer.py
# -------------------------------------------------------------
# CSE 599O:
#
# Implement optimizer state sharding for distributed training.
#
# -------------------------------------------------------------
import argparse
import json
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.optimizer import ParamsT
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Type
from cse599o_basics.util import AdamW, cross_entropy_loss
from cse599o_basics.transformer import Transformer

from cse599o_systems.benchmark_optimized_ddp import dist_setup, dist_cleanup
from cse599o_systems.benchmark_naive_ddp import (
    SEED,
    TRANSFORMER_ARGS,
    ADAMW_ARGS,
    WORLD_SIZE,
    LOCAL_BATCH_SIZE,
    CONTEXT_LENGTH,
    set_seed,
)


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.global_param_count = 0
        self.global_params: list[torch.nn.Parameter] = []

        self.local_optim: Optional[torch.optim.Optimizer] = None
        super().__init__(params, kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        params = param_group["params"]
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        else:
            params = list(params)
        param_group["params"] = params

        local_params: list[torch.nn.Parameter] = []
        for param in params:
            param_id = self.global_param_count
            self.global_param_count += 1
            self.global_params.append(param)

            owner_rank = self.owner_rank_for_param(param_id)
            if owner_rank == self.rank:
                local_params.append(param)

        if local_params:
            self.add_param_group_to_local_optimizer(local_params)
        torch.optim.Optimizer.add_param_group(self, param_group)

    def owner_rank_for_param(self, param_id: int) -> int:
        return param_id % self.world_size

    def add_param_group_to_local_optimizer(
        self,
        params: list[torch.nn.Parameter],
    ) -> None:
        if self.local_optim is None:
            self.local_optim = self.optimizer_cls(params, **self.kwargs)
        else:
            param_group = {"params": params}
            self.local_optim.add_param_group(param_group)

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        **kwargs,
    ) -> Optional[float]:
        loss = None
        if self.local_optim is not None:
            loss = self.local_optim.step(closure=closure, **kwargs)

        for param_id, param in enumerate(self.global_params):
            owner_rank = self.owner_rank_for_param(param_id)
            dist.broadcast(param.data, src=owner_rank)

        return loss


def count_model_bytes(model: torch.nn.Module) -> tuple[int, int]:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_bytes, buffer_bytes


def assert_no_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        assert p.grad is None


def count_grad_bytes(model: torch.nn.Module) -> int:
    grad_bytes = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_bytes += p.grad.numel() * p.grad.element_size()
    return grad_bytes


def count_optimizer_state_bytes(optimizer: AdamW | ShardedOptimizer) -> int:
    if isinstance(optimizer, ShardedOptimizer):
        if optimizer.local_optim is None:
            return 0
        optimizer = optimizer.local_optim

    if not isinstance(optimizer, AdamW):
        raise ValueError(
            "Optimizer must be AdamW or ShardedOptimizer wrapped around AdamW"
        )

    state_bytes = 0
    for state in optimizer.state.values():
        assert len(state) == 3  # t, m, v
        assert "t" in state and "m" in state and "v" in state
        m = state["m"]
        v = state["v"]
        state_bytes += m.numel() * m.element_size()
        state_bytes += v.numel() * v.element_size()
    return state_bytes


@dataclass
class MemoryStats:
    curr_allocated_bytes: int
    curr_reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int


def measure_peak_memory(
    device: torch.device,
    reset_after_measure: bool = True,
) -> MemoryStats:
    torch.cuda.synchronize(device)
    curr_allocated_bytes = torch.cuda.memory_allocated(device)
    curr_reserved_bytes = torch.cuda.memory_reserved(device)
    max_allocated_bytes = torch.cuda.max_memory_allocated(device)
    max_reserved_bytes = torch.cuda.max_memory_reserved(device)

    if reset_after_measure:
        torch.cuda.reset_peak_memory_stats(device)

    return MemoryStats(
        curr_allocated_bytes=curr_allocated_bytes,
        curr_reserved_bytes=curr_reserved_bytes,
        max_allocated_bytes=max_allocated_bytes,
        max_reserved_bytes=max_reserved_bytes,
    )


def assemble_memory_results(
    rank: int,
    profiled_stats: dict[str, MemoryStats],
    computed_bytes: dict[str, int | list[int]],
) -> dict[str, Any]:
    for k, v in computed_bytes.items():
        if isinstance(v, list) and len(v) > 1:
            max_v = max(v)
            min_v = min(v)
            if max_v != min_v:
                print(
                    f"(Rank {rank}) [Warning] Varied number of bytes for {k}: "
                    f"values={v}, max={max_v}, min={min_v}"
                )

    results: dict[str, Any] = {
        "rank": rank,
        "profiled_stats": profiled_stats,
        "computed_bytes": computed_bytes,
    }
    return results


def profile_memory_worker(
    rank: int,
    world_size: int,
    seqs: torch.Tensor,
    output_dir: str,
    result_queue: Optional[mp.Queue] = None,
) -> Optional[dict[str, Any]]:
    """
    Worker process for profiling memory usage. When `world_size`
    is greater than 1, sharded optimizer is used. When `world_size`
    is 1, the optimizer is unsharded.

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
        seqs: Input data tensor.
        output_dir: Directory to save memory profiling outputs.
        result_queue: Multiprocessing queue to collect results.
            The queue should be provided when `world_size` > 1.

    Returns:
        A dictionary containing memory profiling results if
        `world_size` is 1; otherwise, None.
    """

    num_gpus = torch.cuda.device_count()
    if world_size < 1 or world_size > num_gpus:
        raise ValueError(f"Invalid world-size {world_size}, expected 1 to {num_gpus}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank}, expected 0 to {world_size - 1}")

    shard_optim = world_size > 1
    if shard_optim and result_queue is None:
        raise ValueError("Result queue should be provided for distributed training")

    if shard_optim:
        # Setup distributed environment
        dist_setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    memory_base = measure_peak_memory(device)

    # Move data to device
    seqs = seqs.to(device)
    input_ids = seqs[:, :-1]
    target_ids = seqs[:, 1:]
    memory_after_data_move = measure_peak_memory(device)
    data_bytes = seqs.numel() * seqs.element_size()

    # Construct model
    model = Transformer(
        context_length=CONTEXT_LENGTH,
        device=device,
        **TRANSFORMER_ARGS,
    ).to(device)
    memory_after_model_init = measure_peak_memory(device)
    param_bytes, buffer_bytes = count_model_bytes(model)
    assert_no_grads(model)

    if shard_optim:
        # Construct sharded optimizer
        optimizer = ShardedOptimizer(
            model.parameters(),
            AdamW,
            **ADAMW_ARGS,
        )
    else:
        # Construct unsharded optimizer
        optimizer = AdamW(
            model.parameters(),
            **ADAMW_ARGS,
        )
    memory_after_optimizer_init = measure_peak_memory(device)
    optimizer_bytes_after_init = count_optimizer_state_bytes(optimizer)

    # Training: step 1
    # Forward pass
    optimizer.zero_grad()
    logits = model(input_ids)
    memory_after_first_forward = measure_peak_memory(device)

    # Backward pass
    loss = cross_entropy_loss(logits, target_ids)
    loss.backward()
    memory_after_first_backward = measure_peak_memory(device)
    grad_bytes_after_first_backward = count_grad_bytes(model)

    # Optimizer step
    optimizer.step()
    memory_after_first_step = measure_peak_memory(device)
    optimizer_bytes_after_first_step = count_optimizer_state_bytes(optimizer)

    # Training: step 2
    optimizer.zero_grad()
    memory_after_zero_grad = measure_peak_memory(device)
    assert_no_grads(model)

    # Forward pass
    logits = model(input_ids)
    memory_after_second_forward = measure_peak_memory(device)

    # Backward pass
    loss = cross_entropy_loss(logits, target_ids)
    loss.backward()
    memory_after_second_backward = measure_peak_memory(device)
    grad_bytes_after_second_backward = count_grad_bytes(model)

    # Optimizer step
    optimizer.step()
    memory_after_second_step = measure_peak_memory(device)
    optimizer_bytes_after_second_step = count_optimizer_state_bytes(optimizer)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{world_size}gpu_rank{rank}.pickle")
    torch.cuda.memory._dump_snapshot(output_path)
    torch.cuda.memory._record_memory_history(enabled=None)

    profiled_stats = dict(
        memory_base=memory_base,
        memory_after_data_move=memory_after_data_move,
        memory_after_model_init=memory_after_model_init,
        memory_after_optimizer_init=memory_after_optimizer_init,
        memory_after_first_forward=memory_after_first_forward,
        memory_after_first_backward=memory_after_first_backward,
        memory_after_first_step=memory_after_first_step,
        memory_after_zero_grad=memory_after_zero_grad,
        memory_after_second_forward=memory_after_second_forward,
        memory_after_second_backward=memory_after_second_backward,
        memory_after_second_step=memory_after_second_step,
    )
    theoretical_bytes = dict(
        data_bytes=data_bytes,
        param_bytes=param_bytes,
        buffer_bytes=buffer_bytes,
        pre_step_optimizer_bytes=optimizer_bytes_after_init,
        grad_bytes=[
            grad_bytes_after_first_backward,
            grad_bytes_after_second_backward,
        ],
        post_step_optimizer_bytes=[
            optimizer_bytes_after_first_step,
            optimizer_bytes_after_second_step,
        ],
    )
    memory_results = assemble_memory_results(rank, profiled_stats, theoretical_bytes)

    if shard_optim:
        result_queue.put(memory_results)
        dist_cleanup()
        return None
    else:
        return memory_results


def profile_sharding_memory(
    world_size: int,
    data: torch.Tensor,
    output_dir: str,
) -> list[dict[str, Any]]:
    """
    Profile memory usage for sharded optimizer.
    """
    set_seed(SEED)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()
    mp.spawn(
        profile_memory_worker,
        args=(world_size, data, output_dir, result_queue),
        nprocs=world_size,
        join=True,
    )

    global_results = []
    for _ in range(world_size):
        result = result_queue.get()
        global_results.append(result)
    global_results.sort(key=lambda r: r["rank"])

    return global_results


def profile_baseline_memory(
    data: torch.Tensor,
    output_dir: str,
) -> dict[str, Any]:
    """
    Profile memory usage for baseline (unsharded) optimizer.
    """
    set_seed(SEED)

    results = profile_memory_worker(
        rank=0,
        world_size=1,
        seqs=data,
        output_dir=output_dir,
        result_queue=None,
    )
    assert results is not None
    return results


@dataclass
class IterTimeStats:
    total_time_ms: float
    zero_grad_time_ms: float
    fwd_time_ms: float
    loss_time_ms: float
    bwd_time_ms: float
    step_time_ms: float


def assemble_time_results(
    rank: int,
    warmup_iter_times: list[IterTimeStats],
    iter_times: list[IterTimeStats],
) -> dict[str, Any]:
    assert len(iter_times) > 0
    time_stats_dicts = [asdict(t) for t in iter_times]
    time_stats_keys = time_stats_dicts[0].keys()
    aggregate_stats = {
        k: [stat[k] for stat in time_stats_dicts] for k in time_stats_keys
    }

    mean_stats = {k: np.mean(v) for k, v in aggregate_stats.items()}
    std_stats = {k: np.std(v) for k, v in aggregate_stats.items()}
    pct_stats = {}
    total_times = aggregate_stats["total_time_ms"]
    for name, times in aggregate_stats.items():
        if name == "total_time_ms":
            continue
        pct_stats[name] = 100 * np.mean(
            [t / total for t, total in zip(times, total_times)]
        )

    results: dict[str, Any] = {
        "rank": rank,
        "mean": mean_stats,
        "standard_deviation": std_stats,
        "breakdown_in_percentage": pct_stats,
        "details": time_stats_dicts,
        "extra": {
            "warmup_times": [asdict(t) for t in warmup_iter_times],
        },
    }
    return results


def profile_time_worker(
    rank: int,
    world_size: int,
    seqs: torch.Tensor,
    num_trials: int,
    num_warmup_trials: int,
    result_queue: Optional[mp.Queue] = None,
) -> Optional[dict[str, Any]]:
    num_gpus = torch.cuda.device_count()
    if world_size < 1 or world_size > num_gpus:
        raise ValueError(f"Invalid world-size {world_size}, expected 1 to {num_gpus}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank}, expected 0 to {world_size - 1}")

    if num_trials <= 0:
        raise ValueError("Number of trials must be positive")
    if num_warmup_trials < 0:
        raise ValueError("Number of warmup trials cannot be negative")

    shard_optim = world_size > 1
    if shard_optim and result_queue is None:
        raise ValueError("Result queue should be provided for distributed training")

    if shard_optim:
        # Setup distributed environment
        dist_setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    # Move data to device
    seqs = seqs.to(device)
    input_ids = seqs[:, :-1]
    target_ids = seqs[:, 1:]

    # Construct model
    model = Transformer(
        context_length=CONTEXT_LENGTH,
        device=device,
        **TRANSFORMER_ARGS,
    ).to(device)

    # Construct optimizer
    if shard_optim:
        # Construct sharded optimizer
        optimizer = ShardedOptimizer(
            model.parameters(),
            AdamW,
            **ADAMW_ARGS,
        )
    else:
        # Construct unsharded optimizer
        optimizer = AdamW(
            model.parameters(),
            **ADAMW_ARGS,
        )

    # Training loop: warm up, then benchmark
    warmup_iter_times = []
    iter_times = []
    for it in range(num_warmup_trials + num_trials):
        iter_start_ev = torch.cuda.Event(enable_timing=True)
        iter_end_ev = torch.cuda.Event(enable_timing=True)
        fwd_start_ev = torch.cuda.Event(enable_timing=True)
        fwd_end_ev = torch.cuda.Event(enable_timing=True)
        bwd_start_ev = torch.cuda.Event(enable_timing=True)
        bwd_end_ev = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        iter_start_ev.record()
        optimizer.zero_grad()

        fwd_start_ev.record()
        logits = model(input_ids)
        fwd_end_ev.record()

        loss = cross_entropy_loss(logits, target_ids)
        bwd_start_ev.record()
        loss.backward()
        bwd_end_ev.record()

        optimizer.step()
        iter_end_ev.record()
        iter_end_ev.synchronize()
        torch.cuda.synchronize()

        # Times in milliseconds
        iter_time = iter_start_ev.elapsed_time(iter_end_ev)
        zero_grad_time = iter_start_ev.elapsed_time(fwd_start_ev)
        fwd_time = fwd_start_ev.elapsed_time(fwd_end_ev)
        loss_time = fwd_end_ev.elapsed_time(bwd_start_ev)
        bwd_time = bwd_start_ev.elapsed_time(bwd_end_ev)
        step_time = bwd_end_ev.elapsed_time(iter_end_ev)
        time_stats = IterTimeStats(
            total_time_ms=iter_time,
            zero_grad_time_ms=zero_grad_time,
            fwd_time_ms=fwd_time,
            loss_time_ms=loss_time,
            bwd_time_ms=bwd_time,
            step_time_ms=step_time,
        )

        if it >= num_warmup_trials:
            iter_times.append(time_stats)
        else:
            warmup_iter_times.append(time_stats)

    time_results = assemble_time_results(rank, warmup_iter_times, iter_times)
    if shard_optim:
        result_queue.put(time_results)
        dist_cleanup()
        return None
    else:
        return time_results


def profile_sharding_time(
    world_size: int,
    data: torch.Tensor,
    num_trials: int,
    num_warmup_trials: int,
) -> list[dict[str, Any]]:
    """
    Profile time usage for sharded optimizer.
    """
    set_seed(SEED)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()
    mp.spawn(
        profile_time_worker,
        args=(world_size, data, num_trials, num_warmup_trials, result_queue),
        nprocs=world_size,
        join=True,
    )

    global_results = []
    for _ in range(world_size):
        result = result_queue.get()
        global_results.append(result)
    global_results.sort(key=lambda r: r["rank"])

    return global_results


def profile_baseline_time(
    data: torch.Tensor,
    num_trials: int,
    num_warmup_trials: int,
) -> dict[str, Any]:
    """
    Profile time usage for baseline (unsharded) optimizer.
    """
    set_seed(SEED)

    results = profile_time_worker(
        rank=0,
        world_size=1,
        seqs=data,
        num_trials=num_trials,
        num_warmup_trials=num_warmup_trials,
        result_queue=None,
    )
    assert results is not None
    return results


def log_results(title: str, results: dict[str, Any]) -> None:
    print(f"--- {title} ---")
    print(json.dumps(results, indent=4, default=asdict))


if __name__ == "__main__":
    # Set up distributed training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["time", "memory"],
        default="memory",
        help="Profiling mode: time or memory",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=WORLD_SIZE,
        help="Number of distributed processes",
    )
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=LOCAL_BATCH_SIZE,
        help="Local batch size per process",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=CONTEXT_LENGTH,
        help="Context length for the transformer model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Memory profiling arguments
    parser.add_argument(
        "--memory_output_dir",
        type=str,
        default="./sharding_memory_profiles",
        help="Directory to save memory profiling outputs",
    )

    # Time profiling arguments
    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="Number of trials for time profiling",
    )
    parser.add_argument(
        "--num_warmup_trials",
        type=int,
        default=3,
        help="Number of warmup trials for time profiling",
    )
    args = parser.parse_args()

    def log(msg: str) -> None:
        if args.verbose:
            print(msg)

    log(f"Arguments: {args}")

    world_size = args.world_size
    num_gpus = torch.cuda.device_count()
    if world_size < 2 or world_size > num_gpus:
        raise ValueError(f"Invalid world size {world_size}, expected 2 to {num_gpus}")

    # Get input data
    seqs = torch.randint(
        low=0,
        high=TRANSFORMER_ARGS["vocab_size"],
        size=(args.local_batch_size, args.context_length + 1),
    )
    log("[Info] Generated input data.")

    # Run profiling, collect results, and print summary
    if args.mode == "memory":
        output_dir = args.memory_output_dir
        sharding_results = profile_sharding_memory(world_size, seqs, output_dir)
        log("[Info] Completed memory profiling for sharded optimizer.")

        baseline_result = profile_baseline_memory(seqs, output_dir)
        log("[Info] Completed memory profiling for unsharded optimizer.")

        log_results("Memory Profiling Results for Unsharded Optimizer", baseline_result)
        for sharding_result in sharding_results:
            rank = sharding_result["rank"]
            log_results(
                f"Memory Profiling Results for Sharded Optimizer (Rank {rank})",
                sharding_result,
            )
    elif args.mode == "time":
        num_trials = args.num_trials
        num_warmup_trials = args.num_warmup_trials
        sharding_results = profile_sharding_time(
            world_size, seqs, num_trials, num_warmup_trials
        )
        log("[Info] Completed time profiling for sharded optimizer.")

        baseline_result = profile_baseline_time(seqs, num_trials, num_warmup_trials)
        log("[Info] Completed time profiling for unsharded optimizer.")

        log_results("Timing Results for Unsharded Optimizer", baseline_result)
        for sharding_result in sharding_results:
            rank = sharding_result["rank"]
            log_results(
                f"Timing Results for Sharded Optimizer (Rank {rank})", sharding_result
            )
