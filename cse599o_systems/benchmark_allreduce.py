import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import numpy as np
from argparse import ArgumentParser


def setup(rank: int, world_size: int, use_gloo: bool = False) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12888"
    if use_gloo:
        # Initialize Gloo for CPU-based distributed training
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # Initialize NCCL for GPU-based distributed training
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup() -> None:
    dist.destroy_process_group()


def allreduce_iter(
    data: torch.Tensor,
    use_gloo: bool,
) -> float:
    if use_gloo:
        start = time.perf_counter()
        dist.all_reduce(data)
        end = time.perf_counter()
        elapse = (end - start) * 1000  # Convert to milliseconds
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start.record()
        dist.all_reduce(data)
        end.record()

        end.synchronize()
        elapse = start.elapsed_time(end)  # Elapsed time in milliseconds

    return elapse


def allreduce_main(
    rank: int,
    world_size: int,
    tensor_size_mb: int,
    warmup_iters: int,
    benchmark_iters: int,
    use_gloo: bool,
    output_dir: str,
) -> None:
    setup(rank, world_size, use_gloo)
    device = torch.device("cpu" if use_gloo else "cuda")
    tensor_size = tensor_size_mb * 1024 * 1024 // 4  # Number of float32 elements
    data = torch.full((tensor_size,), 0.1, device=device, dtype=torch.float32)

    for _ in range(warmup_iters):
        allreduce_iter(data, use_gloo)

    elapses: list[float] = []
    for _ in range(benchmark_iters):
        elapse = allreduce_iter(data, use_gloo)
        elapses.append(elapse)

    output: list[list[float]] = [None for _ in range(world_size)]
    dist.all_gather_object(output, elapses)

    cleanup()

    if rank == 0:
        filename = (
            f"allreduce_{tensor_size_mb}mb_"
            f"{world_size}{'cpu' if use_gloo else 'gpu'}"
        )
        jsonpath = os.path.join(output_dir, f"{filename}.json")

        os.makedirs(output_dir, exist_ok=True)
        with open(jsonpath, "w") as f:
            json.dump(
                {
                    "world_size": world_size,
                    "tensor_size_mb": tensor_size_mb,
                    "use_gloo": use_gloo,
                    "warmup_iters": warmup_iters,
                    "benchmark_iters": benchmark_iters,
                    "output": output,
                },
                f,
                indent=4,
            )

        csvpath = os.path.join(output_dir, f"{filename}.csv")
        save_stats_to_csv(csvpath, output, tensor_size_mb)


def save_stats_to_csv(
    csvpath: str,
    latencies: list[list[float]],
    tensor_size_mb: int,
) -> None:
    latencies_per_iter = []
    for iter_idx in range(len(latencies[0])):
        iter_latencies = [latencies[rank][iter_idx] for rank in range(len(latencies))]
        latencies_per_iter.append(iter_latencies)

    world_size = len(latencies)
    ring_based_size_per_gpu_mb = 2 * tensor_size_mb * (world_size - 1) / world_size
    ring_based_size_per_gpu_bytes = ring_based_size_per_gpu_mb * 1024 * 1024

    avg_times = []
    max_times = []
    avg_tputs = []
    min_tputs = []
    with open(csvpath, "w") as f:
        f.write(
            "iter,avg_time,std_time,max_time,max_min_diff_time,"
            "avg_tput_per_gpu,min_tput_per_gpu\n"
        )

        for i, elapses in enumerate(latencies_per_iter):
            avg_time = np.mean(elapses)
            max_time = max(elapses)
            std_time = np.std(elapses)
            diff_time = max_time - min(elapses)
            avg_tput_per_gpu = (
                ring_based_size_per_gpu_bytes / avg_time if avg_time != 0 else 0
            )
            min_tput_per_gpu = (
                ring_based_size_per_gpu_bytes / max_time if max_time != 0 else 0
            )

            avg_times.append(avg_time)
            max_times.append(max_time)
            avg_tputs.append(avg_tput_per_gpu)
            min_tputs.append(min_tput_per_gpu)
            f.write(
                f"{i},{avg_time},{std_time},{max_time},{diff_time},"
                f"{avg_tput_per_gpu},{min_tput_per_gpu}\n"
            )

        avg_of_avg_time = np.mean(avg_times)
        avg_of_max_time = np.mean(max_times)
        avg_of_avg_tput = np.mean(avg_tputs)
        avg_of_min_tput = np.mean(min_tputs)
        f.write(
            f"avg,{avg_of_avg_time},N/A,{avg_of_max_time},N/A,"
            f"{avg_of_avg_tput},{avg_of_min_tput}\n"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="Number of processes to launch",
    )
    parser.add_argument(
        "--tensor_size",
        type=int,
        required=True,
        help="Size of a tensor to be reduced in MB",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=5,
        help="Number of warm-up iterations",
    )
    parser.add_argument(
        "--benchmark_iters",
        type=int,
        default=10,
        help="Number of benchmarking iterations",
    )
    parser.add_argument(
        "--use_gloo",
        action="store_true",
        help="Use Gloo backend instead of NCCL",
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")

    world_size = args.world_size
    if world_size < 2:
        raise ValueError("world_size must be at least 2 for all-reduce operation")
    gpu_count = torch.cuda.device_count()
    if not args.use_gloo and world_size > gpu_count:
        raise ValueError(
            f"world_size {world_size} exceeds available GPU count {gpu_count}"
        )

    mp.spawn(
        allreduce_main,
        args=(
            world_size,
            args.tensor_size,
            args.warmup_iters,
            args.benchmark_iters,
            args.use_gloo,
            args.output_dir,
        ),
        nprocs=world_size,
        join=True,
    )
