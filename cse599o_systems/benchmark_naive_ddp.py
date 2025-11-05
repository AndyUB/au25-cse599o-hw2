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
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cse599o_basics.util import AdamW
from tests.common import ToyModel

SEED = 599
ADAMW_PARAMS = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
}


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


def run_naive_ddp_worker(
    rank: int,
    world_size: int,
    data: torch.Tensor,
    num_steps: int,
    ckpt_path: str,
    result_queue: mp.Queue,
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
    optimizer = AdamW(model.parameters(), **ADAMW_PARAMS)

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

    optimizer = AdamW(model.parameters(), **ADAMW_PARAMS)
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(data)
        target = target_list[step]
        loss = mse_loss(output, target)
        loss.backward()
        optimizer.step()
    return model.state_dict()


def verify_naive_ddp(ckpt_path: str) -> None:
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
        args=(world_size, data, num_steps, ckpt_path, result_queue),
        nprocs=world_size,
        join=True,
    )

    # Get model state from DDP
    ddp_state = result_queue.get()

    assert len(no_ddp_state) > 0, "model state from baseline is empty"
    for name in no_ddp_state:
        assert torch.allclose(no_ddp_state[name], ddp_state[name], atol=1e-6)
    print("Naive DDP matches baseline!")


def time_naive_ddp():
    """Timing benchmark for naive DDP with transformer model."""
    # TODO
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["toy", "transformer"], default="toy")
    parser.add_argument("--toy_ckpt_path", type=str, default="toy_model_ckpt.pt")
    args = parser.parse_args()

    if args.model == "toy":
        verify_naive_ddp(args.toy_ckpt_path)
    elif args.model == "transformer":
        time_naive_ddp()
