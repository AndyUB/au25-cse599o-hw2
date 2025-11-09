# sharding_optimizer.py
# -------------------------------------------------------------
# CSE 599O:
#
# Implement optimizer state sharding for distributed training.
#
# -------------------------------------------------------------
import os
from typing import Any, Callable, Optional, Type
import torch
import torch.distributed as dist
import argparse
import torch.multiprocessing as mp
from torch.optim.optimizer import ParamsT
from multiprocessing import Manager
from timeit import default_timer as timer


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


# Add any necessary helper functions here.


# You can change the function and variable names as needed.
def run_distributed_training(
    rank, world_size, num_trials, num_warmup_trials, result_queue
):
    # Setup distributed environment
    # TODO

    # Construct model
    # TODO

    # Construct random input data
    # TODO: Create input data

    # Construct optimizer
    # You can use the SharedOptimizer here
    # TODO

    # Training loop
    # Warm up
    # TODO
    # Benchmark
    # TODO

    if rank == 0:
        # Collect and return the timing results
        pass


if __name__ == "__main__":
    # Set up distributed training parameters
    # Collect results and print timing summary
    # TODO
    pass
