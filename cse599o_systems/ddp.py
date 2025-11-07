from dataclasses import dataclass
import os
import torch
import torch.distributed as dist

SRC_RANK = 0


def sync_params_and_bufs(module: torch.nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=SRC_RANK)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=SRC_RANK)


def get_model_dtype(params: list[torch.nn.Parameter]) -> torch.dtype:
    dtype = None
    for param in params:
        if dtype is None:
            dtype = param.dtype
        else:
            assert dtype == param.dtype
    assert dtype is not None
    return dtype


def get_model_device(params: list[torch.nn.Parameter]) -> torch.device:
    device = None
    for param in params:
        if device is None:
            device = param.device
        else:
            assert device == param.device
    assert device is not None
    return device


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module
        self.world_size = dist.get_world_size()
        self.handles_and_grads = []
        sync_params_and_bufs(self.module)
        self.register_allreduce_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle, grad in self.handles_and_grads:
            handle.wait()
            grad /= self.world_size
        self.handles_and_grads.clear()

    def register_allreduce_hooks(self) -> None:
        def allreduce_hook(param: torch.Tensor) -> None:
            if param.grad is None:
                print(f"Warning: Grad is None for param with shape {param.shape}")
                return

            handle = dist.all_reduce(param.grad, async_op=True)
            self.handles_and_grads.append((handle, param.grad))

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(allreduce_hook)


@dataclass
class BucketEntry:
    param: torch.nn.Parameter
    offset: int
    numel: int


@dataclass
class Bucket:
    flat: torch.Tensor | None
    entries: list[BucketEntry]
    ready_count: int
    handle: dist.Work | None


@dataclass
class ParamStats:
    shape: torch.Size
    numel: int


@dataclass
class BucketStats:
    num_params: int
    bucket_size_bytes: int
    bucket_numel: int
    bucket_dtype: torch.dtype
    param_stats: list[ParamStats]

    def to_dict(self) -> dict:
        return {
            "num_params": self.num_params,
            "bucket_size_bytes": self.bucket_size_bytes,
            "bucket_numel": self.bucket_numel,
            "bucket_dtype": str(self.bucket_dtype),
            "param_stats": [
                {
                    "shape": list(ps.shape),
                    "numel": ps.numel,
                }
                for ps in self.param_stats
            ],
        }


class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()

        if bucket_size_mb <= 0:
            raise ValueError("bucket_size_mb must be positive")
        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.optimize_singleton_buckets = (
            os.getenv("DDP_OPTIMIZE_SINGLETON_BUCKETS", "0") == "1"
        )
        if self.optimize_singleton_buckets:
            print("[Info] Singleton bucket optimization is enabled.")

        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()

        sync_params_and_bufs(self.module)
        self.params = list(p for p in module.parameters() if p.requires_grad)
        self.dtype = get_model_dtype(self.params)
        self.device = get_model_device(self.params)

        self.buckets: list[Bucket] = []
        self.build_buckets()
        self.register_bucket_allreduce_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for bucket in self.buckets:
            is_singleton_to_optimize = (
                len(bucket.entries) == 1 and self.optimize_singleton_buckets
            )

            if bucket.handle is not None:
                bucket.handle.wait()
                bucket.handle = None

                if is_singleton_to_optimize:
                    grad_to_div = bucket.entries[0].param.grad
                else:
                    grad_to_div = bucket.flat
                grad_to_div /= self.world_size

            if not is_singleton_to_optimize:
                for entry in bucket.entries:
                    start = entry.offset
                    end = entry.offset + entry.numel
                    flat_grad = bucket.flat[start:end]
                    grad = entry.param.grad
                    grad.copy_(flat_grad.view_as(grad))

            bucket.ready_count = 0

    def build_buckets(self) -> None:
        bucket_size_bytes = int(self.bucket_size_mb * 1024 * 1024)

        cur_entries: list[BucketEntry] = []
        cur_size_bytes = 0
        cur_numel = 0

        if self.optimize_singleton_buckets:
            add_bucket_fn = add_bucket_optimized
        else:
            add_bucket_fn = add_bucket

        for param in reversed(self.params):
            param_numel = param.numel()
            param_bytes = param_numel * param.element_size()

            if cur_size_bytes + param_bytes > bucket_size_bytes and cur_entries:
                add_bucket_fn(self, cur_numel, cur_entries)
                cur_entries = []
                cur_size_bytes = 0
                cur_numel = 0

            entry = BucketEntry(
                param=param,
                offset=cur_numel,
                numel=param_numel,
            )
            cur_entries.append(entry)
            cur_size_bytes += param_bytes
            cur_numel += param_numel

        if cur_entries:
            add_bucket_fn(self, cur_numel, cur_entries)

    def register_bucket_allreduce_hooks(self) -> None:
        param_to_bucket: dict[
            torch.nn.Parameter,
            tuple[Bucket, BucketEntry],
        ] = {}
        for bucket in self.buckets:
            for entry in bucket.entries:
                param_to_bucket[entry.param] = (bucket, entry)

        def hook(param: torch.Tensor) -> None:
            if param.grad is None:
                print(f"Warning: Grad is None for param with shape {param.shape}")
                return

            bucket, entry = param_to_bucket[param]
            start = entry.offset
            end = entry.offset + entry.numel
            bucket.flat[start:end].copy_(param.grad.view(-1))

            bucket.ready_count += 1
            if bucket.ready_count == len(bucket.entries):
                bucket.handle = dist.all_reduce(bucket.flat, async_op=True)

        def optimized_hook(param: torch.Tensor) -> None:
            if param.grad is None:
                print(f"Warning: Grad is None for param with shape {param.shape}")
                return

            bucket, entry = param_to_bucket[param]
            is_singleton = len(bucket.entries) == 1

            if is_singleton:
                handle = dist.all_reduce(param.grad, async_op=True)
                bucket.handle = handle
            else:
                start = entry.offset
                end = entry.offset + entry.numel
                bucket.flat[start:end].copy_(param.grad.view(-1))

                bucket.ready_count += 1
                if bucket.ready_count == len(bucket.entries):
                    bucket.handle = dist.all_reduce(bucket.flat, async_op=True)

        hook_fn = optimized_hook if self.optimize_singleton_buckets else hook
        for param in self.params:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook_fn)

    def get_bucket_stats(self) -> list[BucketStats]:
        stats = []
        for bucket in self.buckets:
            num_params = len(bucket.entries)
            if bucket.flat is None:
                assert self.optimize_singleton_buckets
                bucket_numel = bucket.entries[0].numel
                bucket_dtype = bucket.entries[0].param.dtype
            else:
                bucket_numel = bucket.flat.numel()
                bucket_dtype = bucket.flat.dtype

            param_stats = []
            for entry in bucket.entries:
                param_stats.append(
                    ParamStats(
                        entry.param.shape,
                        entry.numel,
                    )
                )

            stats.append(
                BucketStats(
                    num_params,
                    bucket_numel * bucket_dtype.itemsize,
                    bucket_numel,
                    bucket_dtype,
                    param_stats,
                )
            )
        return stats


def add_bucket(
    ddp_module: DDPBucketed,
    bucket_numel: int,
    bucket_entries: list[BucketEntry],
) -> None:
    flat = torch.empty(
        bucket_numel,
        dtype=ddp_module.dtype,
        device=ddp_module.device,
    )
    ddp_module.buckets.append(
        Bucket(
            flat=flat,
            entries=bucket_entries,
            ready_count=0,
            handle=None,
        )
    )


def add_singleton_bucket(
    ddp_module: DDPBucketed,
    param: torch.nn.Parameter,
) -> None:
    entry = BucketEntry(
        param=param,
        offset=0,
        numel=param.numel(),
    )
    ddp_module.buckets.append(
        Bucket(
            flat=None,
            entries=[entry],
            ready_count=0,
            handle=None,
        )
    )


def add_bucket_optimized(
    ddp_module: DDPBucketed,
    bucket_numel: int,
    bucket_entries: list[BucketEntry],
) -> None:
    if len(bucket_entries) == 1:
        add_singleton_bucket(ddp_module, bucket_entries[0].param)
    else:
        add_bucket(ddp_module, bucket_numel, bucket_entries)
