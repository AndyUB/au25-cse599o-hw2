import torch
import torch.distributed as dist

SRC_RANK = 0


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module
        self.world_size = dist.get_world_size()
        self.handles_and_grads = []
        self.sync_params_and_bufs()
        self.register_allreduce_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle, grad in self.handles_and_grads:
            handle.wait()
            grad /= self.world_size
        self.handles_and_grads.clear()

    def sync_params_and_bufs(self) -> None:
        for param in self.module.parameters():
            dist.broadcast(param.data, src=SRC_RANK)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=SRC_RANK)

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
