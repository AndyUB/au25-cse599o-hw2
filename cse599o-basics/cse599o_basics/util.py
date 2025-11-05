import math
import os
import typing
import numpy as np
import torch


def cross_entropy_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute the cross-entropy loss between logits and target token IDs.

    Args:
        logits (torch.Tensor): Logits tensor of shape
            (..., seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (..., seq_len).

    Returns:
        torch.Tensor: Scalar tensor representing the average cross-entropy
            loss over the batch.
    """
    if target_ids.dtype != torch.long:
        target_ids = target_ids.long()

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_logits
    exp = torch.exp(shifted)
    sum_exp = torch.sum(exp, dim=-1, keepdim=True)
    unshifted = max_logits + torch.log(sum_exp)
    target_logits = torch.gather(logits, dim=-1, index=target_ids.unsqueeze(-1))
    neg_log_likelihood = (unshifted - target_logits).squeeze(-1)
    return torch.mean(neg_log_likelihood)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        """Initialize the AdamW optimizer.

        Args:
            params: Parameters to optimize.
            lr (float): Learning rate.
            betas ((float, float)): beta1 and beta2 coefficients.
            eps (float): Small constant for numerical stability.
            weight_decay (float): Weight decay coefficient.
        """

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                p: torch.Tensor
                g: torch.Tensor = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]
                state["t"] += 1
                t = state["t"]

                m.mul_(beta1).add_((1 - beta1) * g)
                v.mul_(beta2).add_((1 - beta2) * g * g)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.sub_(alpha_t * m / (torch.sqrt(v) + eps))
                if weight_decay != 0:
                    p.sub_(lr * weight_decay * p)

        return loss


def lr_cosine_schedule(
    iter: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_iters: int
) -> float:
    """
    Compute the learning rate at a given iteration using a cosine annealing
    schedule with warmup.

    Args:
        iter (int): Current iteration number.
        lr_max (float): Maximum learning rate.
        lr_min (float): Minimum learning rate.
        warmup_iters (int): Number of warmup iterations.
        cosine_iters (int): Number of iterations for cosine annealing.

    Returns:
        float: The learning rate at the current iteration.
    """

    if iter < warmup_iters:
        return lr_max * (iter / warmup_iters)
    elif iter > cosine_iters:
        return lr_min
    else:
        progress = (iter - warmup_iters) / (cosine_iters - warmup_iters)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine_decay


def gradient_clipping(
    parameters,
    max_norm: float,
) -> torch.Tensor:
    """
    Clip the gradients of the given parameters to have a maximum norm.

    Args:
        parameters: Iterable of model parameters.
        max_norm (float): Maximum allowed norm of the gradients.
    """

    device = parameters[0].device
    total = torch.zeros((), device=device)
    for p in parameters:
        p: torch.Tensor
        g: torch.Tensor = p.grad
        if g is not None:
            total = total + torch.sum(g.detach().float().to(device) ** 2)
    norm = torch.sqrt(total)

    if norm >= max_norm:
        eps = 1e-6
        clip_coef = max_norm / (norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_coef)


def load_data(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a batch of data for training.

    Args:
        x: Numpy array of token IDs.
        batch_size (int): Number of sequences in the batch.
        context_length (int): Length of each sequence.
        device (str): Device to load the data onto.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - input_ids: Tensor of shape (batch_size, context_length).
            - target_ids: Tensor of shape (batch_size, context_length).
    """

    num_tokens = x.shape[0]
    device = torch.device(device)
    batch_starts = np.random.randint(0, num_tokens - context_length, size=batch_size)
    input_ids = (
        torch.stack(
            [
                torch.from_numpy(x[start : start + context_length].copy())
                for start in batch_starts
            ]
        )
        .long()
        .to(device)
    )
    target_ids = (
        torch.stack(
            [
                torch.from_numpy(x[start + 1 : start + context_length + 1].copy())
                for start in batch_starts
            ]
        )
        .long()
        .to(device)
    )
    return input_ids, target_ids


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    """
    Dump all state from model, optimizer, and iteration into out.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current iteration number.
        out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): File path
            or file-like object to save the checkpoint to.
    """
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Load a checkpoint from src and recover model and optimizer states.

    Args:
        src (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): File path
            or file-like object to load the checkpoint from.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int: The iteration number stored in the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
