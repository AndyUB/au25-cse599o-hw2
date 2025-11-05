import einx
import torch


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a linear transformation module.

        Args:
            in_features (int): Final dimension of the input
            out_features (int): Final dimension of the output
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        sigma_square = 2 / (in_features + out_features)
        sigma = sigma_square**0.5
        torch.nn.init.trunc_normal_(
            self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        output = einx.dot("... in, out in -> ... out", x, self.weight)
        return output


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up the embedding vectors for the given token IDs."""
        return self.weight[token_ids.long()]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float = 1e-5): Epsilon value for numerical stability
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (... d_model)
        and return a tensor of the same shape.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Normalized tensor
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        ms = torch.mean(torch.pow(x, 2), dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        result = (x / rms) * self.weight

        return result.to(in_dtype)


class SiLU(torch.nn.Module):
    def __init__(self):
        """
        Construct the SiLU activation module.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SiLU activation function element-wise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape

        Returns:
            torch.Tensor: Output tensor with the SiLU activation applied element-wise
        """
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct the SwiGLU feed-forward network.

        Args:
            d_model (int): Hidden dimension of the model
            d_ff (int | None = None): Dimension of the feed-forward layer. If None,
                it will be set to 8/3 * d_model rounded to the nearest multiple of
                64, and at least 64.
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """

        super().__init__()
        if d_ff is None:
            d_ff_unrounded = 8 * d_model / 3
            # round d_ff to the nearest multiple of 64, at least 64
            d_ff = 64 * max(1, round(d_ff_unrounded / 64))
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed input tensor of shape (..., d_model) through the SwiGLU
        feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model)
        """
        pre_act = self.w1(x)
        act = self.silu(pre_act)
        gate = self.w3(x)
        return self.w2(act * gate)


class RoPE(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta (float): Theta value for RoPE.
            d_k (int): Dimension of query/key vectors (should be even).
            max_seq_len (int): Maximum sequence length that will be inputted.
            device (torch.device | None = None): Device to store the buffers on.
        """

        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k_range = torch.arange(1, d_k // 2 + 1, device=device)  # (d_k/2,)
        exp_range = (2 * k_range - 2) / d_k  # (d_k/2,)
        inv_powers = 1.0 / (theta**exp_range)  # (d_k/2,)
        positions = torch.arange(max_seq_len, device=device)  # (max_seq_len,)
        angles = torch.outer(positions, inv_powers)  # (max_seq_len, d_k/2)
        cos = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # (max_seq_len, d_k/2)
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to an input tensor of shape (..., seq_len, d_k) and
        return a tensor of the same shape.

        Notes:
        - Accept x with an arbitrary number of batch dimensions.
        - token_positions has shape (..., seq_len) and gives absolute
        positions per token along the sequence dimension.
        - Use token_positions to slice (precomputed) cos/sin tensors
        along the sequence dimension.
        """

        *batch_dims, seq_len, d_k = x.shape
        if d_k != self.d_k:
            raise ValueError(f"Expected last dimension to be {self.d_k}, got {d_k}")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        cos = self.cos[:seq_len]  # (seq_len, d_k/2)
        sin = self.sin[:seq_len]  # (seq_len, d_k/2)
        cos = cos[token_positions]  # (..., seq_len, d_k/2)
        sin = sin[token_positions]  # (..., seq_len, d_k/2)

        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)
        result_even = x_even * cos - x_odd * sin  # (..., seq_len, d_k/2)
        result_odd = x_even * sin + x_odd * cos  # (..., seq_len, d_k/2)
        result = torch.empty_like(x)  # (..., seq_len, d_k)
        result[..., 0::2] = result_even
        result[..., 1::2] = result_odd
        return result


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int = -1): Dimension along which to compute the softmax.

    Returns:
        torch.Tensor: Tensor with the same shape as input,
            with softmax applied along the specified dimension.
    """

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, ..., seq_len, d_k).
        key (torch.Tensor): Key tensor of shape (batch_size, ..., seq_len, d_k).
        value (torch.Tensor): Value tensor of shape (batch_size, ..., seq_len, d_v).
        mask (torch.Tensor | None = None): Optional mask tensor of shape
            (seq_len, seq_len).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, ..., seq_len, d_v)
            after applying attention.
    """
    d_k = query.size(-1)
    attention_scores: torch.Tensor = einx.dot(
        "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k", query, key
    ) / (d_k**0.5)
    if mask is not None:
        attention_scores.masked_fill_(mask == 0, float("-inf"))
    probs = softmax(attention_scores, dim=-1)
    output = einx.dot(
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v", probs, value
    )
    return output


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10_000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attn_fn=None,
    ):
        """
        Construct the Multi-Head Self-Attention (MHSA) module.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            max_seq_len (int = 2048): Maximum sequence length.
            theta (float = 10_000): Theta value for RoPE.
            device (torch.device | None = None): Device to store the parameters on.
            dtype (torch.dtype | None = None): Data type of the parameters.
            attn_fn (callable | None = None): Attention function to use.
                If None, use the default scaled dot-product attention.
        """

        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(
            theta=theta,
            d_k=self.d_head,
            max_seq_len=max_seq_len,
            device=device,
        )

        self.attn_fn = attn_fn or scaled_dot_product_attention

    def forward(
        self,
        x: torch.Tensor,
        apply_rope: bool = True,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention to an input tensor of shape
        (..., seq_len, d_model) and return a tensor of the same shape.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            apply_rope (bool = True): Whether to apply RoPE. Default is True.
            token_positions (torch.Tensor | None = None): Token positions tensor
                of shape (..., seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """

        *batch_dims, seq_len, d_model = x.shape
        q: torch.Tensor = self.q_proj(x)  # (..., seq_len, d_model)
        k: torch.Tensor = self.k_proj(x)  # (..., seq_len, d_model)
        v: torch.Tensor = self.v_proj(x)  # (..., seq_len, d_model)
        multihead_shape = (*batch_dims, seq_len, self.num_heads, self.d_head)
        q = torch.reshape(q, multihead_shape)
        k = torch.reshape(k, multihead_shape)
        v = torch.reshape(v, multihead_shape)
        q = torch.transpose(q, -2, -3)  # (..., num_heads, seq_len, d_head)
        k = torch.transpose(k, -2, -3)  # (..., num_heads, seq_len, d_head)
        v = torch.transpose(v, -2, -3)  # (..., num_heads, seq_len, d_head)

        if apply_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
            q = self.rope(q, token_positions)  # (..., num_heads, seq_len, d_head)
            k = self.rope(k, token_positions)  # (..., num_heads, seq_len, d_head)

        mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=0,
        )  # (seq_len, seq_len)
        attn_output = self.attn_fn(
            q, k, v, mask=mask
        )  # (..., num_heads, seq_len, d_head)
        attn_output = torch.transpose(
            attn_output, -2, -3
        )  # (..., seq_len, num_heads, d_head)
        attn_output = attn_output.contiguous().view(*batch_dims, seq_len, d_model)
        # (..., seq_len, d_model)

        output = self.output_proj(attn_output)  # (..., seq_len, d_model)
        return output


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        max_seq_len: int = 2048,
        theta: float = 10_000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attn_fn=None,
    ):
        """
        Construct a Transformer block.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int | None = None): Dimension of the feed-forward layer.
            max_seq_len (int = 2048): Maximum sequence length.
            theta (float = 10_000): Theta value for RoPE.
            device (torch.device | None = None): Device to store the parameters on.
            dtype (torch.dtype | None = None): Data type of the parameters.
            attn_fn (callable | None = None): Attention function to use.
                If None, use the default scaled dot-product attention.
        """

        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
            attn_fn=attn_fn,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the Transformer block to an input tensor of shape
        (..., seq_len, d_model) and return a tensor of the same shape.

        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """

        n1 = self.ln1(x)
        attn_output = self.attn(n1)
        y = x + attn_output
        n2 = self.ln2(y)
        z = self.ffn(n2)
        return y + z


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        context_length: int = 2048,
        theta: float = 10_000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        attn_fn=None,
    ):
        """
        Construct a Transformer model.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Context length (sequence length).
            num_layers (int): Number of Transformer blocks.
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int | None = None): Dimension of the feed-forward layer.
            context_length (int = 2048): Maximum sequence length.
            theta (float = 10_000): Theta value for RoPE.
            device (torch.device | None = None): Device to store the parameters on.
            dtype (torch.dtype | None = None): Data type of the parameters.
            attn_fn (callable | None = None): Attention function to use.
                If None, use the default scaled dot-product attention.
        """

        super().__init__()
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                    attn_fn=attn_fn,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply the Transformer model to an input tensor of token IDs with
        shape (batch_size, seq_len) and return logits of shape
        (batch_size, seq_len, vocab_size).

        Args:
            token_ids (torch.Tensor): Input tensor of token IDs
                with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits tensor of shape
                (batch_size, seq_len, vocab_size).
        """

        x = self.token_embeddings(token_ids)  # (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
