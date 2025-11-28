"""State space models."""

from abc import ABC, abstractmethod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))


class BaseSSMBlock(eqx.Module, ABC):
    @abstractmethod
    def forward(self, h: Array, x: Array) -> Array: ...

    @abstractmethod
    def forward_sequence(self, x_seq: Array) -> Array: ...

    @abstractmethod
    def get_a_mat(self, x: Array) -> Array: ...

    @abstractmethod
    def get_b_mat(self, x: Array) -> Array: ...


class SSMBlock(BaseSSMBlock):
    a_mat: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        key_a, key_b = jax.random.split(key)
        self.a_mat = glorot(key_a, (hidden_size, hidden_size))
        self.b_mat = glorot(key_b, (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        return self.a_mat

    def get_b_mat(self, x: Array) -> Array:
        return self.b_mat

    def forward(self, h: Array, x: Array) -> Array:
        """Perform a forward pass.

        Args:
            h: Hidden state of shape (H,).
            x: Input of shape (H,).

        Returns:
            Hidden state of shape (H,).
        """
        a_mat = self.get_a_mat(x)
        b_mat = self.get_b_mat(x)
        h = a_mat @ h + b_mat.T @ x
        return h

    def forward_sequence(self, x_seq: Array) -> Array:
        """Perform a forward pass across time.

        Args:
            x_seq: Input sequence of shape (T, H).

        Returns:
            Hidden state sequence of shape (T, H).
        """

        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        a_mat = self.get_a_mat(x_seq)
        h_0 = jnp.zeros(a_mat.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x_seq)
        return h_seq


class DiagSSMBlock(BaseSSMBlock):
    a_diag: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.a_diag = glorot(keys[0], (hidden_size,))
        self.b_mat = glorot(keys[1], (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        return self.a_diag

    def get_b_mat(self, x: Array) -> Array:
        return self.b_mat

    def forward(self, h: Array, x: Array) -> Array:
        """Perform a forward pass.

        Args:
            h: Hidden state of shape (H,).
            x: Input of shape (H,).

        Returns:
            Hidden state of shape (H,).
        """
        a_diag = self.get_a_mat(x)
        b_mat = self.get_b_mat(x)
        h = a_diag * h + b_mat.T @ x
        return h

    def forward_sequence(self, x_seq: Array, *, use_conv: bool = True, recursive_kernel_calc: bool = False) -> Array:
        """Perform a potentially parallelized forward pass across time.

        Args:
            x_seq: Input sequence of shape (T, H).
            use_conv: Whether to use convolution to compute the sequence.
            recursive_kernel_calc: Whether to use a recursive kernel calculation.

        Returns:
            Hidden state sequence of shape (T, H).
        """
        if use_conv:
            return self._forward_sequence_conv(x_seq, recursive_kernel_calc=recursive_kernel_calc)
        else:
            return self._forward_sequence_scan(x_seq)

    def _get_kernel(self, x_seq: Array, length: int) -> Array:
        """Returns the kernel with time as the final dimension."""
        exponents = jnp.arange(length)
        a_diag = self.get_a_mat(x_seq)
        kernel = jnp.power(a_diag[:, None], exponents)  # (H, T)
        kernel = kernel[:, None, :]  # (H, 1, T)
        return kernel

    def _get_kernel_recursive(self, x_seq: Array, length: int) -> Array:
        """Returns the kernel with time as the final dimension."""
        assert length % 2 == 0, "Length must be even."
        a_diag = self.get_a_mat(x_seq)

        def helper(length: int) -> tuple[Array, Array]:
            """Returns the kernel and the sqrt of the diagonal."""
            if length == 1:
                return jnp.ones_like(a_diag)[:, None], a_diag[:, None]

            half_length = length // 2
            kernel_half, a_half = helper(half_length)
            kernel = jnp.concatenate([kernel_half, a_half * kernel_half], axis=-1)
            return kernel, a_half * a_half

        kernel, a_diag = helper(length)
        return kernel[:, None, :]  # (H, 1, L)

    def _forward_sequence_conv(self, x_seq: Array, *, recursive_kernel_calc: bool = False) -> Array:
        """Convolves x (T, H) across time using the kernel."""
        seq_len, hidden_size = x_seq.shape
        b_mat = self.get_b_mat(x_seq)

        s = b_mat.T @ x_seq.T  # (H, T)
        s_padded = jnp.pad(s, ((0, 0), (seq_len - 1, 0)))[None, :, :]  # (1, H, 2T-1)

        if recursive_kernel_calc:
            kernel = self._get_kernel_recursive(x_seq, seq_len)
        else:
            kernel = self._get_kernel(x_seq, seq_len)

        kernel_flipped = jnp.flip(kernel, axis=-1)  # (H, 1, L)

        conv_out = jax.lax.conv_general_dilated(
            s_padded,
            kernel_flipped,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCT", "OIT", "NCT"),  # convolving over time
            feature_group_count=hidden_size,
        )
        conv_out = conv_out[0].T  # (T, H)
        return conv_out

    def _forward_sequence_scan(self, x_seq: Array) -> Array:
        """Naively forward across time."""

        def step(h: Array, x: Array) -> tuple[Array, Array]:
            h = self.forward(h, x)
            return h, h

        a_diag = self.get_a_mat(x_seq)
        h_0 = jnp.zeros(a_diag.shape[0])
        _, h_seq = jax.lax.scan(step, h_0, x_seq)
        return h_seq


class DiscreteDiagSSMBlock(DiagSSMBlock):
    delta: Array

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        init_delta: float = 1.0,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__(hidden_size, key=key)
        self.delta = jnp.array(init_delta)

        # A positive scale helps reduce the gradient at the start.
        self.a_diag = jax.random.uniform(key, (hidden_size,), minval=-1.0, maxval=0.0) * init_scale

    def get_a_mat(self, x: Array) -> Array:
        """Discretize the diagonal matrix using zero-order hold."""
        a_diag_discrete = jnp.exp(self.a_diag * self.delta)
        return a_diag_discrete

    def get_b_mat(self, x: Array) -> Array:
        """Discretize the input matrix using zero-order hold."""
        delta_a_diag = self.a_diag * self.delta
        exp_a_diag = jnp.exp(delta_a_diag)
        delta_a_inv = 1 / delta_a_diag
        delta_b_mat = self.delta * self.b_mat

        b_discrete = delta_a_inv * (exp_a_diag - 1) * delta_b_mat
        return b_discrete


class SSM(eqx.Module):
    vocab_embedding: eqx.nn.Embedding = eqx.field()
    output_layer: eqx.nn.Linear = eqx.field()
    blocks: tuple[BaseSSMBlock, ...] = eqx.field()
    num_layers: int = eqx.field()
    hidden_size: int = eqx.field()
    skip_connections: bool = eqx.field()

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        block_type: Literal["diagonal", "full_rank"] = "full_rank",
        skip_connections: bool = False,
        discretize: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        vocab_key, s4_key = jax.random.split(key, 2)
        self.vocab_embedding = eqx.nn.Embedding(input_size, hidden_size, key=vocab_key)
        self.output_layer = eqx.nn.Linear(hidden_size, output_size, key=key)

        block_keys = jax.random.split(s4_key, num_layers)

        def get_block(key: PRNGKeyArray) -> BaseSSMBlock:
            match block_type:
                case "diagonal":
                    return (
                        DiscreteDiagSSMBlock(hidden_size, key=key, init_delta=0.1)
                        if discretize
                        else DiagSSMBlock(hidden_size, key=key)
                    )
                case "full_rank":
                    if discretize:
                        raise ValueError("Full rank blocks do not support discretization due to instability.")
                    return SSMBlock(hidden_size, key=key)
                case _:
                    raise ValueError(f"Unknown block type: {block_type}")

        self.blocks = tuple(get_block(block_keys[i]) for i in range(num_layers))
        self.skip_connections = skip_connections
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def __call__(self, hs: list[Array], x: Array) -> tuple[list[Array], Array]:
        new_hs = []
        for i, block in enumerate(self.blocks):
            h = block.forward(hs[i], x)
            new_hs.append(h)
            xh = jax.nn.gelu(h)
            x = xh + x if self.skip_connections else xh
        y = self.output_layer(x)
        return new_hs, y

    def _embed_input(self, x: Array) -> Array:
        """U is the input to the S4 cell."""
        return self.vocab_embedding(x)

    def predict_sequence(self, x_seq: Array) -> Array:
        x_emb = jax.vmap(self._embed_input)(x_seq)
        for block in self.blocks:
            h = block.forward_sequence(x_emb)
            # h = block.naive_forward_sequence(x_emb)
            h = jax.nn.gelu(h)
            x_emb = h + x_emb if self.skip_connections else h
        y = jax.vmap(self.output_layer)(x_emb)
        return y

    def generate_sequence(self, prompt_seq: Array, max_len: int) -> Array:
        hs = [jnp.zeros(self.hidden_size) for _ in range(self.num_layers)]
        prompt_seq_embedded = jax.vmap(self._embed_input)(prompt_seq)

        def encode_step(hs: list[Array], x: Array) -> tuple[list[Array], Array]:
            hs, y = self(hs, x)
            return hs, y

        def decode_step(
            carry: tuple[list[Array], Array, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[list[Array], Array, PRNGKeyArray], Array]:
            hs, last_token, rng = carry
            token_embedded = self._embed_input(last_token)
            hs, y = self(hs, token_embedded)
            token = jax.random.categorical(rng, y)
            rng = jax.random.split(rng)[0]
            return (hs, token, rng), token

        hs, _ = jax.lax.scan(encode_step, hs, prompt_seq_embedded)
        _, sequence = jax.lax.scan(decode_step, (hs, prompt_seq[-1], jax.random.key(0)), None, length=max_len)

        return sequence
