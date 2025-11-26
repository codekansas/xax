"""Defines components for diffusion models.

This module provides implementations for:
- ODE solvers (Euler, Heun, RK4)
- Gaussian diffusion models
- Consistency models
"""

import math
from abc import ABC, abstractmethod
from typing import Callable, Literal, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

ODESolverType = Literal["euler", "heun", "rk4"]
DiffusionLossFn = Literal["mse", "l1", "pseudo-huber"]
DiffusionPredMode = Literal["pred_x_0", "pred_eps", "pred_v"]
SigmaType = Literal["upper_bound", "lower_bound"]
DiffusionBetaSchedule = Literal["linear", "quad", "warmup", "const", "cosine", "jsd"]


def cast_solver_type(s: str) -> ODESolverType:
    assert s in get_args(ODESolverType), f"Unknown solver {s}"
    return cast(ODESolverType, s)


def cast_diffusion_loss_fn(loss: str) -> DiffusionLossFn:
    assert loss in get_args(DiffusionLossFn), f"Unknown loss function: {loss}"
    return cast(DiffusionLossFn, loss)

def cast_diffusion_pred_mode(pred_mode: str) -> DiffusionPredMode:
    assert pred_mode in get_args(DiffusionPredMode), f"Unknown prediction mode: {pred_mode}"
    return cast(DiffusionPredMode, pred_mode)

def cast_sigma_type(sigma_type: str) -> SigmaType:
    assert sigma_type in get_args(SigmaType), f"Unknown sigma type: {sigma_type}"
    return cast(SigmaType, sigma_type)

def cast_beta_schedule(schedule: str) -> DiffusionBetaSchedule:
    assert schedule in get_args(DiffusionBetaSchedule), f"Unknown schedule type: {schedule}"
    return cast(DiffusionBetaSchedule, schedule)


def append_dims(x: Array, num_dims: int) -> Array:
    """Appends dimensions to an array.

    Args:
        x: The array to append dimensions to.
        num_dims: The number of dimensions to append.

    Returns:
        The array with appended dimensions.
    """
    return x.reshape(x.shape + (1,) * num_dims)


def vanilla_add_fn(a: Array, b: Array, ta: Array, tb: Array) -> Array:
    dt = append_dims(tb - ta, a.ndim - ta.ndim)
    return a + b * dt


def pseudo_huber_loss(pred: Array, target: Array, dim: int = -1, factor: float = 0.00054) -> Array:
    """Computes the pseudo-Huber loss.

    Args:
        pred: The predicted values.
        target: The target values.
        dim: The dimension over which to compute the loss.
        factor: The factor for the pseudo-Huber loss.

    Returns:
        The loss values.
    """
    diff = pred - target
    return factor * jnp.sqrt(1 + (diff / factor) ** 2) - factor


class BaseODESolver(ABC):
    @abstractmethod
    def step(
        self,
        samples: Array,
        t: Array,
        next_t: Array,
        func: Callable[[Array, Array], Array],
        add_fn: Callable[[Array, Array, Array, Array], Array] = vanilla_add_fn,
    ) -> Array:
        """Steps the current state forward in time.

        Args:
            samples: The current samples, with shape ``(N, *)``.
            t: The current time step, with shape ``(N)``.
            next_t: The next time step, with shape ``(N)``.
            func: The function to use to compute the derivative, with signature
                ``(samples, t) -> deriv``.
            add_fn: The addition function to use, which has the signature
                ``(a, b, ta, tb) -> a + b * (tb - ta)``.

        Returns:
            The next sample, with shape ``(N, *)``.
        """

    def __call__(
        self,
        samples: Array,
        t: Array,
        next_t: Array,
        func: Callable[[Array, Array], Array],
        add_fn: Callable[[Array, Array, Array, Array], Array] = vanilla_add_fn,
    ) -> Array:
        return self.step(samples, t, next_t, func, add_fn)


class EulerODESolver(BaseODESolver):
    """The Euler method for solving ODEs."""

    def step(
        self,
        samples: Array,
        t: Array,
        next_t: Array,
        func: Callable[[Array, Array], Array],
        add_fn: Callable[[Array, Array, Array, Array], Array] = vanilla_add_fn,
    ) -> Array:
        x = func(samples, t)
        return add_fn(samples, x, t, next_t)


class HeunODESolver(BaseODESolver):
    """The Heun method for solving ODEs."""

    def step(
        self,
        samples: Array,
        t: Array,
        next_t: Array,
        func: Callable[[Array, Array], Array],
        add_fn: Callable[[Array, Array, Array, Array], Array] = vanilla_add_fn,
    ) -> Array:
        k1 = func(samples, t)
        k2 = func(add_fn(samples, k1, t, next_t), next_t)
        x = (k1 + k2) / 2
        return add_fn(samples, x, t, next_t)


class RK4ODESolver(BaseODESolver):
    """The fourth-order Runge-Kutta method for solving ODEs."""

    def step(
        self,
        samples: Array,
        t: Array,
        next_t: Array,
        func: Callable[[Array, Array], Array],
        add_fn: Callable[[Array, Array, Array, Array], Array] = vanilla_add_fn,
    ) -> Array:
        dt = next_t - t
        half_t = t + dt / 2
        k1 = func(samples, t)
        k2 = func(add_fn(samples, k1 / 2, t, half_t), half_t)
        k3 = func(add_fn(samples, k2, t, half_t), half_t)
        k4 = func(add_fn(samples, k3, t, next_t), next_t)
        x = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return add_fn(samples, x, t, next_t)


def get_ode_solver(s: ODESolverType) -> BaseODESolver:
    """Returns an ODE solver for a given key.

    Args:
        s: The solver key to retrieve.

    Returns:
        The solver object.
    """
    match s:
        case "euler":
            return EulerODESolver()
        case "heun":
            return HeunODESolver()
        case "rk4":
            return RK4ODESolver()
        case _:
            raise ValueError(f"Unknown solver type: {s}")


def _warmup_beta_schedule(
    beta_start: float,
    beta_end: float,
    num_timesteps: int,
    warmup: float,
) -> Array:
    betas = jnp.full((num_timesteps,), beta_end)
    warmup_time = int(num_timesteps * warmup)
    betas = betas.at[:warmup_time].set(jnp.linspace(beta_start, beta_end, warmup_time))
    return betas


def _cosine_beta_schedule(
    num_timesteps: int,
    offset: float = 0.008,
) -> Array:
    rng = jnp.arange(num_timesteps, dtype=jnp.float32)
    f_t = jnp.cos((rng / (num_timesteps - 1) + offset) / (1 + offset) * jnp.pi / 2) ** 2
    bar_alpha = f_t / f_t[0]
    beta = jnp.zeros_like(bar_alpha)
    beta = beta.at[1:].set((1 - (bar_alpha[1:] / bar_alpha[:-1])).clip(0, 0.999))
    return beta


def get_diffusion_beta_schedule(
    schedule: DiffusionBetaSchedule,
    num_timesteps: int,
    *,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    warmup: float = 0.1,
    cosine_offset: float = 0.008,
) -> Array:
    """Returns a beta schedule for the given schedule type.

    Args:
        schedule: The schedule type.
        num_timesteps: The total number of timesteps.
        beta_start: The initial beta value, for linear, quad, and warmup
            schedules.
        beta_end: The final beta value, for linear, quad, warmup and const
            schedules.
        warmup: The fraction of timesteps to use for the warmup schedule
            (between 0 and 1).
        cosine_offset: The cosine offset, for cosine schedules.

    Returns:
        The beta schedule, an array with shape ``(num_timesteps)``.
    """
    match schedule:
        case "linear":
            return jnp.linspace(beta_start, beta_end, num_timesteps)
        case "quad":
            return jnp.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        case "warmup":
            return _warmup_beta_schedule(beta_start, beta_end, num_timesteps, warmup)
        case "const":
            return jnp.full((num_timesteps,), beta_end)
        case "cosine":
            return _cosine_beta_schedule(num_timesteps, cosine_offset)
        case "jsd":
            return jnp.linspace(num_timesteps, 1, num_timesteps) ** -1.0
        case _:
            raise NotImplementedError(f"Unknown schedule type: {schedule}")


class GaussianDiffusion(eqx.Module):
    """Defines a module which provides utility functions for Gaussian diffusion.

    Parameters:
        beta_schedule: The beta schedule type to use.
        num_beta_steps: The number of beta steps to use.
        pred_mode: The prediction mode, which determines what the model should
            predict. Can be one of:

            - ``"pred_x_0"``: Predicts the initial noise.
            - ``"pred_eps"``: Predicts the noise at the current timestep.
            - ``"pred_v"``: Predicts the velocity of the noise.

        loss: The type of loss to use. Can be one of:

                - ``"mse"``: Mean squared error.
                - ``"l1"``: Mean absolute error.

        sigma_type: The type of sigma to use. Can be one of:

                - ``"upper_bound"``: The upper bound of the posterior noise.
                - ``"lower_bound"``: The lower bound of the posterior noise.

        solver: The ODE solver to use for running incremental model steps.
            If not set, will default to using the built-in ODE solver.
        beta_start: The initial beta value, for linear, quad, and warmup
            schedules.
        beta_end: The final beta value, for linear, quad, warmup and const
            schedules.
        warmup: The fraction of timesteps to use for the warmup schedule
            (between 0 and 1).
        cosine_offset: The cosine offset, for cosine schedules.
    """

    bar_alpha: Array = eqx.field()
    beta_schedule: DiffusionBetaSchedule = eqx.field(static=True)
    num_beta_steps: int = eqx.field(static=True)
    pred_mode: DiffusionPredMode = eqx.field(static=True)
    loss_fn: DiffusionLossFn = eqx.field(static=True)
    sigma_type: SigmaType = eqx.field(static=True)
    beta_start: float = eqx.field(static=True)
    beta_end: float = eqx.field(static=True)
    warmup: float = eqx.field(static=True)
    cosine_offset: float = eqx.field(static=True)
    num_timesteps: int = eqx.field(static=True)
    solver: BaseODESolver = eqx.field()

    def __init__(
        self,
        beta_schedule: DiffusionBetaSchedule = "linear",
        num_beta_steps: int = 1000,
        pred_mode: DiffusionPredMode = "pred_x_0",
        loss: DiffusionLossFn = "mse",
        sigma_type: SigmaType = "upper_bound",
        solver: ODESolverType = "euler",
        *,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        warmup: float = 0.1,
        cosine_offset: float = 0.008,
    ) -> None:
        self.beta_schedule = beta_schedule
        self.num_beta_steps = num_beta_steps
        self.pred_mode = pred_mode
        self.loss_fn = loss
        self.sigma_type = sigma_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup = warmup
        self.cosine_offset = cosine_offset
        self.num_timesteps = num_beta_steps - 1

        # Initialize bar_alpha with placeholder, will be set in reset_parameters
        self.bar_alpha = jnp.empty(self.num_beta_steps)

        # The ODE solver to use.
        self.solver = get_ode_solver(solver)

    def reset_parameters(self) -> "GaussianDiffusion":
        """Resets the parameters of the diffusion model."""
        # Gets the beta schedule from the given parameters.
        betas = get_diffusion_beta_schedule(
            schedule=self.beta_schedule,
            num_timesteps=self.num_beta_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            warmup=self.warmup,
            cosine_offset=self.cosine_offset,
        )

        assert betas.ndim == 1

        assert not (betas < 0).any(), "Betas must be non-negative."
        assert not (betas > 1).any(), "Betas must be less than or equal to 1."

        bar_alpha = jnp.cumprod(1.0 - betas, axis=0)

        return eqx.tree_at(lambda x: x.bar_alpha, self, bar_alpha)

    def get_noise(self, key: PRNGKeyArray, x: Array) -> Array:
        return jax.random.normal(key, x.shape, dtype=x.dtype)

    def loss_tensors(self, key: PRNGKeyArray, model: Callable[[Array, Array], Array], x: Array) -> tuple[Array, Array]:
        """Computes the loss for a given sample.

        Args:
            key: Random key for sampling.
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            x: The input data, with shape ``(*)``

        Returns:
            A tuple of (pred_target, gt_target).
        """
        bsz = x.shape[0]
        t_key, eps_key = jax.random.split(key)
        t_sample = jax.random.randint(t_key, (bsz,), 1, self.num_timesteps + 1)
        eps = self.get_noise(eps_key, x)
        bar_alpha = self.bar_alpha[t_sample].reshape(-1, *[1] * (x.ndim - 1))
        x_t = jnp.sqrt(bar_alpha) * x + jnp.sqrt(1 - bar_alpha) * eps
        pred_target = model(x_t, t_sample)
        match self.pred_mode:
            case "pred_x_0":
                gt_target = x
            case "pred_eps":
                gt_target = eps
            case "pred_v":
                gt_target = jnp.sqrt(bar_alpha) * eps - jnp.sqrt(1 - bar_alpha) * x
            case _:
                raise NotImplementedError(f"Unknown pred_mode: {self.pred_mode}")
        return pred_target, gt_target

    def loss(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        x: Array,
        loss: DiffusionLossFn | Callable[[Array, Array], Array] = "mse",
        loss_dim: int = -1,
        loss_factor: float = 0.00054,
    ) -> Array:
        pred_target, gt_target = self.loss_tensors(key, model, x)
        if callable(loss):
            return loss(pred_target, gt_target)
        match loss:
            case "mse":
                return (pred_target - gt_target) ** 2
            case "l1":
                return jnp.abs(pred_target - gt_target)
            case "pseudo-huber":
                return pseudo_huber_loss(pred_target, gt_target, dim=loss_dim, factor=loss_factor)
            case _:
                raise NotImplementedError(f"Unknown loss: {loss}")

    def partial_sample(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        reference_sample: Array,
        start_percent: float,
        sampling_timesteps: int | None = None,
        solver: BaseODESolver | None = None,
    ) -> Array:
        """Samples from the model, starting from a given reference sample.

        Partial sampling takes a reference sample, adds some noise to it, then
        denoises the sample using the model. This can be used for doing
        style transfer, where the reference sample is the source image which
        the model redirects to look more like some target style.

        Args:
            key: Random key for sampling.
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            reference_sample: The reference sample, with shape ``(*)``.
            start_percent: What percent of the diffusion process to start from;
                0 means that all of the diffusion steps will be used, while 1
                means that none of the diffusion steps will be used.
            sampling_timesteps: The number of timesteps to sample for. If
                ``None``, then the full number of timesteps will be used.
            solver: The ODE solver to use for running incremental model steps.
                If not set, will default to using the built-in ODE solver.

        Returns:
            The samples, with shape ``(sampling_timesteps + 1, *)``.
        """
        assert 0.0 <= start_percent <= 1.0
        num_timesteps = round(self.num_timesteps * start_percent)
        scalar_t_start = num_timesteps
        noise = self.get_noise(key, reference_sample)
        bar_alpha = self.bar_alpha[scalar_t_start].reshape(-1, *[1] * (noise.ndim - 1))
        x = jnp.sqrt(bar_alpha) * reference_sample + jnp.sqrt(1 - bar_alpha) * noise
        return self._sample_common(
            key=key,
            model=model,
            x=x,
            solver=solver,
            sampling_timesteps=sampling_timesteps,
            start_percent=start_percent,
        )

    def sample(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        shape: tuple[int, ...],
        sampling_timesteps: int | None = None,
        solver: BaseODESolver | None = None,
    ) -> Array:
        """Samples from the model.

        Args:
            key: Random key for sampling.
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            shape: The shape of the samples.
            sampling_timesteps: The number of timesteps to sample for. If
                ``None``, then the full number of timesteps will be used.
            solver: The ODE solver to use for running incremental model steps.
                If not set, will default to using the built-in ODE solver.

        Returns:
            The samples, with shape ``(sampling_timesteps + 1, *)``.
        """
        init_key, sample_key = jax.random.split(key)
        x = jax.random.normal(init_key, shape)
        return self._sample_common(
            key=sample_key,
            model=model,
            x=x,
            solver=solver,
            sampling_timesteps=sampling_timesteps,
            start_percent=0.0,
        )

    def _get_t_tensor(self, t: int, x: Array) -> Array:
        return jnp.full((x.shape[0],), t, dtype=jnp.int32)

    def _get_bar_alpha(self, t: Array, x: Array) -> Array:
        # When using non-integer timesteps, like when using the RK4 ODE solver,
        # we interpolate the `bar_alpha` values. Since `bar_alpha` is a
        # cumulative product we need to do a weighted geometric mean rather than
        # a linear mean. Side note: This code works for both the case where
        # `t_max - t_min` is 1 and where it is 0.
        if t.dtype in (jnp.float32, jnp.float64):
            t_min = jnp.floor(t).astype(jnp.int32)
            t_max = jnp.ceil(t).astype(jnp.int32)
            bar_alpha_min = self.bar_alpha[t_min]
            bar_alpha_max = self.bar_alpha[t_max]
            w_min = t - t_min.astype(t.dtype)
            factor = bar_alpha_max / bar_alpha_min
            bar_alpha = jnp.power(factor, w_min) * bar_alpha_min
        else:
            bar_alpha = self.bar_alpha[t]
        return append_dims(bar_alpha, x.ndim - bar_alpha.ndim)

    def _run_model(self, model: Callable[[Array, Array], Array], x: Array, t: Array, bar_alpha: Array) -> Array:
        # Use model to predict x_0.
        match self.pred_mode:
            case "pred_x_0":
                return model(x, t)
            case "pred_eps":
                pred_eps = model(x, t)
                return (x - jnp.sqrt(1 - bar_alpha) * pred_eps) / jnp.sqrt(bar_alpha)
            case "pred_v":
                pred_v = model(x, t)
                return jnp.sqrt(bar_alpha) * x - jnp.sqrt(1 - bar_alpha) * pred_v
            case _:
                raise AssertionError(f"Invalid {self.pred_mode=}.")

    def _sample_step(
        self,
        model: Callable[[Array, Array], Array],
        x: Array,
        solver: BaseODESolver,
        scalar_t_start: int,
        scalar_t_end: int,
    ) -> Array:
        def func(x: Array, t: Array) -> Array:
            return self._run_model(model, x, t, self._get_bar_alpha(t, x))

        def add_fn(x: Array, pred_x: Array, t: Array, next_t: Array) -> Array:
            bar_alpha, bar_alpha_next = self._get_bar_alpha(t, x), self._get_bar_alpha(next_t, x)
            lhs_factor = jnp.sqrt(bar_alpha / bar_alpha_next) * (1 - bar_alpha_next)
            rhs_factor = jnp.sqrt(bar_alpha_next) * (1 - bar_alpha / bar_alpha_next)
            lhs = lhs_factor * x
            rhs = rhs_factor * pred_x
            return (lhs + rhs) / (1 - bar_alpha)

        t_start, t_end = self._get_t_tensor(scalar_t_start, x), self._get_t_tensor(scalar_t_end, x)

        return solver.step(x, t_start, t_end, func, add_fn)

    def _add_noise(self, key: PRNGKeyArray, x: Array, scalar_t_start: int, scalar_t_end: int) -> Array:
        t_start, t_end = self._get_t_tensor(scalar_t_start, x), self._get_t_tensor(scalar_t_end, x)
        bar_alpha_start, bar_alpha_end = self._get_bar_alpha(t_start, x), self._get_bar_alpha(t_end, x)

        # Forward model posterior noise
        match self.sigma_type:
            case "upper_bound":
                std = jnp.sqrt(1 - bar_alpha_start / bar_alpha_end)
                noise = std * self.get_noise(key, x)
            case "lower_bound":
                std = jnp.sqrt((1 - bar_alpha_start / bar_alpha_end) * (1 - bar_alpha_end) / (1 - bar_alpha_start))
                noise = std * self.get_noise(key, x)
            case _:
                raise AssertionError(f"Invalid {self.sigma_type=}.")

        return x + noise

    def _sample_common(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        x: Array,
        solver: BaseODESolver | None = None,
        sampling_timesteps: int | None = None,
        start_percent: float = 0.0,
    ) -> Array:
        assert 0.0 <= start_percent <= 1.0
        if solver is None:
            solver = self.solver

        sampling_timesteps = self.num_timesteps if sampling_timesteps is None else sampling_timesteps
        assert 1 <= sampling_timesteps <= self.num_timesteps

        # Start sampling at `start_percent` instead of at zero.
        num_timesteps = round(self.num_timesteps * (1 - start_percent))
        sampling_timesteps = round(sampling_timesteps * (1 - start_percent))

        subseq = jnp.round(jnp.linspace(num_timesteps, 0, sampling_timesteps + 1)).astype(jnp.int32)
        samples = jnp.zeros((sampling_timesteps + 1, *x.shape), dtype=x.dtype)
        samples = samples.at[-1].set(x)

        key, init_key = jax.random.split(key)
        current_x = x
        current_key = init_key

        for idx in range(sampling_timesteps):
            t_start = int(subseq[idx])
            t_end = int(subseq[idx + 1])
            current_x = self._sample_step(model, current_x, solver, t_start, t_end)
            if t_end != 0:
                current_key, noise_key = jax.random.split(current_key)
                current_x = self._add_noise(noise_key, current_x, t_start, t_end)
            samples = samples.at[sampling_timesteps - 1 - idx].set(current_x)

        return samples


class ConsistencyModel(eqx.Module):
    """Defines a module which implements consistency diffusion models.

    This model introduces an auxiliary consistency penalty to the loss function
    to encourage the ODE to be smooth, allowing for few-step inference.

    This also implements the improvements to vanilla diffusion described in
    ``Elucidating the Design Space of Diffusion-Based Generative Models``.

    Parameters:
        total_steps: The maximum number of training steps, used for determining
            the discretization step schedule.
        sigma_data: The standard deviation of the data.
        sigma_max: The maximum standard deviation for the diffusion process.
        sigma_min: The minimum standard deviation for the diffusion process.
        rho: The rho constant for the noise schedule.
        p_mean: A constant which controls the distribution of timesteps to
            sample for training. Training biases towards sampling timesteps
            from the less noisy end of the spectrum to improve convergence.
        p_std: Another constant that controls the distribution of timesteps for
            training, used in conjunction with ``p_mean``.
        start_scales: The number of different discretization scales to use at
            the start of training. At the start of training, a small number of
            scales is used to encourage the model to learn more quickly, which
            is increased over time.
        end_scales: The number of different discretization scales to use at the
            end of training.
    """

    total_steps: int | None = eqx.field(static=True)
    sigma_data: float = eqx.field(static=True)
    sigma_max: float = eqx.field(static=True)
    sigma_min: float = eqx.field(static=True)
    rho: float = eqx.field(static=True)
    p_mean: float = eqx.field(static=True)
    p_std: float = eqx.field(static=True)
    start_scales: int = eqx.field(static=True)
    end_scales: int = eqx.field(static=True)

    def __init__(
        self,
        total_steps: int | None = None,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
        p_mean: float = -1.1,
        p_std: float = 2.0,
        start_scales: int = 20,
        end_scales: int = 1280,
    ) -> None:
        self.total_steps = total_steps
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std
        self.start_scales = start_scales
        self.end_scales = end_scales

    def get_noise(self, key: PRNGKeyArray, x: Array) -> Array:
        return jax.random.normal(key, x.shape, dtype=x.dtype)

    def loss_tensors(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        x: Array,
        step: int,
    ) -> tuple[Array, Array, Array, Array]:
        """Computes the consistency model loss.

        Args:
            key: Random key for sampling.
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            x: The input data, with shape ``(*)``
            step: The current training step, used to determine the number of
                discretization steps to use.

        Returns:
            A tuple of (y_current, y_next, sigma_next, sigma_current).
        """
        dims = x.ndim

        # The number of discretization steps runs on a schedule.
        num_scales = self._get_num_scales(step)

        # Rather than randomly sampling some timesteps for training, we bias the
        # samples to be closer to the less noisy end, which improves training
        # stability. This distribution is defined as a function of the standard
        # deviations.
        timesteps = self._sample_timesteps(key, x, num_scales)
        t_current, t_next = timesteps / (num_scales - 1), (timesteps + 1) / (num_scales - 1)

        # Converts timesteps to sigmas.
        sigmas = self._get_sigmas(jnp.stack((t_next, t_current)))
        sigma_next, sigma_current = sigmas[0], sigmas[1]

        noise_key, dropout_key = jax.random.split(key)
        noise = self.get_noise(noise_key, x)
        x_current = x + noise * append_dims(sigma_current, dims - sigma_current.ndim)
        y_current = self._call_model(model, x_current, sigma_current)

        # Resets the dropout state and runs the target model.
        # In JAX, we need to use the same key for deterministic behavior
        # For dropout, we'd need to handle this differently if using dropout
        x_next = x + noise * append_dims(sigma_next, dims - sigma_next.ndim)
        y_next = self._call_model(model, x_next, sigma_next)

        return y_current, y_next, sigma_next, sigma_current

    def loss_function(
        self,
        y_current: Array,
        y_next: Array,
        loss: DiffusionLossFn | Callable[[Array, Array], Array] = "mse",
        loss_dim: int = -1,
        loss_factor: float = 0.00054,
    ) -> Array:
        if callable(loss):
            return loss(y_current, y_next)
        match loss:
            case "mse":
                return (y_current - y_next) ** 2
            case "l1":
                return jnp.abs(y_current - y_next)
            case "pseudo-huber":
                return pseudo_huber_loss(y_current, y_next, dim=loss_dim, factor=loss_factor)
            case _:
                raise NotImplementedError(f"Unknown loss: {loss}")

    def loss(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        x: Array,
        step: int,
        loss: DiffusionLossFn | Callable[[Array, Array], Array] = "pseudo-huber",
        loss_dim: int = -1,
        loss_factor: float = 0.00054,
    ) -> Array:
        y_current, y_next, sigma_next, sigma_current = self.loss_tensors(key, model, x, step)
        loss_value = self.loss_function(y_current, y_next, loss, loss_dim, loss_factor)
        weights = 1 / (sigma_current - sigma_next)
        weights = weights.reshape(-1, *([1] * (loss_value.ndim - 1)))
        return loss_value * weights

    def partial_sample(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        reference_sample: Array,
        start_percent: float,
        num_steps: int,
    ) -> Array:
        """Samples from the model, starting from a given reference sample.

        Partial sampling takes a reference sample, adds some noise to it, then
        denoises the sample using the model. This can be used for doing
        style transfer, where the reference sample is the source image which
        the model redirects to look more like some target style.

        Args:
            key: Random key for sampling.
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            reference_sample: The reference sample, with shape ``(*)``.
            start_percent: The percentage of timesteps to start sampling from.
            num_steps: The number of sampling steps to use.

        Returns:
            The samples, with shape ``(num_steps + 1, *)``, with the first
            sample (i.e., ``samples[0]``) as the denoised output and the last
            sample (i.e., ``samples[-1]``) as the reference sample.
        """
        assert 0.0 <= start_percent <= 1.0
        timesteps = jnp.linspace(start_percent, 1, num_steps + 1)
        sigmas = self._get_sigmas(timesteps)
        x = reference_sample
        init_key, sample_key = jax.random.split(key)
        x = x + jax.random.normal(init_key, x.shape, dtype=x.dtype) * sigmas[0]
        samples = jnp.zeros((num_steps + 1, *x.shape), dtype=x.dtype)
        samples = samples.at[num_steps].set(x)

        current_x = x
        current_key = sample_key

        for i in range(num_steps):
            current_x = self._call_model(model, current_x, sigmas[i : i + 1])
            samples = samples.at[num_steps - 1 - i].set(current_x)
            if i < num_steps - 1:
                current_key, noise_key = jax.random.split(current_key)
                current_x = current_x + self.get_noise(noise_key, current_x) * sigmas[i + 1 : i + 2]

        return samples

    def sample(
        self,
        key: PRNGKeyArray,
        model: Callable[[Array, Array], Array],
        shape: tuple[int, ...],
        num_steps: int,
    ) -> Array:
        """Samples from the model.

        Args:
            key: Random key for sampling.
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            shape: The shape of the samples.
            num_steps: The number of sampling steps to use.

        Returns:
            The samples, with shape ``(num_steps + 1, *)``, with the first
            sample (i.e., ``samples[0]``) as the denoised output and the last
            sample (i.e., ``samples[-1]``) as the random noise.
        """
        init_key, sample_key = jax.random.split(key)
        timesteps = jnp.linspace(0, 1, num_steps + 1)
        sigmas = self._get_sigmas(timesteps)
        x = jax.random.normal(init_key, shape) * sigmas[0]
        samples = jnp.zeros((num_steps + 1, *x.shape), dtype=x.dtype)
        samples = samples.at[num_steps].set(x)

        current_x = x
        current_key = sample_key

        for i in range(num_steps):
            current_x = self._call_model(model, current_x, sigmas[i : i + 1])
            samples = samples.at[num_steps - 1 - i].set(current_x)
            if i < num_steps - 1:
                current_key, noise_key = jax.random.split(current_key)
                current_x = current_x + self.get_noise(noise_key, current_x) * sigmas[i + 1 : i + 2]

        return samples

    def _get_scalings(self, sigma: Array) -> tuple[Array, Array, Array]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def _call_model(self, model: Callable[[Array, Array], Array], x_t: Array, sigmas: Array) -> Array:
        c_skip, c_out, c_in = (append_dims(x, x_t.ndim - x.ndim) for x in self._get_scalings(sigmas))
        timesteps = 1000 * 0.25 * jnp.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, timesteps)
        denoised = c_out * model_output + c_skip * x_t
        return denoised

    def _get_sigmas(self, timesteps: Array) -> Array:
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas: Array = (max_inv_rho + timesteps * (min_inv_rho - max_inv_rho)) ** self.rho
        sigmas = jnp.where(timesteps >= 1.0, jnp.full_like(sigmas, self.sigma_min), sigmas)
        return sigmas

    def _get_noise_distribution(self, sigma_next: Array, sigma_current: Array) -> Array:
        denom = jnp.sqrt(2) * self.p_std
        lhs = jax.scipy.special.erf((jnp.log(sigma_next) - self.p_mean) / denom)
        rhs = jax.scipy.special.erf((jnp.log(sigma_current) - self.p_mean) / denom)
        return rhs - lhs

    def _sample_timesteps(self, key: PRNGKeyArray, x: Array, num_scales: int) -> Array:
        timesteps = jnp.linspace(0, 1, num_scales)
        sigmas = self._get_sigmas(timesteps)
        noise_dist = self._get_noise_distribution(sigmas[1:], sigmas[:-1])
        # Normalize the distribution
        noise_dist = noise_dist / (noise_dist.sum() + 1e-8)
        timesteps = jax.random.categorical(key, jnp.log(noise_dist + 1e-8), shape=(x.shape[0],))
        return timesteps

    def _get_num_scales(self, step: int) -> int:
        if self.total_steps is None:
            return self.end_scales + 1
        num_steps = min(self.total_steps, step)
        k_prime = math.floor(self.total_steps / (math.log2(self.end_scales / self.start_scales) + 1))
        return min(self.start_scales * 2 ** math.floor(num_steps / k_prime), self.end_scales) + 1
