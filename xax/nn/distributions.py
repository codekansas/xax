"""Defines some probability distribution helper functions.

In general, it is preferrable to use Distrax or another library, but we wanted
to have a simple interface of our own so that we can quickly upgrade Jax
versions (since Distrax is tied pretty closely to Tensorflow).
"""

__all__ = [
    "ProbabilityDistribution",
    "Categorical",
    "Normal",
    "MixtureOfGaussians",
]

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

STD_CLIP = 1e-6
LOGIT_CLIP = 6.0


class ProbabilityDistribution(ABC):
    @abstractmethod
    def log_prob(self, x_n: Array) -> Array: ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Array: ...

    @abstractmethod
    def mode(self) -> Array: ...

    @abstractmethod
    def entropy(self) -> Array: ...


class Categorical(ProbabilityDistribution):
    def __init__(self, logits_nc: Array, logit_clip: float = LOGIT_CLIP) -> None:
        """Initialize a categorical distribution.

        Args:
            logits_nc: Array of shape (..., n_categories) containing logits
            logit_clip: Clipping value for logits
        """
        self.logits_nc = jnp.clip(logits_nc, -logit_clip, logit_clip)

    @property
    def num_categories(self) -> int:
        return self.logits_nc.shape[-1]

    def log_prob(self, x_n: Array) -> Array:
        log_probs_n = jax.nn.log_softmax(self.logits_nc, axis=-1)
        return log_probs_n[x_n]

    def sample(self, key: PRNGKeyArray) -> Array:
        return jax.random.categorical(key, self.logits_nc, axis=-1)

    def mode(self) -> Array:
        return self.logits_nc.argmax(axis=-1)

    def entropy(self) -> Array:
        probs = jax.nn.softmax(self.logits_nc, axis=-1)
        log_probs = jax.nn.log_softmax(self.logits_nc, axis=-1)
        return -jnp.sum(probs * log_probs, axis=-1)


class Normal(ProbabilityDistribution):
    def __init__(self, loc_n: Array, scale_n: Array, std_clip: float = STD_CLIP) -> None:
        """Initialize a normal distribution.

        Args:
            loc_n: Mean of the distribution
            scale_n: Standard deviation of the distribution
            std_clip: Minimum standard deviation
        """
        self.loc_n = loc_n
        self.scale_n = jnp.clip(scale_n, min=std_clip)

    def log_prob(self, x_n: Array) -> Array:
        return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(self.scale_n) - (x_n - self.loc_n) ** 2 / (2 * self.scale_n**2)

    def sample(self, key: PRNGKeyArray) -> Array:
        return self.loc_n + self.scale_n * jax.random.normal(key, self.loc_n.shape)

    def mode(self) -> Array:
        return self.loc_n

    def entropy(self) -> Array:
        return jnp.log(2 * jnp.pi * jnp.e) + jnp.log(self.scale_n)


class MixtureOfGaussians(ProbabilityDistribution):
    def __init__(
        self,
        means_nm: Array,
        stds_nm: Array,
        logits_nm: Array,
        std_clip: float = STD_CLIP,
        logit_clip: float = LOGIT_CLIP,
    ) -> None:
        """Initialize a mixture of Gaussians.

        Args:
            means_nm: Array of shape (..., n_components) containing means
            stds_nm: Array of shape (..., n_components) containing standard deviations
            logits_nm: Array of shape (..., n_components) containing mixing logits
            std_clip: Minimum standard deviation
            logit_clip: Clipping value for logits
        """
        self.means_nm = means_nm
        self.stds_nm = jnp.clip(stds_nm, min=std_clip)
        self.logits_nm = jnp.clip(logits_nm, -logit_clip, logit_clip)

    def log_prob(self, x_n: Array) -> Array:
        # Expand x to match component dimensions
        x_n_expanded = x_n[..., None]  # Shape: (..., 1)

        # Compute log probabilities for each component
        component_log_probs = (
            -0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(self.stds_nm)
            - (x_n_expanded - self.means_nm) ** 2 / (2 * self.stds_nm**2)
        )

        # Compute mixing weights
        mixing_logits = jax.nn.log_softmax(self.logits_nm, axis=-1)

        # Combine using log-sum-exp trick for numerical stability
        return jax.scipy.special.logsumexp(component_log_probs + mixing_logits, axis=-1)

    def sample(self, key: PRNGKeyArray) -> Array:  # Sample component indices
        component_key, sample_key = jax.random.split(key)
        component_indices = jax.random.categorical(component_key, self.logits_nm, axis=-1)

        # Sample from selected components using advanced indexing
        # We need to handle the case where we have batch dimensions
        batch_shape = self.means_nm.shape[:-1]  # All dimensions except the last (components)

        # Reshape for easier indexing
        means_flat = self.means_nm.reshape(-1, self.means_nm.shape[-1])
        stds_flat = self.stds_nm.reshape(-1, self.stds_nm.shape[-1])
        indices_flat = component_indices.reshape(-1)

        # Get selected means and stds
        selected_means = means_flat[jnp.arange(len(indices_flat)), indices_flat]
        selected_stds = stds_flat[jnp.arange(len(indices_flat)), indices_flat]

        # Generate random noise
        noise = jax.random.normal(sample_key, selected_means.shape)

        # Reshape back to original batch shape
        samples_n = selected_means + selected_stds * noise
        return samples_n.reshape(batch_shape)

    def mode(self) -> Array:
        """Return the mode of the mixture (approximate - returns mean of highest weight component)."""
        mixing_weights = jax.nn.softmax(self.logits_nm, axis=-1)
        max_weight_idx = jnp.argmax(mixing_weights, axis=-1)

        # Use advanced indexing to get the means of the highest weight components
        batch_shape = self.means_nm.shape[:-1]
        means_flat = self.means_nm.reshape(-1, self.means_nm.shape[-1])
        indices_flat = max_weight_idx.reshape(-1)

        selected_means = means_flat[jnp.arange(len(indices_flat)), indices_flat]
        return selected_means.reshape(batch_shape)

    def entropy(self) -> Array:
        """Compute entropy of the mixture (approximate)."""
        mixing_weights = jax.nn.softmax(self.logits_nm, axis=-1)
        component_entropies = jnp.log(2 * jnp.pi * jnp.e) + jnp.log(self.stds_nm)

        # Weighted sum of component entropies plus mixing entropy
        weighted_entropies = jnp.sum(mixing_weights * component_entropies, axis=-1)
        mixing_entropy = -jnp.sum(mixing_weights * jnp.log(mixing_weights + 1e-8), axis=-1)

        return weighted_entropies + mixing_entropy
