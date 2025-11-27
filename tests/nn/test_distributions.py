"""Unit tests for different probability distributions."""

import jax
import jax.numpy as jnp

import xax


class TestCategorical:
    """Test the Categorical distribution."""

    def test_init(self) -> None:
        """Test initialization."""
        logits = jnp.array([1.0, 2.0, 3.0])
        cat = xax.Categorical(logits)
        assert cat.logits_nc.shape == (3,)
        assert jnp.array_equal(cat.logits_nc, logits)

    def test_log_prob(self) -> None:
        """Test log probability computation."""
        logits = jnp.array([1.0, 2.0, 3.0])
        cat = xax.Categorical(logits)

        # Test single category
        x = jnp.array(1)  # Second category
        log_prob = cat.log_prob(x)
        expected_log_probs = jax.nn.log_softmax(logits)
        assert jnp.allclose(log_prob, expected_log_probs[1])

        # Test multiple categories
        x = jnp.array([0, 2])  # First and third categories
        log_probs = cat.log_prob(x)
        assert jnp.allclose(log_probs, expected_log_probs[jnp.array([0, 2])])

    def test_sample(self) -> None:
        """Test sampling."""
        logits = jnp.array([1.0, 2.0, 3.0])
        cat = xax.Categorical(logits)

        key = jax.random.key(0)
        samples = cat.sample(key)

        # Check shape
        assert samples.shape == ()
        # Check that samples are valid indices
        assert 0 <= samples < 3

    def test_sample_batch(self) -> None:
        """Test sampling with batch dimensions."""
        logits = jnp.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
        cat = xax.Categorical(logits)

        key = jax.random.key(0)
        samples = cat.sample(key)

        # Check shape
        assert samples.shape == (2,)
        # Check that samples are valid indices
        assert jnp.all((0 <= samples) & (samples < 3))

    def test_mode(self) -> None:
        """Test mode computation."""
        logits = jnp.array([1.0, 2.0, 3.0])
        cat = xax.Categorical(logits)

        mode = cat.mode()
        expected_mode = jnp.argmax(logits)
        assert mode == expected_mode

    def test_entropy(self) -> None:
        """Test entropy computation."""
        logits = jnp.array([1.0, 2.0, 3.0])
        cat = xax.Categorical(logits)

        entropy = cat.entropy()

        # Compute expected entropy manually
        probs = jax.nn.softmax(logits)
        expected_entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))

        assert jnp.allclose(entropy, expected_entropy)

    def test_entropy_uniform(self) -> None:
        """Test entropy for uniform distribution."""
        logits = jnp.array([0.0, 0.0, 0.0])  # Uniform distribution
        cat = xax.Categorical(logits)

        entropy = cat.entropy()
        expected_entropy = jnp.log(3.0)  # log(n) for uniform over n categories

        assert jnp.allclose(entropy, expected_entropy)


class TestNormal:
    """Test the Normal distribution."""

    def test_init(self) -> None:
        """Test initialization."""
        mean = jnp.array(0.0)
        std = jnp.array(1.0)
        normal = xax.Normal(mean, std)
        assert normal.loc_n.shape == ()
        assert normal.scale_n.shape == ()
        assert jnp.array_equal(normal.loc_n, mean)
        assert jnp.array_equal(normal.scale_n, std)

    def test_log_prob(self) -> None:
        """Test log probability computation."""
        mean = jnp.array(0.0)
        std = jnp.array(1.0)
        normal = xax.Normal(mean, std)

        x_n = jnp.array(1.0)
        log_prob = normal.log_prob(x_n)

        # Manual computation
        expected_log_prob = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(std) - 0.5 * ((x_n - mean) / std) ** 2
        assert jnp.allclose(log_prob, expected_log_prob)

    def test_log_prob_batch(self) -> None:
        """Test log probability computation with batch dimensions."""
        mean = jnp.array([0.0, 1.0])
        std = jnp.array([1.0, 2.0])
        normal = xax.Normal(mean, std)

        x_n = jnp.array([0.5, 1.5])
        log_probs = normal.log_prob(x_n)

        # Check shape
        assert log_probs.shape == (2,)
        # Check that log probabilities are finite
        assert jnp.all(jnp.isfinite(log_probs))

    def test_sample(self) -> None:
        """Test sampling."""
        mean_n = jnp.array(0.0)
        std_n = jnp.array(1.0)
        normal = xax.Normal(mean_n, std_n)

        key = jax.random.key(0)
        samples = normal.sample(key)

        # Check shape
        assert samples.shape == ()
        # Check that samples are reasonable (within 4 standard deviations)
        assert jnp.abs(samples - mean_n) < 4 * std_n

    def test_sample_batch(self) -> None:
        """Test sampling with batch dimensions."""
        mean_n = jnp.array([0.0, 1.0])
        std_n = jnp.array([1.0, 2.0])
        normal = xax.Normal(mean_n, std_n)

        key = jax.random.key(0)
        samples = normal.sample(key)

        # Check shape
        assert samples.shape == (2,)
        # Check that samples are reasonable
        assert jnp.all(jnp.abs(samples - mean_n) < 4 * std_n)

    def test_mode(self) -> None:
        """Test mode computation."""
        mean_n = jnp.array(0.0)
        std_n = jnp.array(1.0)
        normal = xax.Normal(mean_n, std_n)

        mode = normal.mode()
        assert jnp.allclose(mode, mean_n)

    def test_entropy(self) -> None:
        """Test entropy computation."""
        mean_n = jnp.array(0.0)
        std_n = jnp.array(1.0)
        normal = xax.Normal(mean_n, std_n)

        entropy = normal.entropy()
        expected_entropy = jnp.log(2 * jnp.pi * jnp.e) + jnp.log(std_n)
        assert jnp.allclose(entropy, expected_entropy)

    def test_standard_normal(self) -> None:
        """Test standard normal distribution."""
        normal = xax.Normal(jnp.array(0.0), jnp.array(1.0))

        # Test entropy
        entropy = normal.entropy()
        # log(2πe) for standard normal
        expected_entropy = jnp.log(2 * jnp.pi * jnp.e)
        assert jnp.allclose(entropy, expected_entropy)


class TestMixtureOfGaussians:
    """Test the MixtureOfGaussians distribution."""

    def test_init(self) -> None:
        """Test initialization."""
        means = jnp.array([0.0, 2.0])
        stds = jnp.array([1.0, 1.0])
        logits = jnp.array([0.0, 0.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        assert mixture.means_nm.shape == (2,)
        assert mixture.stds_nm.shape == (2,)
        assert mixture.logits_nm.shape == (2,)
        assert jnp.array_equal(mixture.means_nm, means)
        assert jnp.array_equal(mixture.stds_nm, stds)
        assert jnp.array_equal(mixture.logits_nm, logits)

    def test_init_batch(self) -> None:
        """Test initialization with batch dimensions."""
        means = jnp.array([[0.0, 2.0], [1.0, 3.0]])
        stds = jnp.array([[1.0, 1.0], [0.5, 1.5]])
        logits = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        assert mixture.means_nm.shape == (2, 2)
        assert mixture.stds_nm.shape == (2, 2)
        assert mixture.logits_nm.shape == (2, 2)

    def test_log_prob(self) -> None:
        """Test log probability computation."""
        means = jnp.array([0.0, 2.0])
        stds = jnp.array([1.0, 1.0])
        logits = jnp.array([0.0, 0.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        x = jnp.array(1.0)
        log_prob = mixture.log_prob(x)

        # Check that log probability is finite
        assert jnp.isfinite(log_prob)
        # Check that log probability is reasonable (not too large negative)
        assert log_prob > -100

    def test_log_prob_batch(self) -> None:
        """Test log probability computation with batch dimensions."""
        means = jnp.array([[0.0, 2.0], [1.0, 3.0]])
        stds = jnp.array([[1.0, 1.0], [0.5, 1.5]])
        logits = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        x = jnp.array([1.0, 2.0])
        log_probs = mixture.log_prob(x)

        # Check shape
        assert log_probs.shape == (2,)
        # Check that log probabilities are finite
        assert jnp.all(jnp.isfinite(log_probs))

    def test_sample(self) -> None:
        """Test sampling."""
        means = jnp.array([0.0, 2.0])
        stds = jnp.array([1.0, 1.0])
        logits = jnp.array([0.0, 0.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        key = jax.random.key(0)
        samples = mixture.sample(key)

        # Check shape - should be scalar since we have scalar inputs
        assert samples.shape == ()
        # Check that samples are reasonable
        # (within 4 standard deviations of either mean)
        assert jnp.any(jnp.abs(samples - means) < 4 * stds)

    def test_sample_batch(self) -> None:
        """Test sampling with batch dimensions."""
        means = jnp.array([[0.0, 2.0], [1.0, 3.0]])
        stds = jnp.array([[1.0, 1.0], [0.5, 1.5]])
        logits = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        key = jax.random.key(0)
        samples = mixture.sample(key)

        # Check shape - should match the batch dimensions
        assert samples.shape == (2,)
        # Check that samples are reasonable
        for i in range(2):
            assert jnp.any(jnp.abs(samples[i] - means[i]) < 4 * stds[i])

    def test_mode(self) -> None:
        """Test mode computation."""
        means = jnp.array([0.0, 2.0])
        stds = jnp.array([1.0, 1.0])
        # Second component has much higher weight
        logits = jnp.array([0.0, 5.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        mode = mixture.mode()
        # Should be close to the mean of the highest weight component
        expected_mode = means[1]  # Second component
        assert jnp.allclose(mode, expected_mode)

    def test_entropy(self) -> None:
        """Test entropy computation."""
        means = jnp.array([0.0, 2.0])
        stds = jnp.array([1.0, 1.0])
        logits = jnp.array([0.0, 0.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        entropy = mixture.entropy()

        # Check that entropy is finite and positive
        assert jnp.isfinite(entropy)
        assert entropy > 0


class TestDistributionProperties:
    """Test properties that should hold for all distributions."""

    def test_categorical_properties(self) -> None:
        """Test properties specific to categorical distributions."""
        logits = jnp.array([1.0, 2.0, 3.0])
        cat = xax.Categorical(logits)

        # Test that log probabilities sum to 1 when exponentiated
        x = jnp.arange(3)
        log_probs = cat.log_prob(x)
        probs = jnp.exp(log_probs)
        assert jnp.allclose(jnp.sum(probs), 1.0)

    def test_normal_properties(self) -> None:
        """Test properties specific to normal distributions."""
        mean = jnp.array(0.0)
        std = jnp.array(1.0)
        normal = xax.Normal(mean, std)

        # Test symmetry around mean
        x1 = jnp.array(1.0)
        x2 = jnp.array(-1.0)
        log_prob1 = normal.log_prob(x1)
        log_prob2 = normal.log_prob(x2)
        assert jnp.allclose(log_prob1, log_prob2)

    def test_mixture_weights(self) -> None:
        """Test that mixture respects mixing weights."""
        means = jnp.array([0.0, 10.0])  # Well-separated means
        stds = jnp.array([1.0, 1.0])
        # Second component has much higher weight
        logits = jnp.array([0.0, 5.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        # Test near second component
        x = jnp.array(10.0)
        log_prob = mixture.log_prob(x)

        # Should be close to second component's log probability
        # (but not exactly equal due to mixing)
        second_component_log_prob = xax.Normal(means[1], stds[1]).log_prob(x)
        # The mixture log prob should be slightly lower due to the mixing term
        assert log_prob < second_component_log_prob
        # But it should be close
        assert jnp.abs(log_prob - second_component_log_prob) < 1.0

    def test_mixture_sampling(self) -> None:
        """Test that mixture sampling produces reasonable results."""
        means = jnp.array([0.0, 10.0])  # Well-separated means
        stds = jnp.array([1.0, 1.0])
        # Second component has much higher weight
        logits = jnp.array([0.0, 5.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        key = jax.random.key(0)
        samples = [mixture.sample(jax.random.fold_in(key, i)) for i in range(50)]
        samples = jnp.array(samples)

        # Most samples should be near the second component (higher weight)
        near_second = jnp.abs(samples - means[1]) < 3 * stds[1]
        # At least 70% near second component
        assert jnp.mean(near_second) > 0.7


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_categorical_single_component(self) -> None:
        """Test categorical with single component."""
        logits = jnp.array([1.0])
        cat = xax.Categorical(logits)

        # Test sampling
        key = jax.random.key(0)
        samples = cat.sample(key)
        assert samples == 0  # Only possible value

        # Test log probability
        log_prob = cat.log_prob(jnp.array(0))
        assert jnp.allclose(log_prob, 0.0)  # log(1) = 0

    def test_mixture_single_component(self) -> None:
        """Test mixture with single component (should behave like normal)."""
        means = jnp.array([0.0])
        stds = jnp.array([1.0])
        logits = jnp.array([0.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        normal = xax.Normal(means[0], stds[0])

        # Test log probability
        x = jnp.array(1.0)
        mixture_log_prob = mixture.log_prob(x)
        normal_log_prob = normal.log_prob(x)
        assert jnp.allclose(mixture_log_prob, normal_log_prob)

        # Test sampling - both should produce reasonable samples from the same distribution
        key = jax.random.key(0)
        mixture_sample = mixture.sample(key)
        normal_sample = normal.sample(key)
        # Both should be reasonable samples from N(0, 1)
        assert jnp.abs(mixture_sample) < 4  # Within 4 standard deviations
        assert jnp.abs(normal_sample) < 4  # Within 4 standard deviations

    def test_mixture_extreme_weights(self) -> None:
        """Test mixture with extreme weight differences."""
        means = jnp.array([0.0, 10.0])
        stds = jnp.array([1.0, 1.0])
        # Extreme weight difference
        logits = jnp.array([-100.0, 100.0])
        mixture = xax.MixtureOfGaussians(means, stds, logits)

        # Test mode
        mode = mixture.mode()
        assert jnp.allclose(mode, means[1])  # Should be second component

        # Test sampling
        key = jax.random.key(0)
        samples = [mixture.sample(jax.random.fold_in(key, i)) for i in range(50)]
        samples = jnp.array(samples)

        # All samples should be near the second component
        near_second = jnp.abs(samples - means[1]) < 3 * stds[1]
        assert jnp.all(near_second)
