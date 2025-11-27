"""Tests for the pytree utils."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree

import xax


def make_shuffle_test_data() -> PyTree:
    """Function to get simple pytree data for shuffling tests."""
    return {
        "observations": jnp.array(
            [
                # env 0
                [
                    [1, 2],  # t0 - feature_dim=2
                    [3, 4],  # t1
                    [5, 6],  # t2
                ],
                # env 1
                [
                    [7, 8],
                    [9, 10],
                    [11, 12],
                ],
            ]
        ),
        "actions": jnp.array(
            [
                # env 0
                [
                    [101, 102],
                    [201, 202],
                    [301, 302],
                ],
                # env 1
                [
                    [401, 402],
                    [501, 502],
                    [601, 602],
                ],
            ]
        ),
    }


def make_nested_dict_data() -> dict[str, PyTree]:
    """Function to get a deeply nested dictionary structure with arrays at the leaf nodes."""
    # Create base arrays with shape (num_envs=2, num_timesteps=3, feature_dim=2)
    obs_array = jnp.arange(1, 2 * 3 * 2 + 1).reshape(2, 3, 2)
    action_array = jnp.arange(101, 2 * 3 * 2 + 101).reshape(2, 3, 2)
    reward_array = jnp.arange(201, 2 * 3 + 201).reshape(2, 3)

    # Create the nested dictionary structure
    return {"level1": {"level2": {"observations": obs_array, "actions": action_array, "rewards": reward_array}}}


@pytest.mark.parametrize(
    "pytree, expected",
    [
        ({"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}, False),
        ({"a": jnp.array([1, jnp.nan, 3]), "b": jnp.array([4, 5, 6])}, True),
        ({"a": jnp.array([1, 2, 3]), "b": jnp.array([4, jnp.nan, 6])}, True),
        ({"a": jnp.array([1, 2, 3]), "b": {"c": jnp.array([4, 5, 6])}}, False),
        ({"a": jnp.array([1, 2, 3]), "b": {"c": jnp.array([4, jnp.nan, 6])}}, True),
        ({"a": jnp.array([]), "b": jnp.array([])}, False),
        ({"a": [1, 2, 3], "b": (4, 5, 6)}, False),
    ],
)
def test_pytree_has_nans(pytree: PyTree, expected: bool) -> None:
    assert xax.pytree_has_nans(pytree) == expected


def test_reshuffle_pytree() -> None:
    """Test reshuffle_pytree with fixed PRNG key."""
    shuffle_test_data = make_shuffle_test_data()

    # Test reshuffling along the first dimension (num_envs=2)
    reshuffled_data = xax.reshuffle_pytree(shuffle_test_data, (2,), jax.random.key(42))

    # Pre-computed expected values for PRNGKey(42) and dimension size 2
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ],
            [
                [7, 8],
                [9, 10],
                [11, 12],
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [101, 102],
                [201, 202],
                [301, 302],
            ],
            [
                [401, 402],
                [501, 502],
                [601, 602],
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["actions"], expected_actions)

    # Test reshuffling along the first two dimensions (num_envs=2, num_timesteps=3)
    reshuffled_data = xax.reshuffle_pytree(shuffle_test_data, (2, 3), jax.random.key(43))

    # Pre-computed expected values for PRNGKey(43) and flattened dimension size 6
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [11, 12],  # Original (1, 2) data
                [9, 10],  # Original (1, 1) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [3, 4],  # Original (0, 1) data
                [5, 6],  # Original (0, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [601, 602],  # Original (1, 2) data
                [501, 502],  # Original (1, 1) data
            ],
            [
                [401, 402],  # Original (1, 0) data
                [201, 202],  # Original (0, 1) data
                [301, 302],  # Original (0, 2) data
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["actions"], expected_actions)


def test_reshuffle_pytree_independently() -> None:
    """Test reshuffle_pytree_independently with fixed PRNG key."""
    shuffle_test_data = make_shuffle_test_data()
    # Test reshuffling along the first two dimensions independently
    reshuffled_data = xax.reshuffle_pytree_independently(shuffle_test_data, (2, 3), jax.random.key(44))

    # Pre-computed expected values for PRNGKey(44)
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [9, 10],  # Original (1, 1) data
                [7, 8],  # Original (1, 0) data
                [11, 12],  # Original (1, 2) data
            ],
            [
                [3, 4],  # Original (0, 1) data
                [1, 2],  # Original (0, 0) data
                [5, 6],  # Original (0, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [501, 502],  # Original (1, 1) data
                [401, 402],  # Original (1, 0) data
                [601, 602],  # Original (1, 2) data
            ],
            [
                [201, 202],  # Original (0, 1) data
                [101, 102],  # Original (0, 0) data
                [301, 302],  # Original (0, 2) data
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["actions"], expected_actions)

    # Verify that the reshuffling is done independently for each dimension

    # Check that the same permutation is applied to both environments
    # Find the permutation for dimension 1 (timesteps)
    # For env 0 in the reshuffled data (which is env 1 in the original data)
    env0_perm = []
    for t in range(3):
        for orig_t in range(3):
            if jnp.array_equal(reshuffled_data["observations"][0, t], shuffle_test_data["observations"][1, orig_t]):
                env0_perm.append(orig_t)
                break

    # For env 1 in the reshuffled data (which is env 0 in the original data)
    env1_perm = []
    for t in range(3):
        for orig_t in range(3):
            if jnp.array_equal(reshuffled_data["observations"][1, t], shuffle_test_data["observations"][0, orig_t]):
                env1_perm.append(orig_t)
                break

    # The permutation should be the same for both environments
    assert env0_perm == env1_perm
    assert env0_perm == [1, 0, 2]


def test_reshuffle_pytree_along_dims() -> None:
    """Test reshuffle_pytree_along_dims with fixed PRNG key."""
    shuffle_test_data = make_shuffle_test_data()
    # Test reshuffling along the time dimension only (dimension 1)
    time_reshuffled_data = xax.reshuffle_pytree_along_dims(shuffle_test_data, (1,), (3,), jax.random.key(45))

    # Pre-computed expected values for PRNGKey(45)
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
                [5, 6],  # Original (0, 2) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
                [11, 12],  # Original (1, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [201, 202],  # Original (0, 1) data
                [301, 302],  # Original (0, 2) data
            ],
            [
                [401, 402],  # Original (1, 0) data
                [501, 502],  # Original (1, 1) data
                [601, 602],  # Original (1, 2) data
            ],
        ]
    )

    assert jnp.array_equal(time_reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(time_reshuffled_data["actions"], expected_actions)

    # Test reshuffling along both env and time dimensions
    env_time_reshuffled_data = xax.reshuffle_pytree_along_dims(shuffle_test_data, (0, 1), (2, 3), jax.random.key(46))

    # Pre-computed expected values for PRNGKey(46)
    # Based on actual output
    expected_env_time_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [9, 10],  # Original (1, 1) data
                [7, 8],  # Original (1, 0) data
            ],
            [
                [5, 6],  # Original (0, 2) data
                [11, 12],  # Original (1, 2) data
                [3, 4],  # Original (0, 1) data
            ],
        ]
    )

    expected_env_time_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [501, 502],  # Original (1, 1) data
                [401, 402],  # Original (1, 0) data
            ],
            [
                [301, 302],  # Original (0, 2) data
                [601, 602],  # Original (1, 2) data
                [201, 202],  # Original (0, 1) data
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(env_time_reshuffled_data["observations"], expected_env_time_obs)
    assert jnp.array_equal(env_time_reshuffled_data["actions"], expected_env_time_actions)


def test_compare_reshuffle_methods() -> None:
    """Compare the behavior of different reshuffle methods."""
    shuffle_test_data = make_shuffle_test_data()
    # Reshuffle using reshuffle_pytree
    reshuffled1 = xax.reshuffle_pytree(shuffle_test_data, (2, 3), jax.random.key(42))

    # Pre-computed expected values for PRNGKey(42) with reshuffle_pytree
    # Based on actual output
    expected_reshuffled1 = jnp.array(
        [
            [
                [9, 10],  # Original (1, 1) data
                [5, 6],  # Original (0, 2) data
                [11, 12],  # Original (1, 2) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
            ],
        ]
    )

    assert jnp.array_equal(reshuffled1["observations"], expected_reshuffled1)

    reshuffled2 = xax.reshuffle_pytree_along_dims(shuffle_test_data, (0, 1), (2, 3), jax.random.key(42))

    # Verify that both methods produce the same result when reshuffling the same dimensions
    # with the same key (since reshuffle_pytree_along_dims uses reshuffle_pytree internally)
    assert jnp.array_equal(reshuffled1["observations"], reshuffled2["observations"])

    # Compare with reshuffle_pytree_independently which should produce different results
    # since it uses a different reshuffling approach
    reshuffled3 = xax.reshuffle_pytree_independently(shuffle_test_data, (2, 3), jax.random.key(42))

    # Pre-computed expected values for PRNGKey(42) with reshuffle_pytree_independently
    # Based on actual output
    expected_reshuffled3 = jnp.array(
        [
            [
                [11, 12],  # Original (1, 2) data
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
            ],
            [
                [5, 6],  # Original (0, 2) data
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
            ],
        ]
    )

    assert jnp.array_equal(reshuffled3["observations"], expected_reshuffled3)
    assert not jnp.array_equal(reshuffled1["observations"], reshuffled3["observations"])

    # Verify that reshuffle_pytree_along_dims with only the time dimension
    # produces different results than reshuffling both dimensions
    reshuffled4 = xax.reshuffle_pytree_along_dims(shuffle_test_data, (1,), (3,), jax.random.key(42))

    # Pre-computed expected values for PRNGKey(42) with reshuffle_pytree_along_dims on dim 1
    # Based on actual output
    expected_reshuffled4 = jnp.array(
        [
            [
                [5, 6],  # Original (0, 2) data
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
            ],
            [
                [11, 12],  # Original (1, 2) data
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
            ],
        ]
    )

    assert jnp.array_equal(reshuffled4["observations"], expected_reshuffled4)

    assert not jnp.array_equal(reshuffled2["observations"], reshuffled4["observations"])

    # Verify that when reshuffling only the time dimension, the env dimension is preserved
    # For each environment, the set of timesteps should be the same before and after
    for env in range(2):
        orig_timesteps = jnp.sort(shuffle_test_data["observations"][env, :, 0])
        reshuffled_timesteps = jnp.sort(reshuffled4["observations"][env, :, 0])
        assert jnp.array_equal(orig_timesteps, reshuffled_timesteps)


def test_reshuffle_pytree_with_nested_dict() -> None:
    """Test reshuffling a deeply nested dictionary structure."""
    nested_dict_data = make_nested_dict_data()

    # Test reshuffling along the first dimension (num_envs=2)
    reshuffled_data = xax.reshuffle_pytree(nested_dict_data, (2,), jax.random.key(42))

    # Pre-computed expected values for PRNGKey(42) and dimension size 2
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
                [5, 6],  # Original (0, 2) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
                [11, 12],  # Original (1, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [103, 104],  # Original (0, 1) data
                [105, 106],  # Original (0, 2) data
            ],
            [
                [107, 108],  # Original (1, 0) data
                [109, 110],  # Original (1, 1) data
                [111, 112],  # Original (1, 2) data
            ],
        ]
    )

    expected_rewards = jnp.array(
        [
            [201, 202, 203],  # Original (0, :) data
            [204, 205, 206],  # Original (1, :) data
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data["level1"]["level2"]["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["level1"]["level2"]["actions"], expected_actions)
    assert jnp.array_equal(reshuffled_data["level1"]["level2"]["rewards"], expected_rewards)

    # Test reshuffling along the first two dimensions (num_envs=2, num_timesteps=3)
    reshuffled_data_2d = xax.reshuffle_pytree(nested_dict_data, (2, 3), jax.random.key(43))

    # Pre-computed expected values for PRNGKey(43) and flattened dimension size 6
    # Based on actual output
    expected_obs_2d = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [11, 12],  # Original (1, 2) data
                [9, 10],  # Original (1, 1) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [3, 4],  # Original (0, 1) data
                [5, 6],  # Original (0, 2) data
            ],
        ]
    )

    expected_actions_2d = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [111, 112],  # Original (1, 2) data
                [109, 110],  # Original (1, 1) data
            ],
            [
                [107, 108],  # Original (1, 0) data
                [103, 104],  # Original (0, 1) data
                [105, 106],  # Original (0, 2) data
            ],
        ]
    )

    expected_rewards_2d = jnp.array(
        [
            [201, 206, 205],  # Corresponding to the observations pattern
            [204, 202, 203],  # Corresponding to the observations pattern
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data_2d["level1"]["level2"]["observations"], expected_obs_2d)
    assert jnp.array_equal(reshuffled_data_2d["level1"]["level2"]["actions"], expected_actions_2d)
    assert jnp.array_equal(reshuffled_data_2d["level1"]["level2"]["rewards"], expected_rewards_2d)


def test_reshuffle_pytree_independently_with_nested_dict() -> None:
    """Test reshuffling a deeply nested dictionary structure independently."""
    nested_dict_data = make_nested_dict_data()

    # Reshuffle along the first two dimensions independently
    reshuffled_data = xax.reshuffle_pytree_independently(nested_dict_data, (2, 3), jax.random.key(44))

    # Pre-computed expected values for PRNGKey(44)
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [9, 10],  # Original (1, 1) data
                [7, 8],  # Original (1, 0) data
                [11, 12],  # Original (1, 2) data
            ],
            [
                [3, 4],  # Original (0, 1) data
                [1, 2],  # Original (0, 0) data
                [5, 6],  # Original (0, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [109, 110],  # Original (1, 1) data
                [107, 108],  # Original (1, 0) data
                [111, 112],  # Original (1, 2) data
            ],
            [
                [103, 104],  # Original (0, 1) data
                [101, 102],  # Original (0, 0) data
                [105, 106],  # Original (0, 2) data
            ],
        ]
    )

    expected_rewards = jnp.array(
        [
            [205, 204, 206],
            [202, 201, 203],
        ]
    )

    assert jnp.array_equal(reshuffled_data["level1"]["level2"]["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["level1"]["level2"]["actions"], expected_actions)
    assert jnp.array_equal(reshuffled_data["level1"]["level2"]["rewards"], expected_rewards)

    assert reshuffled_data["level1"]["level2"]["observations"].shape == (2, 3, 2)
    assert reshuffled_data["level1"]["level2"]["actions"].shape == (2, 3, 2)
    assert reshuffled_data["level1"]["level2"]["rewards"].shape == (2, 3)


def test_reshuffle_pytree_along_dims_with_nested_dict() -> None:
    """Test reshuffling a deeply nested dictionary structure along specific dimensions."""
    nested_dict_data = make_nested_dict_data()
    # Test reshuffling along the time dimension only (dimension 1)
    time_reshuffled_data = xax.reshuffle_pytree_along_dims(nested_dict_data, (1,), (3,), jax.random.key(45))

    # Pre-computed expected values for PRNGKey(45)
    # Based on actual output
    expected_time_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
                [5, 6],  # Original (0, 2) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
                [11, 12],  # Original (1, 2) data
            ],
        ]
    )

    expected_time_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [103, 104],  # Original (0, 1) data
                [105, 106],  # Original (0, 2) data
            ],
            [
                [107, 108],  # Original (1, 0) data
                [109, 110],  # Original (1, 1) data
                [111, 112],  # Original (1, 2) data
            ],
        ]
    )

    expected_time_rewards = jnp.array(
        [
            [201, 202, 203],  # Original (0, :) data
            [204, 205, 206],  # Original (1, :) data
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(time_reshuffled_data["level1"]["level2"]["observations"], expected_time_obs)
    assert jnp.array_equal(time_reshuffled_data["level1"]["level2"]["actions"], expected_time_actions)
    assert jnp.array_equal(time_reshuffled_data["level1"]["level2"]["rewards"], expected_time_rewards)

    # Test reshuffling along both env and time dimensions
    env_time_reshuffled_data = xax.reshuffle_pytree_along_dims(nested_dict_data, (0, 1), (2, 3), jax.random.key(46))

    # Pre-computed expected values for PRNGKey(46)
    # Based on actual output
    expected_env_time_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [9, 10],  # Original (1, 1) data
                [7, 8],  # Original (1, 0) data
            ],
            [
                [5, 6],  # Original (0, 2) data
                [11, 12],  # Original (1, 2) data
                [3, 4],  # Original (0, 1) data
            ],
        ]
    )

    expected_env_time_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [109, 110],  # Original (1, 1) data
                [107, 108],  # Original (1, 0) data
            ],
            [
                [105, 106],  # Original (0, 2) data
                [111, 112],  # Original (1, 2) data
                [103, 104],  # Original (0, 1) data
            ],
        ]
    )

    expected_env_time_rewards = jnp.array(
        [
            [201, 205, 204],
            [203, 206, 202],
        ]
    )

    assert jnp.array_equal(env_time_reshuffled_data["level1"]["level2"]["observations"], expected_env_time_obs)
    assert jnp.array_equal(env_time_reshuffled_data["level1"]["level2"]["actions"], expected_env_time_actions)
    assert jnp.array_equal(env_time_reshuffled_data["level1"]["level2"]["rewards"], expected_env_time_rewards)
