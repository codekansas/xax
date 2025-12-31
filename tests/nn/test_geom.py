"""Tests geometry functions."""

import math

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import xax


def test_quat_to_euler() -> None:
    quat = jnp.array([1, 0, 0, 0])
    euler = xax.quat_to_euler(quat)
    assert jnp.allclose(euler, jnp.array([0, 0, 0]))


def test_euler_to_quat() -> None:
    euler = jnp.array([0, 0, 0])
    quat = xax.euler_to_quat(euler)
    assert jnp.allclose(quat, jnp.array([1, 0, 0, 0]))


def test_rotation_equivalence() -> None:
    """Test that Euler angles and quaternions represent the same rotation."""
    rng = jax.random.key(0)
    euler = jax.random.uniform(rng, (10, 3), minval=-jnp.pi, maxval=jnp.pi)
    quat = xax.euler_to_quat(euler)
    euler_again = xax.quat_to_euler(quat)
    quat_again = xax.euler_to_quat(euler_again)

    # Quaternions q and -q represent the same rotation
    # We'll compare the absolute dot product which should be close to 1
    dot_product = jnp.abs(jnp.sum(quat * quat_again, axis=-1))
    assert jnp.allclose(dot_product, 1.0, atol=1e-5)


def test_quat_to_euler_special_cases() -> None:
    """Test quaternion to Euler conversion for special cases."""
    # Test gimbal lock case (pitch = 90 degrees)
    # This is approximately a rotation of 90 degrees around Y axis
    gimbal_lock_quat = jnp.array([0.7071, 0, 0.7071, 0])
    euler = xax.quat_to_euler(gimbal_lock_quat)

    # Pitch should be close to pi/2
    assert jnp.isclose(euler[1], jnp.pi / 2, atol=1e-2)

    # Test non-normalized quaternion
    non_normalized = jnp.array([2.0, 1.0, 0.5, 0.3])
    euler = xax.quat_to_euler(non_normalized)

    # Should still work and give valid Euler angles
    normalized_euler = xax.quat_to_euler(non_normalized / jnp.linalg.norm(non_normalized))
    assert jnp.allclose(euler, normalized_euler)


def test_euler_to_quat_batch() -> None:
    """Test batch processing of Euler angles to quaternions."""
    # Test with a batch of Euler angles
    batch_euler = jnp.array(
        [
            [0, 0, 0],  # Identity
            [jnp.pi / 4, 0, 0],  # 45 deg roll
            [0, jnp.pi / 4, 0],  # 45 deg pitch
            [0, 0, jnp.pi / 4],  # 45 deg yaw
        ]
    )

    batch_quat = xax.euler_to_quat(batch_euler)

    # Expected quaternions for these rotations
    expected_quat = jnp.array(
        [
            [1, 0, 0, 0],  # Identity
            [jnp.cos(jnp.pi / 8), jnp.sin(jnp.pi / 8), 0, 0],  # 45 deg roll
            [jnp.cos(jnp.pi / 8), 0, jnp.sin(jnp.pi / 8), 0],  # 45 deg pitch
            [jnp.cos(jnp.pi / 8), 0, 0, jnp.sin(jnp.pi / 8)],  # 45 deg yaw
        ]
    )

    assert jnp.allclose(batch_quat, expected_quat, atol=1e-5)


@pytest.mark.parametrize(
    "vector, euler, expected",
    [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, math.pi / 2]), np.array([0.0, 1.0, 0.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, math.pi / 2, 0.0]), np.array([0.0, 0.0, -1.0])),
    ],
)
def test_rotate_vector_by_quat(
    vector: np.ndarray,
    euler: np.ndarray,
    expected: np.ndarray,
) -> None:
    vector_jnp = jnp.array(vector)
    euler_jnp = jnp.array(euler)
    expected_jnp = jnp.array(expected)
    quat = xax.euler_to_quat(euler_jnp)
    rotated_vector = xax.rotate_vector_by_quat(vector_jnp, quat)
    assert jnp.allclose(rotated_vector, expected_jnp, atol=1e-4)


def test_get_projected_gravity_vector_from_quat() -> None:
    """Test gravity vector projection from quaternion orientation."""
    # Identity quaternion (no rotation) - gravity should point down in local frame
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    gravity = xax.get_projected_gravity_vector_from_quat(identity_quat)
    assert jnp.allclose(gravity, jnp.array([0.0, 0.0, -9.81]), atol=1e-3)

    # 90 degree rotation around X-axis - gravity should point along -Y
    x90_quat = jnp.array([jnp.cos(jnp.pi / 4), jnp.sin(jnp.pi / 4), 0.0, 0.0])
    gravity = xax.get_projected_gravity_vector_from_quat(x90_quat)
    assert jnp.allclose(gravity, jnp.array([0.0, -9.81, 0.0]), atol=1e-3)

    # 90 degree rotation around Y-axis - gravity should point along X
    y90_quat = jnp.array([jnp.cos(jnp.pi / 4), 0.0, jnp.sin(jnp.pi / 4), 0.0])
    gravity = xax.get_projected_gravity_vector_from_quat(y90_quat)
    assert jnp.allclose(gravity, jnp.array([9.81, 0.0, 0.0]), atol=1e-3)

    # Test with batch of quaternions
    batch_quat = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Identity
            [jnp.cos(jnp.pi / 4), jnp.sin(jnp.pi / 4), 0.0, 0.0],  # 90 deg around X
            [jnp.cos(jnp.pi / 4), 0.0, jnp.sin(jnp.pi / 4), 0.0],  # 90 deg around Y
        ]
    )

    batch_gravity = jax.vmap(xax.get_projected_gravity_vector_from_quat)(batch_quat)
    expected_gravity = jnp.array([[0.0, 0.0, -9.81], [0.0, -9.81, 0.0], [9.81, 0.0, 0.0]])
    assert jnp.allclose(batch_gravity, expected_gravity, atol=1e-5)


def test_cubic_bezier_interpolation() -> None:
    """Test cubic bezier interpolation function."""
    # Test at start point (x=0)
    y_start = jnp.array(1.0)
    y_end = jnp.array(2.0)
    x = jnp.array(0.0)
    result = xax.cubic_bezier_interpolation(y_start, y_end, x)
    assert jnp.allclose(result, y_start)

    # Test at end point (x=1)
    x = jnp.array(1.0)
    result = xax.cubic_bezier_interpolation(y_start, y_end, x)
    assert jnp.allclose(result, y_end)

    # Test at midpoint (x=0.5)
    x = jnp.array(0.5)
    result = xax.cubic_bezier_interpolation(y_start, y_end, x)
    # At midpoint, the value should be between start and end
    assert result > y_start and result < y_end

    # Test with arrays
    y_start = jnp.array([1.0, 2.0, 3.0])
    y_end = jnp.array([2.0, 3.0, 4.0])
    x = jnp.array([0.0, 0.5, 1.0])
    result = xax.cubic_bezier_interpolation(y_start, y_end, x)
    expected = jnp.array([1.0, 2.5, 4.0])
    assert jnp.allclose(result, expected, atol=1e-5)


def test_quat_to_rotmat() -> None:
    """Test conversion from quaternion to rotation matrix."""
    # Identity quaternion should give identity matrix
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    rotmat = xax.quat_to_rotmat(identity_quat)
    assert jnp.allclose(rotmat, jnp.eye(3), atol=1e-5)

    # 90 degree rotation around Z-axis
    z90_quat = jnp.array([jnp.cos(jnp.pi / 4), 0.0, 0.0, jnp.sin(jnp.pi / 4)])
    rotmat = xax.quat_to_rotmat(z90_quat)
    expected = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert jnp.allclose(rotmat, expected, atol=1e-5)

    # Test with batch of quaternions
    batch_quat = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Identity
            [jnp.cos(jnp.pi / 4), 0.0, 0.0, jnp.sin(jnp.pi / 4)],  # 90 deg around Z
        ]
    )
    batch_rotmat = jax.vmap(xax.quat_to_rotmat)(batch_quat)
    expected_batch = jnp.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Identity
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # 90 deg around Z
        ]
    )
    assert jnp.allclose(batch_rotmat, expected_batch, atol=1e-5)


def test_rotation_matrix_to_quat() -> None:
    """Test conversion from rotation matrix to quaternion."""
    # Test with identity matrix
    rotmat = jnp.eye(3)
    quat = xax.rotation_matrix_to_quat(rotmat)
    assert jnp.allclose(quat, jnp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-5)

    # Test with 90 degree rotation around Z-axis
    rotmat = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    quat = xax.rotation_matrix_to_quat(rotmat)
    assert jnp.allclose(quat, jnp.array([jnp.cos(jnp.pi / 4), 0.0, 0.0, jnp.sin(jnp.pi / 4)]), atol=1e-5)

    # Test with batch of rotation matrices
    batch_rotmat = jnp.array([jnp.eye(3), rotmat])
    batch_quat = jax.vmap(xax.rotation_matrix_to_quat)(batch_rotmat)
    expected_batch = jnp.array([[1.0, 0.0, 0.0, 0.0], [jnp.cos(jnp.pi / 4), 0.0, 0.0, jnp.sin(jnp.pi / 4)]])
    assert jnp.allclose(batch_quat, expected_batch, atol=1e-5)


def test_rotation_matrix_to_quat_bijection() -> None:
    rng = jax.random.key(0)
    batch_size = 10
    quat = jax.random.normal(rng, (batch_size, 4))
    rotmat = xax.quat_to_rotmat(quat)
    quat_again = xax.rotation_matrix_to_quat(rotmat)
    rotmat_again = xax.quat_to_rotmat(quat_again)

    assert jnp.allclose(rotmat, rotmat_again, atol=1e-5)


def test_normalize() -> None:
    """Test vector normalization function."""
    # Test with a simple vector
    v = jnp.array([3.0, 4.0])
    normalized = xax.normalize(v)
    expected = jnp.array([0.6, 0.8])
    assert jnp.allclose(normalized, expected)

    # Test with zero vector (should handle eps)
    v = jnp.array([0.0, 0.0])
    normalized = xax.normalize(v)
    # Should not be NaN or Inf
    assert jnp.all(jnp.isfinite(normalized))

    # Test with batch of vectors
    v_batch = jnp.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
    normalized_batch = xax.normalize(v_batch, axis=-1)
    expected_batch = jnp.array([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]])
    assert jnp.allclose(normalized_batch, expected_batch)


def test_rotation6d_to_rotation_matrix() -> None:
    """Test conversion from 6D rotation representation to rotation matrix."""
    # Test with identity rotation
    r6d = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rotmat = xax.rotation6d_to_rotation_matrix(r6d)
    assert jnp.allclose(rotmat, jnp.eye(3), atol=1e-5)

    # Test with a known rotation (90 degrees around Z-axis)
    r6d = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    rotmat = xax.rotation6d_to_rotation_matrix(r6d)
    expected = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert jnp.allclose(rotmat, expected, atol=1e-5)

    # Test with batch of rotations
    r6d_batch = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Identity
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 90 deg around Z
        ]
    )
    rotmat_batch = xax.rotation6d_to_rotation_matrix(r6d_batch)
    expected_batch = jnp.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Identity
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],  # 90 deg around Z
        ]
    )
    assert jnp.allclose(rotmat_batch, expected_batch, atol=1e-5)


def test_rotation_matrix_to_rotation6d() -> None:
    """Test conversion from rotation matrix to 6D rotation representation."""
    # Test with identity matrix
    rotmat = jnp.eye(3)
    r6d = xax.rotation_matrix_to_rotation6d(rotmat)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert jnp.allclose(r6d, expected)

    # Test with a known rotation (90 degrees around Z-axis)
    rotmat = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    r6d = xax.rotation_matrix_to_rotation6d(rotmat)
    expected = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert jnp.allclose(r6d, expected)

    # Test with batch of rotation matrices
    rotmat_batch = jnp.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Identity
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],  # 90 deg around Z
        ]
    )
    r6d_batch = xax.rotation_matrix_to_rotation6d(rotmat_batch)
    expected_batch = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Identity
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 90 deg around Z
        ]
    )
    assert jnp.allclose(r6d_batch, expected_batch)


def test_rotation_conversion_roundtrip() -> None:
    """Test roundtrip conversion between rotation representations."""
    # Generate random rotation matrices
    rng = jax.random.key(0)
    batch_size = 10

    # Generate random 6D rotations
    r6d = jax.random.normal(rng, (batch_size, 6))

    # Convert to rotation matrix and back
    rotmat = xax.rotation6d_to_rotation_matrix(r6d)
    r6d_again = xax.rotation_matrix_to_rotation6d(rotmat)

    # Convert back to rotation matrix
    rotmat_again = xax.rotation6d_to_rotation_matrix(r6d_again)

    # The rotation matrices should be equivalent (represent the same rotation)
    # We can't compare the 6D representations directly as they're not unique
    assert jnp.allclose(rotmat, rotmat_again, atol=1e-5)

    # The rotation matrices should be orthogonal
    identity = jnp.eye(3)
    for i in range(batch_size):
        # R * R^T should be identity
        orthogonality = jnp.dot(rotmat[i], rotmat[i].T)
        assert jnp.allclose(orthogonality, identity, atol=1e-5)

        # det(R) should be 1
        det = jnp.linalg.det(rotmat[i])
        assert jnp.isclose(det, 1.0, atol=1e-5)


def test_quat_mul() -> None:
    """Test quaternion multiplication function."""
    # Identity quaternion (no rotation)
    quat_identity = jnp.array([1.0, 0.0, 0.0, 0.0])

    # Basic identity test: identity * identity = identity
    quat_result = xax.quat_mul(quat_identity, quat_identity)
    assert jnp.allclose(quat_result, quat_identity, atol=1e-6)
    assert jnp.isclose(jnp.linalg.norm(quat_result), 1.0, atol=1e-6)

    # 90-degree rotations around principal axes
    quat_x_90 = jnp.array([0.7071068, 0.7071068, 0.0, 0.0])  # 90° about X-axis
    quat_y_90 = jnp.array([0.7071068, 0.0, 0.7071068, 0.0])  # 90° about Y-axis
    quat_z_90 = jnp.array([0.7071068, 0.0, 0.0, 0.7071068])  # 90° about Z-axis

    # 90° X followed by 90° X should equal 180° X rotation
    quat_x_180 = xax.quat_mul(quat_x_90, quat_x_90)
    expected_quat_x_180 = jnp.array([0.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(quat_x_180, expected_quat_x_180, atol=1e-6)
    assert jnp.isclose(jnp.linalg.norm(quat_x_180), 1.0, atol=1e-6)

    # Test non-commutativity of quaternion multiplication
    quat_x_then_y = xax.quat_mul(quat_x_90, quat_y_90)
    quat_y_then_x = xax.quat_mul(quat_y_90, quat_x_90)
    assert not jnp.allclose(quat_x_then_y, quat_y_then_x, atol=1e-6)

    # Batch multiplication test
    quat_batch_lhs = jnp.stack([quat_identity, quat_x_90, quat_y_90])
    quat_batch_rhs = jnp.stack([quat_identity, quat_x_90, quat_z_90])
    quat_batch_result = xax.quat_mul(quat_batch_lhs, quat_batch_rhs)

    # Expected results from scalar multiplications
    expected_batch_result = jnp.stack(
        [
            xax.quat_mul(quat_identity, quat_identity),
            xax.quat_mul(quat_x_90, quat_x_90),
            xax.quat_mul(quat_y_90, quat_z_90),
        ]
    )
    assert jnp.allclose(quat_batch_result, expected_batch_result, atol=1e-6)

    # Ensure all batch outputs are unit quaternions
    quat_norms = jnp.linalg.norm(quat_batch_result, axis=1)
    assert jnp.allclose(quat_norms, 1.0, atol=1e-6)
