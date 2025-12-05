from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
    import jax._src.interpreters.ad

import jax.numpy as jnp
import numpy as np
import optax
import optax.tree_utils as otu

from jax_grid_search import ProgressBar, condition, optimize


# Define a simple quadratic function: f(x) = (x - 3)^2
def quadratic(x: "jax._src.interpreters.ad.JVPTracer") -> "jax._src.interpreters.ad.JVPTracer":
    return jnp.sum((x - 3.0) ** 2)


def test_optimize_quadratic() -> None:
    # Run the optimizer with non-verbose mode.

    with ProgressBar() as p:
        init_params = jnp.array([0.0])
        solver = optax.lbfgs()
        final_params, final_state = optimize(
            init_params,
            quadratic,
            solver,
            max_iter=50,
            tol=1e-4,
            progress=p,
        )

    # The minimum of (x-3)^2 is at x=3.0.
    np.testing.assert_allclose(final_params, jnp.array([3.0]), atol=1e-2)

    # The objective value should be near 0 at the minimum.
    final_value = quadratic(final_params)
    np.testing.assert_allclose(final_value, 0.0, atol=1e-2)

    # Check that the optimizer state shows that at least one iteration was performed.
    count = otu.tree_get(final_state.state, "count")
    assert count is not None and count > 0


# ======================================================================
# Tests for condition function
# ======================================================================


def test_condition_minmax_scaling() -> None:
    """Test min-max scaling with parameter bounds."""

    # Define a simple function with known minimum
    def simple_fn(params):
        x, y = params["x"], params["y"]
        return (x - 25.0) ** 2 + (y - 1.5) ** 2

    # Define bounds
    lower = {"x": 10.0, "y": 0.5}
    upper = {"x": 40.0, "y": 3.0}

    # Apply conditioning
    wrapped_fn, to_opt, from_opt = condition(simple_fn, lower=lower, upper=upper)

    # Test roundtrip transformation
    physical_params = {"x": 25.0, "y": 1.5}
    opt_params = to_opt(physical_params)
    recovered_params = from_opt(opt_params)

    np.testing.assert_allclose(recovered_params["x"], physical_params["x"], atol=1e-6)
    np.testing.assert_allclose(recovered_params["y"], physical_params["y"], atol=1e-6)

    # Test that optimization works in [0,1] space
    init_opt_params = to_opt({"x": 15.0, "y": 0.8})

    with ProgressBar() as p:
        final_opt_params, _ = optimize(
            init_opt_params,
            wrapped_fn,
            optax.lbfgs(),
            max_iter=100,
            tol=1e-6,
            progress=p,
        )

    # Convert back to physical space
    final_physical = from_opt(final_opt_params)

    # Check that we found the minimum
    np.testing.assert_allclose(final_physical["x"], 25.0, atol=1e-1)
    np.testing.assert_allclose(final_physical["y"], 1.5, atol=1e-1)

    # Verify metadata is attached
    assert wrapped_fn.mode == "bounds"
    assert wrapped_fn.lower == lower
    assert wrapped_fn.upper == upper


def test_condition_custom_transforms() -> None:
    """Test custom transformations (log transform)."""

    # Function with temperature parameter (positive only)
    def temp_fn(params):
        T = params["T"]
        # Minimum at T = 20
        return ((jnp.log(T) - jnp.log(20.0)) ** 2).squeeze()

    # Apply log transform
    transform_fn = {"T": jnp.log}
    inv_transform_fn = {"T": jnp.exp}

    wrapped_fn, to_opt, from_opt = condition(
        temp_fn,
        transform_fn=transform_fn,
        inv_transform_fn=inv_transform_fn,
    )

    # Test roundtrip
    physical_params = {"T": 15.0}
    opt_params = to_opt(physical_params)
    recovered_params = from_opt(opt_params)

    np.testing.assert_allclose(recovered_params["T"], physical_params["T"], atol=1e-6)

    # Optimize in transformed space
    init_opt_params = to_opt({"T": 10.0})

    with ProgressBar() as p:
        final_opt_params, _ = optimize(
            init_opt_params,
            wrapped_fn,
            optax.lbfgs(),
            max_iter=100,
            tol=1e-6,
            progress=p,
        )

    final_physical = from_opt(final_opt_params)

    # Should find T = 20
    np.testing.assert_allclose(final_physical["T"], 20.0, atol=1e-1)

    # Verify metadata
    assert wrapped_fn.mode == "custom"
    assert wrapped_fn.transform_fn == transform_fn
    assert wrapped_fn.inv_transform_fn == inv_transform_fn


def test_condition_output_normalization() -> None:
    """Test output normalization with factor."""

    # Function with large output values
    def large_output_fn(x):
        return (1000.0 * (x - 3.0) ** 2).squeeze()

    # Apply normalization
    factor = 1000.0
    wrapped_fn, to_opt, from_opt = condition(large_output_fn, factor=factor)

    # Check that output is scaled
    test_x = jnp.array([0.0])
    original_value = large_output_fn(test_x)
    wrapped_value = wrapped_fn(test_x)

    np.testing.assert_allclose(wrapped_value, original_value / factor, atol=1e-6)

    # Optimize with normalized function
    init_params = jnp.array([0.0])

    with ProgressBar() as p:
        final_params, _ = optimize(
            init_params,
            wrapped_fn,
            optax.lbfgs(),
            max_iter=100,
            tol=1e-6,
            progress=p,
        )

    # Should still find correct minimum
    np.testing.assert_allclose(final_params, jnp.array([3.0]), atol=1e-1)

    # Verify metadata
    assert wrapped_fn.factor == factor


def test_condition_identity_mode() -> None:
    """Test identity mode (no conditioning)."""

    def simple_fn(x):
        return (x - 3.0) ** 2

    # Call condition with no arguments
    wrapped_fn, to_opt, from_opt = condition(simple_fn)

    # Verify transformations are identity
    test_x = jnp.array([5.0])
    np.testing.assert_allclose(to_opt(test_x), test_x, atol=1e-10)
    np.testing.assert_allclose(from_opt(test_x), test_x, atol=1e-10)

    # Verify wrapped function is unchanged (except for factor=1.0)
    np.testing.assert_allclose(wrapped_fn(test_x), simple_fn(test_x), atol=1e-10)

    # Verify metadata
    assert wrapped_fn.mode == "identity"
    assert wrapped_fn.factor == 1.0


def test_condition_validation_errors() -> None:
    """Test validation errors for invalid parameter combinations."""

    def dummy_fn(x):
        return x**2

    # Error: both bounds and custom transforms
    lower = {"x": 0.0}
    upper = {"x": 1.0}
    transform_fn = {"x": jnp.log}
    inv_transform_fn = {"x": jnp.exp}

    try:
        condition(
            dummy_fn,
            lower=lower,
            upper=upper,
            transform_fn=transform_fn,
            inv_transform_fn=inv_transform_fn,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot specify both" in str(e)

    # Error: only lower bound specified
    try:
        condition(dummy_fn, lower=lower)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Must specify both lower and upper" in str(e)

    # Error: only transform_fn specified
    try:
        condition(dummy_fn, transform_fn=transform_fn)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Must specify both transform_fn and inv_transform_fn" in str(e)


def test_condition_pytree_parameters() -> None:
    """Test conditioning with pytree (dict) parameters."""

    # Function with multiple parameters
    def multi_param_fn(params):
        a = params["a"]
        b = params["b"]
        c = params["c"]
        return (a - 5.0) ** 2 + (b - 10.0) ** 2 + (c - 2.0) ** 2

    # Apply bounds to dict params
    lower = {"a": 0.0, "b": 5.0, "c": 0.0}
    upper = {"a": 10.0, "b": 15.0, "c": 5.0}

    wrapped_fn, to_opt, from_opt = condition(multi_param_fn, lower=lower, upper=upper)

    # Test that tree structure is preserved
    physical_params = {"a": 5.0, "b": 10.0, "c": 2.0}
    opt_params = to_opt(physical_params)

    assert isinstance(opt_params, dict)
    assert set(opt_params.keys()) == {"a", "b", "c"}

    recovered_params = from_opt(opt_params)
    np.testing.assert_allclose(recovered_params["a"], physical_params["a"], atol=1e-6)
    np.testing.assert_allclose(recovered_params["b"], physical_params["b"], atol=1e-6)
    np.testing.assert_allclose(recovered_params["c"], physical_params["c"], atol=1e-6)

    # Optimize
    init_opt_params = to_opt({"a": 1.0, "b": 7.0, "c": 1.0})

    with ProgressBar() as p:
        final_opt_params, _ = optimize(
            init_opt_params,
            wrapped_fn,
            optax.lbfgs(),
            max_iter=100,
            tol=1e-6,
            progress=p,
        )

    final_physical = from_opt(final_opt_params)

    # Check convergence to minimum
    np.testing.assert_allclose(final_physical["a"], 5.0, atol=1e-1)
    np.testing.assert_allclose(final_physical["b"], 10.0, atol=1e-1)
    np.testing.assert_allclose(final_physical["c"], 2.0, atol=1e-1)
