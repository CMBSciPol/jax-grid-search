import os
import shutil
from typing import Callable, Iterator

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from jax_grid_search import DistributedGridSearch


# Automatically clean up the "results" directory before and after each test.
@pytest.fixture(autouse=True)
def clean_results_dir() -> Iterator[None]:
    results_dir = "results"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    yield
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)


# Fixture for the objective function used by grid search.
# This fixture is parameterized to yield either:
#   - A function that returns a dict with a scalar value ("scalar")
#   - A function that returns a dict with a non-scalar value ("non-scalar")
@pytest.fixture(params=[True, False], ids=["scalar", "non-scalar"])
def objective_function(request) -> Callable[[Array, Array, Array, Array], dict[str, Array]]:
    # Scalar case: returns a dictionary with a scalar value.
    def objective_function(
        x: Array,
        y: Array,
        z: Array,
        w: Array,
    ) -> dict[str, Array]:
        value = x**2 + y**2 + z**2 - w**2
        if request.param == "scalar":
            return {"value": value.sum()}
        else:
            return {"value": value}

    return objective_function


# Fixture for the search space with parameterization.
# For each parameter value p in [1, 2, 3], each array is built as:
#   jnp.arange(4 * p).reshape(-1, p)
# which yields 4 rows and p columns.
@pytest.fixture(params=[1, 2])
def search_space(request) -> dict[str, Array]:
    return {
        "x": jnp.arange(4 * request.param).reshape(-1, request.param).squeeze(),
        "y": jnp.arange(4 * request.param).reshape(-1, request.param).squeeze(),
        "z": jnp.arange(4 * request.param).reshape(-1, request.param).squeeze(),
        "w": jnp.arange(4 * request.param).reshape(-1, request.param).squeeze(),
    }


# Fixture for an updated search space that depends on the parameterized search_space.
# It uses the same number of columns p but creates arrays with 5 rows (i.e. total elements = 5*p).
@pytest.fixture
def updated_search_space(search_space: dict[str, Array]) -> dict[str, Array]:
    p = search_space["x"].shape[0]
    dim = search_space["x"].ndim
    new_total = 2 * p
    return {
        "x": jnp.arange(new_total * dim).reshape(-1, dim).squeeze(),
        "y": jnp.arange(p * dim).reshape(-1, dim).squeeze(),
        "z": jnp.arange(p * dim).reshape(-1, dim).squeeze(),
        "w": jnp.arange(p * dim).reshape(-1, dim).squeeze(),
    }


def test_grid_search(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
) -> None:
    grid_search = DistributedGridSearch(objective_function, search_space, batch_size=8, progress_bar=True, log_every=0.1)
    grid_search.run()

    results = grid_search.stack_results("results")
    values = results["value"]

    # For each returned value, reduce it to a scalar if necessary.
    reduced_values = values.mean(axis=tuple(range(1, values.ndim)))

    # Assert that the first (reduced) value is the minimum.
    assert reduced_values[0] == jnp.min(reduced_values)

    best_x = results["x"][0]
    best_y = results["y"][0]
    best_z = results["z"][0]
    best_w = results["w"][0]

    expected_value = objective_function(best_x, best_y, best_z, best_w)["value"]
    assert (values[0] == expected_value).all()


def test_resume(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
    updated_search_space: dict[str, Array],
) -> None:
    # First run with the initial search space.
    grid_search = DistributedGridSearch(objective_function, search_space, batch_size=8, progress_bar=True, log_every=0.1)
    expected_n_combinations = jax.tree.reduce(lambda x, y: x * y.shape[0], search_space, 1)
    assert grid_search.n_combinations == expected_n_combinations

    grid_search.run()
    full_results = grid_search.stack_results("results")

    # Now resume with the same search space.
    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=8,
        progress_bar=True,
        log_every=0.1,
        old_results=full_results,
    )
    # Since all combinations have been processed, we expect 0 remaining.
    assert grid_search.n_combinations == 0

    # Resume with an updated search space.
    new_expected_n_combinations = jax.tree.reduce(lambda x, y: x * y.shape[0], updated_search_space, 1)
    remaining_combinations = new_expected_n_combinations - expected_n_combinations

    grid_search = DistributedGridSearch(
        objective_function,
        updated_search_space,
        batch_size=8,
        progress_bar=True,
        log_every=0.1,
        old_results=full_results,
    )
    print(f"remaining_combinations: {remaining_combinations}")
    print(f"grid_search.n_combinations: {grid_search.n_combinations}")
    assert grid_search.n_combinations == remaining_combinations

    grid_search.run()

    full_results = grid_search.stack_results("results")
    grid_search = DistributedGridSearch(
        objective_function,
        search_space,
        batch_size=8,
        progress_bar=True,
        log_every=0.1,
        old_results=full_results,
    )
    # After resuming, there should be no remaining combinations.
    assert grid_search.n_combinations == 0


def test_suggest_batch(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
) -> None:
    if jax.devices()[0].platform == "cpu":
        pytest.skip("Test only works for GPU devices")
    grid_search = DistributedGridSearch(objective_function, search_space, batch_size=None, progress_bar=True, log_every=0.1)

    max_size = grid_search.suggest_batch_size()

    memory_stats = jax.devices()[0].memory_stats()
    max_device_memory = memory_stats["bytes_limit"] - memory_stats["bytes_in_use"]

    # Size of one call.
    sample_params = jax.tree.map(lambda x: x[0], search_space)
    compiled = (
        jax.jit(objective_function)
        .lower(
            sample_params["x"],
            sample_params["y"],
            sample_params["z"],
            sample_params["w"],
        )
        .compile()
    )
    mem_analysis = compiled.memory_analysis()

    one_call_mem = mem_analysis.argument_size_in_bytes + mem_analysis.output_size_in_bytes + mem_analysis.temp_size_in_bytes

    assert (max_device_memory / one_call_mem) - max_size < 0.5


def test_bad_objective_fn(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
    search_space: dict[str, Array],
) -> None:
    def bad_objective_fn(
        x: Array,
        y: Array,
        z: Array,
        w: Array,
    ) -> Array:
        good_res = objective_function(x, y, z, w)
        return good_res["value"]  # Return only the value and not the dict

    grid_search = DistributedGridSearch(bad_objective_fn, search_space, batch_size=8, progress_bar=True, log_every=0.1)
    with pytest.raises(KeyError):
        grid_search.run()

    def no_val_objective_fn(
        x: Array,
        y: Array,
        z: Array,
        w: Array,
    ) -> dict[str, Array]:
        good_res = objective_function(x, y, z, w)
        return {"not_value": good_res["value"]}  # Return an unexpected key

    grid_search = DistributedGridSearch(no_val_objective_fn, search_space, batch_size=8, progress_bar=True, log_every=0.1)

    with pytest.raises(KeyError):
        grid_search.run()


def test_vectorized_basic(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
) -> None:
    """Test basic vectorized functionality with equal-length parameter arrays."""
    search_space = {
        "x": jnp.array([1.0, 2.0, 3.0]),
        "y": jnp.array([0.5, 1.5, 2.5]),
        "z": jnp.array([0.0, 1.0, 2.0]),
        "w": jnp.array([2.0, 3.0, 4.0]),
    }

    grid_search = DistributedGridSearch(
        objective_function, search_space, batch_size=8, progress_bar=True, log_every=0.1, strategy="vectorized"
    )

    # Should have exactly 3 combinations: (1,0.5,0,2), (2,1.5,1,3), (3,2.5,2,4)
    assert grid_search.n_combinations == 3

    grid_search.run()
    results = grid_search.stack_results("results")

    assert results is not None
    assert len(results["x"]) == 3

    # Verify the combinations are paired correctly
    expected_combinations = [(1.0, 0.5, 0.0, 2.0), (2.0, 1.5, 1.0, 3.0), (3.0, 2.5, 2.0, 4.0)]
    actual_combinations = list(zip(results["x"], results["y"], results["z"], results["w"]))

    # Results are sorted by value, so we need to verify all expected combinations exist
    for expected in expected_combinations:
        assert expected in actual_combinations


def test_vectorized_validation() -> None:
    """Test that vectorized strategy validates equal array lengths."""

    def dummy_fn(x: Array, y: Array) -> dict[str, Array]:
        return {"value": x + y}

    # Test invalid strategy
    with pytest.raises(ValueError, match="Invalid strategy 'invalid'"):
        DistributedGridSearch(dummy_fn, {"x": jnp.array([1.0])}, strategy="invalid")

    # Test with mismatched lengths
    search_space_mismatch = {
        "x": jnp.array([1.0, 2.0, 3.0]),  # length 3
        "y": jnp.array([0.5, 1.5]),  # length 2
    }

    with pytest.raises(ValueError, match="all parameter arrays must have the same length"):
        DistributedGridSearch(dummy_fn, search_space_mismatch, strategy="vectorized")

    # Test with matching lengths should work
    search_space_match = {
        "x": jnp.array([1.0, 2.0, 3.0]),
        "y": jnp.array([0.5, 1.5, 2.5]),
    }

    grid_search = DistributedGridSearch(dummy_fn, search_space_match, strategy="vectorized")
    assert grid_search.n_combinations == 3


def test_vectorized_vs_cartesian(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
) -> None:
    """Test that vectorized strategy produces different results than cartesian strategy."""
    search_space = {
        "x": jnp.array([1.0, 2.0]),
        "y": jnp.array([0.5, 1.5]),
        "z": jnp.array([0.0, 1.0]),
        "w": jnp.array([2.0, 3.0]),
    }

    # Cartesian strategy should have 2^4 = 16 combinations
    grid_search_cartesian = DistributedGridSearch(objective_function, search_space, batch_size=8, progress_bar=False, strategy="cartesian")
    assert grid_search_cartesian.n_combinations == 16

    # Vectorized strategy should have 2 combinations
    grid_search_vectorized = DistributedGridSearch(
        objective_function, search_space, batch_size=8, progress_bar=False, strategy="vectorized"
    )
    assert grid_search_vectorized.n_combinations == 2


def test_vectorized_resume(
    objective_function: Callable[[Array, Array, Array, Array], dict[str, Array]],
) -> None:
    """Test resume functionality with vectorized strategy."""
    search_space = {
        "x": jnp.array([1.0, 2.0, 3.0]),
        "y": jnp.array([0.5, 1.5, 2.5]),
        "z": jnp.array([0.0, 1.0, 2.0]),
        "w": jnp.array([2.0, 3.0, 4.0]),
    }

    # First run
    grid_search = DistributedGridSearch(objective_function, search_space, batch_size=8, progress_bar=False, strategy="vectorized")
    grid_search.run()
    results = grid_search.stack_results("results")

    # Resume with same search space should have 0 combinations left
    grid_search_resume = DistributedGridSearch(
        objective_function, search_space, batch_size=8, progress_bar=False, strategy="vectorized", old_results=results
    )
    assert grid_search_resume.n_combinations == 0


def test_vectorized_edge_cases() -> None:
    """Test edge cases for vectorized strategy."""

    def simple_fn(x: Array) -> dict[str, Array]:
        return {"value": x**2}

    # Single element arrays
    search_space_single = {
        "x": jnp.array([1.0]),
    }

    grid_search = DistributedGridSearch(simple_fn, search_space_single, strategy="vectorized")
    assert grid_search.n_combinations == 1

    # Empty arrays should work but produce 0 combinations
    search_space_empty = {
        "x": jnp.array([]),
    }

    grid_search_empty = DistributedGridSearch(simple_fn, search_space_empty, strategy="vectorized")
    assert grid_search_empty.n_combinations == 0
