"""
JAX Grid Search - Distributed optimization library for JAX

This package provides two complementary optimization approaches:

1. **DistributedGridSearch**: Discrete parameter space exploration using distributed
   computing. Supports both cartesian and vectorized combination strategies, automatic
   memory management, resume functionality, and multi-dimensional parameters.

2. **optimize**: Continuous optimization using gradient-based methods via Optax.
   Includes convergence monitoring, parameter bounds, update history logging,
   and progress tracking via jax-progress package.

Main Components:
    DistributedGridSearch: Main class for parallel discrete parameter optimization
    optimize: Function for continuous optimization with various Optax optimizers
    condition: Function conditioning (parameter transforms + output normalization)

For progress tracking, use TqdmProgressMeter from jax_progress:
    >>> from jax_progress import TqdmProgressMeter

Example:
    >>> import jax.numpy as jnp
    >>> from jax_grid_search import DistributedGridSearch
    >>>
    >>> def objective(x, y):
    ...     return {"value": (x - 2)**2 + (y - 1)**2}
    >>>
    >>> search_space = {
    ...     "x": jnp.linspace(0, 4, 21),
    ...     "y": jnp.linspace(-1, 3, 21)
    ... }
    >>>
    >>> grid_search = DistributedGridSearch(objective, search_space)
    >>> grid_search.run()
    >>> results = grid_search.stack_results("results")

For comprehensive tutorials and examples, see the examples/ directory.
"""

from ._gridding import DistributedGridSearch
from ._optimizers import condition, default_desc_cb, optimize

__all__ = ["DistributedGridSearch", "optimize", "condition", "default_desc_cb"]
