"""
JAX Grid Search - Distributed optimization library for JAX

This package provides:

1. **DistributedGridSearch**: Discrete parameter space exploration using distributed
   computing. Supports both cartesian and vectorized combination strategies, automatic
   memory management, resume functionality, and multi-dimensional parameters.

2. **optimize**: DEPRECATED. Please use `furax_cs.minimize` instead.

Main Components:
    DistributedGridSearch: Main class for parallel discrete parameter optimization
    optimize: Deprecated function (raises NotImplementedError)

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
from ._optimizers import optimize

__all__ = ["DistributedGridSearch", "optimize"]
