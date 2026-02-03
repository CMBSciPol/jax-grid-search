# Distributed Grid Search using JAX

[![Tests](https://github.com/CMBSciPol/jax-grid-search/actions/workflows/tests.yml/badge.svg)](https://github.com/CMBSciPol/jax-grid-search/actions/workflows/tests.yml)
[![Notebooks](https://img.shields.io/github/actions/workflow/status/CMBSciPol/jax-grid-search/notebooks.yml?logo=jupyter&label=notebooks)](https://github.com/CMBSciPol/jax-grid-search/actions/workflows/notebooks.yml)
[![Code Formatting](https://github.com/CMBSciPol/jax-grid-search/actions/workflows/formatting.yml/badge.svg)](https://github.com/CMBSciPol/jax-grid-search/actions/workflows/formatting.yml)
[![Upload Python Package](https://github.com/CMBSciPol/jax-grid-search/actions/workflows/python-publish.yml/badge.svg)](https://github.com/CMBSciPol/jax-grid-search/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/jax-grid-search.svg)](https://badge.fury.io/py/jax-grid-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://doi.org/10.5281/zenodo.17674777"><img src="https://zenodo.org/badge/917061582.svg" alt="DOI"></a>

## About

This package is designed to minimize likelihoods computed by [FURAX](https://github.com/CMBSciPol/furax), a JAX-based CMB analysis framework. It provides distributed grid search capabilities specifically optimized for:

- **Spatial spectral index variability:** Efficiently explore parameter spaces for spatially-varying spectral indices in foreground models
- **Foreground component optimization:** Test and compare different foreground component configurations to find the optimal model choice
- **Likelihood model optimization:** Systematically search through discrete model configurations

The distributed grid search is built to handle the computational demands of CMB likelihood analysis, leveraging JAX's performance and enabling efficient parallel exploration of discrete parameter spaces.

> **Note:** Continuous optimization features (formerly `optimize`) have been moved to [furax-cs](https://github.com/CMBSciPol/furax-cs). Please use `furax_cs.minimize` for gradient-based optimization.

---

This repository provides:

1. **Distributed Grid Search for Discrete Optimization:**
   Explore a parameter space by evaluating a user-defined objective function on a grid of discrete values. The search runs in parallel across available processes, automatically handling batching, progress tracking, and result aggregation.

---

## Getting Started

### Installation

Install the required dependencies via pip:

```bash
pip install jax_grid_search
```

---

##  Examples and Tutorials

For comprehensive tutorials and hands-on examples, see the **[examples directory](./examples/)** which contains:

- **Interactive Jupyter notebooks** covering basic to advanced concepts
- **Distributed computing examples** with MPI setup
- **Complete API demonstrations** with visualization

**Start here**: [Examples README](./examples/README.md) for guided learning paths.

---

## Usage Examples

### Distributed Grid Search (Discrete Optimization)

Define your objective function and parameter grid, then run a distributed grid search. The objective function must return a dictionary with a `"value"` key.

```python
import jax.numpy as jnp
from jax_grid_search import DistributedGridSearch

# Define a discrete objective function
def objective_fn(param1, param2):
    # Example: combine sine and cosine evaluations
    result = jnp.sin(param1) + jnp.cos(param2)
    return {"value": result}

# Define the search space (discrete values)
search_space = {
    "param1": jnp.linspace(0, 3.14, 10),
    "param2": jnp.linspace(0, 3.14, 10)
}

# Initialize and run the grid search
grid_search = DistributedGridSearch(
    objective_fn=objective_fn,
    search_space=search_space,
    progress_bar=True,     # Enable progress updates
    log_every=0.1,         # Log progress every 10%
    result_dir="results"   # Directory for intermediate results
)
grid_search.run()

# Retrieve the aggregated results
results = grid_search.stack_results("results")
print("Grid Search Results:", results)
```

#### Resuming a Grid Search

To resume a grid search from a previous checkpoint, simply load the results and pass them to the `DistributedGridSearch` constructor:

```python

results = grid_search.stack_results("results")

# Initialize and run the grid search
grid_search = DistributedGridSearch(
    objective_fn=objective_fn,
    search_space=search_space,
    progress_bar=True,     # Enable progress updates
    log_every=0.1,         # Log progress every 10%
    result_dir="results"   # Directory for intermediate results
    old_results=results    # Pass the previous results to resume the search
)
grid_search.run()
```

#### Running a distributed grid search

To run the grid search across multiple processes, use the mpirun (or srun):

```bash
mpirun -n 4 python grid_search_example.py
```

To run the following code in script

```python
import jax
jax.distributed.initialize()


# Initialize and run the grid search
grid_search = DistributedGridSearch(
    objective_fn=objective_fn,
    search_space=search_space,
    progress_bar=True,     # Enable progress updates
    log_every=0.1,         # Log progress every 10%
    result_dir="results"   # Directory for intermediate results
    old_results=results    # Pass the previous results to resume the search
)
grid_search.run()
```

You need to make sure that the number of combinitions in the search space is divisible by the number of processes.

#### Vectorized Strategy

For element-wise parameter pairing instead of full Cartesian products, use the `"vectorized"` strategy:

```python
# All parameter arrays must have the same length for vectorized strategy
search_space = {
    "learning_rate": jnp.array([0.01, 0.1, 0.5]),     # 3 values
    "batch_size": jnp.array([32, 64, 128]),           # 3 values
    "dropout": jnp.array([0.1, 0.2, 0.3])             # 3 values
}

# This creates 3 combinations: (0.01,32,0.1), (0.1,64,0.2), (0.5,128,0.3)
grid_search = DistributedGridSearch(
    objective_fn=objective_fn,
    search_space=search_space,
    strategy="vectorized"  # Use vectorized instead of cartesian
)
```

#### Multi-dimensional Parameters

The library supports multi-dimensional parameter arrays, where each parameter can be a matrix or tensor instead of a scalar. This is useful for optimizing structured parameters like filter kernels, weight matrices, or spatial configurations:

```python
# Each parameter is a set of 2D matrices to be optimized
search_space = {
    "kernel": jnp.array([
        [[1.0, 0.5], [0.0, 1.0]],    # 2x2 edge detection kernel
        [[-1.0, 0.0], [0.0, -1.0]],  # 2x2 negative edge kernel
        [[0.5, 0.5], [0.5, 0.5]]     # 2x2 smoothing kernel
    ]),
    "bias_matrix": jnp.array([
        [[0.1, 0.1], [0.1, 0.1]],    # 2x2 uniform bias
        [[0.0, 0.2], [0.2, 0.0]],    # 2x2 diagonal bias
        [[0.05, 0.15], [0.15, 0.05]] # 2x2 gradient bias
    ])
}

def image_filter_objective(kernel, bias_matrix):
    """Objective function with 2D matrix parameters."""
    response = kernel**2 - bias_matrix**2
    return {"value": response.sum()}  # Scalar output for optimization
```

**Result Sorting**:
- For scalar outputs: Results sorted by objective value (ascending)
- For multi-dimensional outputs: Results sorted by mean of all output elements

See [02-advanced-grid-search.ipynb](./examples/02-advanced-grid-search.ipynb) for complete examples with visualization.

## Citation

```
@software{Kabalan_JAX_Distributed_Grid_2025,
          author = {Kabalan, Wassim},
          month = apr,
          title = {{JAX Distributed Grid Search for Hyperparameter Tuning}},
          url = {https://github.com/CMBSciPol/jax-grid-search},
          version = {0.1.8},
          year = {2025}
}
```
