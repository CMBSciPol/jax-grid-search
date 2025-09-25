# Distributed Grid Search & Continuous Optimization using JAX

[![Testing](https://github.com/ASKabalan/jax-grid-search/actions/workflows/tests.yml/badge.svg)](https://github.com/ASKabalan/jax-grid-search/actions/workflows/tests.yml)
[![Code Formatting](https://github.com/ASKabalan/jax-grid-search/actions/workflows/formatting.yml/badge.svg)](https://github.com/ASKabalan/jax-grid-search/actions/workflows/formatting.yml)
[![Upload Python Package](https://github.com/ASKabalan/jax-grid-search/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ASKabalan/jax-grid-search/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/jax-grid-search.svg)](https://badge.fury.io/py/jax-grid-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides two complementary optimization tools:

1. **Distributed Grid Search for Discrete Optimization:**
   Explore a parameter space by evaluating a user-defined objective function on a grid of discrete values. The search runs in parallel across available processes, automatically handling batching, progress tracking, and result aggregation.

2. **Continuous Optimization with Optax:**
   Minimize continuous functions using gradient-based methods (such as LBFGS). This routine leverages Optax for iterative parameter updates and includes built-in progress monitoring.

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

- **5 interactive Jupyter notebooks** covering basic to advanced concepts
- **Distributed computing examples** with MPI setup
- **Complete API demonstrations** with visualization

**Start here**: [Examples README](./examples/README.md) for guided learning paths.

---

## Usage Examples

### 1. Distributed Grid Search (Discrete Optimization)

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

### 2. Continuous Optimization using Optax

Use the continuous optimization routine to minimize a function with gradient-based methods (e.g., LBFGS). The example below minimizes a simple quadratic function.

```python
import jax.numpy as jnp
import optax
from jax_grid_search import optimize , ProgressBar

# Define a continuous objective function (e.g., quadratic)
def quadratic(x):
    return jnp.sum((x - 3.0) ** 2)

# Initial parameters and an optimizer (e.g., LBFGS)
init_params = jnp.array([0.0])
optimizer = optax.lbfgs()

with ProgressBar() as p:
    # Run continuous optimization with progress monitoring (optional)
    best_params, opt_state = optimize(
        init_params,
        quadratic,
        opt=optimizer,
        max_iter=50,
        tol=1e-10,
        progress=p  # Replace with a ProgressBar instance for visual updates if desired
)

print("Optimized Parameters:", best_params)
```

#### Using Different Optimizers

The library supports various Optax optimizers beyond LBFGS:

```python
import optax
from jax_grid_search import optimize, ProgressBar

def rosenbrock(x):
    # Classic optimization test function
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

init_params = jnp.array([-1.0, 1.0])

# Try different optimizers
optimizers = {
    "LBFGS": optax.lbfgs(),
    "Adam": optax.adam(learning_rate=0.01),
    "SGD": optax.sgd(learning_rate=0.1),
    "RMSprop": optax.rmsprop(learning_rate=0.01)
}

with ProgressBar() as p:
    for name, optimizer in optimizers.items():
        result, state = optimize(
            init_params, rosenbrock, optimizer,
            max_iter=1000, tol=1e-8, progress=p
        )
        print(f"{name}: {result}, final value: {rosenbrock(result)}")
```

#### Parameter Bounds and Constraints

Use box constraints to limit parameter values during optimization:

```python
# Constrain parameters to [0, 10] range
lower_bounds = jnp.array([0.0, 0.0])
upper_bounds = jnp.array([10.0, 10.0])

with ProgressBar() as p:
    result, state = optimize(
        init_params,
        objective_function,
        optax.adam(0.1),
        max_iter=100,
        tol=1e-6,
        progress=p,
        lower_bound=lower_bounds,
        upper_bound=upper_bounds
    )
```

#### Update History and Debugging

Track optimization progress for analysis and debugging:

```python
with ProgressBar() as p:
    result, state = optimize(
        init_params,
        objective_function,
        optax.lbfgs(),
        max_iter=100,
        tol=1e-8,
        progress=p,
        log_updates=True  # Enable update history logging
    )

# Plot optimization history
import matplotlib.pyplot as plt
if state.update_history is not None:
    history = state.update_history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history[:, 0])
    plt.ylabel('Update Norm')
    plt.xlabel('Iteration')
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.plot(history[:, 1])
    plt.ylabel('Objective Value')
    plt.xlabel('Iteration')
    plt.show()
```

#### Running multiple optimization tasks with vmap

You can run multiple optimization tasks in parallel using `jax.vmap`. This is useful when optimizing multiple functions or parameters simultaneously.

(This is very usefull for simulating multiple noise realizations for example)

You can use `progress_id` to track the progress of each optimization task running in parallel.

```python
import jax
import jax.numpy as jnp
import optax

# Define multiple objective functions
def objective_fn(x , normal):
    return jnp.sum(((x - 3.0) ** 2) + normal)

with ProgressBar() as p:

    def solve_one(seed):
        init_params = jnp.array([0.0])
        normal = jax.random.normal(jax.random.PRNGKey(seed), init_params.shape)
        optimizer = optax.lbfgs()
        # Run continuous optimization with progress monitoring (optional)
        best_params, opt_state = optimize(
            init_params,
            objective_fn,
            opt=optimizer,
            max_iter=50,
            tol=1e-4,
            progress=p,
            progress_id=seed,
            normal=normal
        )

        return best_params

    jax.vmap(solve_one)(jnp.arange(10))

```

**Distributed Execution:**
- Ensure the number of parameter combinations is reasonable for the number of processes
- Use `jax.distributed.initialize()` before creating the grid search
- Check that all processes can access the same result directory

### 4. Optimizing Likelihood parameters and models

You can use the continuous optimization to optimize the parameters of a model that is defined in a function.
For performance purposes, you need to make sure that the discrete parameters that can control the likelihood model can be jitted (using `lax.cond` for example or other lax control flow functions).

## Citation

@misc{kabalan2025jaxgridsearch,
  author       = {Kabalan, Wassim},
  title        = {JAX Distributed Grid Search for Hyperparameter Tuning},
  year         = {2025},
  version      = {0.1.5},
  howpublished = {\url{https://github.com/asKabalan/jax-grid-search}},
  note         = {Accessed: 2025-04-08}
}
