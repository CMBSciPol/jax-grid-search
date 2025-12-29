# JAX Grid Search Examples

This directory contains comprehensive examples demonstrating both distributed grid search and continuous optimization capabilities.

## Overview

The examples are organized to progressively build understanding from basic concepts to advanced techniques:

### Grid Search Examples
- **[01-basic-grid-search.ipynb](./01-basic-grid-search.ipynb)** - Fundamental grid search concepts and usage
- **[02-advanced-grid-search.ipynb](./02-advanced-grid-search.ipynb)** - Advanced features like vectorized strategy and resuming

### Continuous Optimization Examples
- **[03-basic-optimization.ipynb](./03-basic-optimization.ipynb)** - Getting started with continuous optimization
- **[04-advanced-optimization.ipynb](./04-advanced-optimization.ipynb)** - Advanced optimization techniques and debugging

### Function Conditioning
- **[06-function-conditioning.ipynb](./06-function-conditioning.ipynb)** - Parameter transformation and output normalization

### Distributed Computing
- **[05-distributed-grid-search.ipynb](./05-distributed-grid-search.ipynb)** - Multi-process grid search with MPI
- **[05-distributed-grid-search.py](./05-distributed-grid-search.py)** - Companion Python script for MPI execution

## Quick Start

### Prerequisites

```bash
# Install the package
pip install jax_grid_search

# For distributed examples, install MPI
# Ubuntu/Debian: sudo apt-get install mpich or libopenmpi-dev
# Or use your HPC cluster's MPI implementation
```

### Running Examples

**Distributed Examples:**
```bash
cd examples
mpirun -n 4 python 05-distributed-grid-search.py
# Or use srun instead of mpirun depending on your HPC cluster
```

## Example Contents

### 01-basic-grid-search.ipynb
- Creating objective functions with proper return format
- Defining parameter search spaces with `jnp.linspace` and `jnp.arange`
- Running grid search with automatic batch sizing
- Result aggregation and visualization with matplotlib
- Saving and loading intermediate results
- Understanding memory considerations

### 02-advanced-grid-search.ipynb
- **Vectorized strategy** for element-wise parameter pairing
- **Resume functionality** using `old_results` to continue interrupted searches
- **Memory management** with automatic and manual batch sizing
- **Multiple return values** from objective functions
- **Progress customization** with different logging frequencies

### 03-basic-optimization.ipynb
- Simple quadratic function optimization with LBFGS
- **Different optimizers**: LBFGS, Adam, SGD, RMSprop
- **Progress tracking** with `TqdmProgressMeter` from `jax_progress`
- **Convergence monitoring** with tolerance and iteration limits
- **Parameter bounds** using box constraints
- **Result visualization** and optimization trajectories

### 04-advanced-optimization.ipynb
- **Update history logging** with `log_updates=True` and analysis plots
- **Parallel optimization** using `jax.vmap` for multiple problems
- **Progress tracking** with automatic vmap handling via `jax_progress`
- **Custom optimizers** and Optax optimizer chains

### 05-distributed-grid-search.ipynb + .py
- **MPI setup** and JAX distributed initialization
- **Process distribution** and rank-based computation
- **Result aggregation** across multiple processes
- **Performance scaling** analysis and best practices
- **HPC cluster compatibility** (SLURM vs OpenMPI)

##  Best Practices

### Objective Function Design
- Always return a dictionary with a `"value"` key
- Use JAX-compatible operations (`jnp` instead of `np`)
- Avoid Python control flow (use `jax.lax.cond`, `jnp.where`)
- Consider numerical stability for optimization

### Parameter Conditioning
- Use min-max scaling when parameters have vastly different scales
- Apply log transforms for positive-only parameters (temperature, amplitude)
- Normalize output by problem size (e.g., number of pixels × components)
- Verify transformations with to_opt/from_opt roundtrips
