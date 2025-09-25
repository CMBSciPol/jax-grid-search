# JAX Grid Search Examples

This directory contains comprehensive examples demonstrating both distributed grid search and continuous optimization capabilities.

## 📋 Overview

The examples are organized to progressively build understanding from basic concepts to advanced techniques:

### Grid Search Examples
- **[01-basic-grid-search.ipynb](./01-basic-grid-search.ipynb)** - Fundamental grid search concepts and usage
- **[02-advanced-grid-search.ipynb](./02-advanced-grid-search.ipynb)** - Advanced features like vectorized strategy and resuming

### Continuous Optimization Examples
- **[03-basic-optimization.ipynb](./03-basic-optimization.ipynb)** - Getting started with continuous optimization
- **[04-advanced-optimization.ipynb](./04-advanced-optimization.ipynb)** - Advanced optimization techniques and debugging

### Distributed Computing
- **[05-distributed-grid-search.ipynb](./05-distributed-grid-search.ipynb)** - Multi-process grid search with MPI
- **[05-distributed-grid-search.py](./05-distributed-grid-search.py)** - Companion Python script for MPI execution

## 🚀 Quick Start

### Prerequisites

```bash
# Install the package
pip install jax_grid_search

# For distributed examples, install MPI
# Ubuntu/Debian: sudo apt-get install mpich or libopenmpi-dev
# Or use your HPC cluster's MPI implementation
```

### Running Examples

**Jupyter Notebooks:**
```bash
# Launch Jupyter in the examples directory
cd examples
jupyter lab
```

**Distributed Examples:**
```bash
cd examples
mpirun -n 4 python 05-distributed-grid-search.py
# Or use srun instead of mpirun depending on your HPC cluster
```

## 📚 Example Contents

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
- **Progress tracking** with ProgressBar integration
- **Convergence monitoring** with tolerance and iteration limits
- **Parameter bounds** using box constraints
- **Result visualization** and optimization trajectories

### 04-advanced-optimization.ipynb
- **Update history logging** with `log_updates=True` and analysis plots
- **Parallel optimization** using `jax.vmap` for multiple problems
- **Progress tracking** multiple concurrent optimizations with unique IDs
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

### Memory Management
- Use `memory_limit` parameter for automatic GPU batch sizing
- Monitor memory usage during large grid searches
- Consider checkpointing for long-running searches

### Performance Optimization
- Enable JIT compilation for objective functions when possible
- Use appropriate batch sizes based on your hardware
- Consider distributed execution for large parameter spaces

### Distributed Computing
- Ensure all processes can access the result directory
- Use appropriate MPI implementations for your cluster
- Monitor load balancing across processes
