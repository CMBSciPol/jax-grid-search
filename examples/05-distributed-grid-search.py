#!/usr/bin/env python3
"""
Distributed Grid Search Script

This script runs distributed grid search using MPI and saves results for analysis.
It's designed to be called from the companion Jupyter notebook.

Usage:
    mpirun -n 4 python 05-distributed-grid-search.py
    mpirun -n 8 python 05-distributed-grid-search.py
"""

import os
import sys
import time

# Set environment variables for better distributed performance
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
import jax.numpy as jnp
from jax_grid_search import DistributedGridSearch


def neural_network_objective(learning_rate, batch_size, dropout, num_layers, optimizer_type):
    """
    Neural network hyperparameter optimization objective.

    Simulates realistic interactions between hyperparameters.
    """
    # Learning rate should be balanced with batch size
    lr_batch_interaction = jnp.abs(jnp.log10(learning_rate) + jnp.log10(batch_size / 64.0))

    # Deeper networks need different dropout rates
    optimal_dropout = jnp.clip(0.1 + 0.05 * (num_layers - 2), 0.0, 0.6)
    dropout_penalty = (dropout - optimal_dropout)**2

    # Different optimizers work better with different learning rates
    optimizer_lr_penalty = jnp.where(
        optimizer_type == 0,  # SGD
        jnp.where(learning_rate > 0.1, (learning_rate - 0.1)**2, 0.0),
        jnp.where(
            optimizer_type == 1,  # Adam
            jnp.where(learning_rate > 0.001, (learning_rate - 0.001)**2 * 0.1, 0.0),
            0.0  # AdamW
        )
    )

    # Architecture complexity penalty
    complexity_penalty = 0.01 * jnp.maximum(0, num_layers - 8)**2

    # Batch size efficiency (powers of 2 are typically better)
    batch_efficiency = 0.1 * jnp.abs(jnp.log2(batch_size) - jnp.round(jnp.log2(batch_size)))

    # Combine all factors
    total_cost = (lr_batch_interaction +
                  dropout_penalty +
                  optimizer_lr_penalty +
                  complexity_penalty +
                  batch_efficiency)

    # Add some realistic noise
    noise = 0.01 * jnp.sin(123.45 * learning_rate + 67.89 * dropout + 42.0 * num_layers)

    return {
        "value": total_cost + noise,
        "lr_batch_interaction": lr_batch_interaction,
        "dropout_penalty": dropout_penalty,
        "optimizer_penalty": optimizer_lr_penalty,
        "complexity_penalty": complexity_penalty,
        "batch_efficiency": batch_efficiency,
        "noise_component": noise,
        "predicted_accuracy": 0.95 - 0.1 * (total_cost + noise)
    }


def create_search_space():
    """Create a realistic hyperparameter search space."""
    return {
        "learning_rate": jnp.logspace(-4, -1, 15),  # 1e-4 to 1e-1, 15 values
        "batch_size": jnp.array([16, 32, 48, 64, 96, 128, 192, 256]),  # 8 values
        "dropout": jnp.linspace(0.0, 0.6, 7),  # 7 values (0.0 to 0.6)
        "num_layers": jnp.arange(2, 9),  # 7 values (2 to 8 layers)
        "optimizer_type": jnp.array([0, 1, 2])  # 3 optimizers: SGD, Adam, AdamW
    }


def main():
    """Main function for distributed grid search."""
    try:
        # Initialize JAX distributed computing
        jax.distributed.initialize()

        rank = jax.process_index()
        total_processes = jax.process_count()

        # Create search space
        search_space = create_search_space()

        # Calculate total combinations
        total_combinations = 1
        for values in search_space.values():
            total_combinations *= len(values)

        if rank == 0:
            print(f"Distributed Grid Search Started")
            print(f"Total processes: {total_processes}")
            print(f"Total combinations: {total_combinations:,}")
            print(f"Combinations per process: ~{total_combinations // total_processes:,}")

        # Set up distributed grid search
        result_dir = "distributed_results"

        grid_search = DistributedGridSearch(
            objective_fn=neural_network_objective,
            search_space=search_space,
            batch_size=128,
            progress_bar=(rank == 0),  # Only show progress bar on rank 0
            result_dir=result_dir
        )

        # Run the distributed grid search
        start_time = time.time()
        grid_search.run()
        elapsed_time = time.time() - start_time

        if rank == 0:
            print(f"Distributed grid search completed in {elapsed_time:.1f} seconds")
            print(f"Results saved to '{result_dir}/' directory")
            print("Ready for result analysis in notebook!")

    except Exception as e:
        rank = getattr(jax, 'process_index', lambda: '?')()
        print(f"ERROR in process {rank}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()