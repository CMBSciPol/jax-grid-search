from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jaxtyping import Array, PyTree

from ._progressbar import ProgressBar


class OptimizerState(NamedTuple):
    """
    Container for optimization state during continuous optimization.

    This NamedTuple tracks the complete state of an optimization process,
    including current and best parameters, optimizer internals, and optional
    update history for debugging.

    Attributes:
        params: Current parameter values at this optimization step.
        state: Internal optimizer state (e.g., momentum buffers, line search state).
        updates: Parameter updates computed at this step (before applying to params).
        update_norm: L2 norm of the current parameter updates.
        value: Objective function value at current parameters.
        best_val: Best (lowest) objective function value seen so far.
        best_params: Parameter values that achieved the best objective value.
        update_history: Optional array logging [update_norm, value] history if log_updates=True.
    """

    params: PyTree
    state: PyTree
    updates: PyTree
    update_norm: float
    value: float
    best_val: float
    best_params: PyTree
    update_history: Optional[Array]


def _debug_callback(
    id: int,
    arguments: Any,
) -> str:
    """
    Format progress information for optimization monitoring.

    This callback function creates a human-readable progress string for the ProgressBar
    during optimization. It displays the current update norm, convergence tolerance,
    iteration number, and objective value.

    Args:
        id: Progress bar task ID (typically progress_id from optimize function).
        arguments: Tuple containing (update_norm, tol, iter_num, value, max_iters).

    Returns:
        Formatted string showing optimization progress in scientific notation.

    Example output:
        "Optimizing 0... update 1e-03 => 1e-10 at iter 15 value 2e-01"
    """
    update_norm, tol, iter_num, value, max_iters = arguments
    return f"Optimizing {id}... update {update_norm:.0e} => {tol:.0e} at iter {iter_num} value {value:.0e}"


@partial(jax.jit, static_argnums=(1, 2, 3, 5, 9))
def optimize(
    init_params: Array,
    objective_fn: Callable[[Array], Array],
    opt: optax._src.base.GradientTransformationExtraArgs,
    max_iter: int,
    tol: Array,
    progress: Optional[ProgressBar] = None,
    progress_id: int = 0,
    upper_bound: Optional[Array] = None,
    lower_bound: Optional[Array] = None,
    log_updates: bool = False,
    **kwargs: Any,
) -> tuple[Array, OptimizerState]:
    """
    Optimize a function using gradient-based methods with Optax optimizers.

    This function performs JIT-compiled continuous optimization using various Optax optimizers
    (LBFGS, Adam, SGD, etc.) with built-in convergence checking, progress monitoring, and
    optional parameter bounds.

    Args:
        init_params: Initial parameter values to start optimization from.
        objective_fn: Objective function to minimize. Must be JAX-compatible and differentiable.
            Should accept parameters and return a scalar value.
        opt: Optax optimizer instance (e.g., optax.lbfgs(), optax.adam()).
        max_iter: Maximum number of optimization iterations.
        tol: Convergence tolerance. Optimization stops when update norm < tol.
        progress: Optional ProgressBar instance for tracking optimization progress.
        progress_id: ID for progress tracking when running multiple optimizations in parallel.
        upper_bound: Optional upper bounds for parameters (used with box projection).
        lower_bound: Optional lower bounds for parameters (used with box projection).
        log_updates: If True, logs update norms and values for debugging.
        **kwargs: Additional keyword arguments passed to the objective function.

    Returns:
        tuple: (best_params, final_optimizer_state)
            - best_params: Parameters that achieved the lowest objective value
            - final_optimizer_state: OptimizerState containing optimization history

    Example:
        >>> import jax.numpy as jnp
        >>> import optax
        >>> from jax_grid_search import optimize, ProgressBar
        >>>
        >>> def quadratic(x):
        ...     return jnp.sum((x - 3.0) ** 2)
        >>>
        >>> init_params = jnp.array([0.0])
        >>> optimizer = optax.lbfgs()
        >>>
        >>> with ProgressBar() as p:
        ...     best_params, state = optimize(
        ...         init_params, quadratic, optimizer,
        ...         max_iter=50, tol=1e-10, progress=p
        ...     )
        >>> print(f"Optimized parameters: {best_params}")

    Note:
        This function is JIT-compiled for performance. The objective function must be
        JAX-compatible (using jnp instead of np, avoiding Python control flow).
        Use jax.lax.cond or other JAX control flow primitives for conditional logic.
    """
    # Define a function that computes both value and gradient of the objective.
    value_and_grad_fun = jax.value_and_grad(objective_fn)
    update_history = jnp.zeros((max_iter, 2)) if log_updates else None

    # Single optimization step.
    def step(carry: OptimizerState) -> OptimizerState:
        value, grad = value_and_grad_fun(carry.params, **kwargs)  # Compute value and gradient
        updates, state = opt.update(
            grad, carry.state, carry.params, value=carry.value, grad=grad, value_fn=objective_fn, **kwargs
        )  # Perform update
        update_norm = otu.tree_l2_norm(updates)  # Compute update norm
        params = optax.apply_updates(carry.params, updates)  # Update params
        if upper_bound is not None and lower_bound is not None:
            params = optax.projections.projection_box(params, lower_bound, upper_bound)  # Apply box constraints
        if log_updates and carry.update_history is not None:
            iter_num = otu.tree_get(carry.state, "count")
            to_log = jnp.array([update_norm, value])
            carry = carry._replace(update_history=carry.update_history.at[iter_num].set(to_log))

        best_params = jax.tree.map(
            lambda x, y: jnp.where((carry.best_val < value) | jnp.isnan(value), x, y),
            carry.best_params,
            carry.params,
        )
        best_val = jnp.where((carry.best_val < value) | jnp.isnan(value), carry.best_val, value)

        if progress:
            iter_num = otu.tree_get(carry.state, "count")
            progress.update(progress_id, (update_norm, tol, iter_num, carry.value, max_iter), desc_cb=_debug_callback, total=max_iter)

        return carry._replace(
            params=params,
            state=state,
            updates=updates,
            value=value,
            best_val=best_val,
            best_params=best_params,
            update_norm=update_norm,
        )

    # Stopping condition.
    def continuing_criterion(carry: OptimizerState) -> Any:
        iter_num = otu.tree_get(carry.state, "count")  # Get iteration count from optimizer state
        iter_num = 0 if iter_num is None else iter_num
        update_norm = carry.update_norm
        return (iter_num == 0) | ((iter_num < max_iter) & (update_norm >= tol))

    # Initialize optimizer state.
    init_state = OptimizerState(init_params, opt.init(init_params), init_params, jnp.inf, jnp.inf, jnp.inf, init_params, update_history)

    # Run the while loop.
    if progress:
        progress.create_task(progress_id, total=max_iter)
    final_opt_state = jax.lax.while_loop(continuing_criterion, step, init_state)
    if progress:
        progress.finish(progress_id, total=max_iter)

    # Was the last evaluation better than the best?
    best_params = jax.tree.map(
        lambda x, y: jnp.where((final_opt_state.best_val < final_opt_state.value) | jnp.isnan(final_opt_state.value), x, y),
        final_opt_state.best_params,
        final_opt_state.params,
    )
    best_value: float = jnp.where(
        (final_opt_state.best_val < final_opt_state.value) | jnp.isnan(final_opt_state.value),
        final_opt_state.best_val,
        final_opt_state.value,
    )  # type: ignore[assignment]
    final_opt_state = final_opt_state._replace(best_params=best_params, best_val=best_value)

    return final_opt_state.best_params, final_opt_state
