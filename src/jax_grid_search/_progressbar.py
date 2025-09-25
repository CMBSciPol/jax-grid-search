import sys
from typing import Any, Callable, Dict, Optional

import jax
from jax.debug import callback
from rich.progress import Progress, Task, TaskID

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from jaxtyping import Array, PyTree


def _base_cb(id: int, arguments: PyTree[Array]) -> Optional[str]:
    return None


@jax.tree_util.register_static
class ProgressBar:
    """
    JAX-compatible progress bar using Rich library for visual progress tracking.

    This class integrates with JAX's callback system to provide thread-safe progress
    updates that work with JIT compilation and distributed computing. It uses the Rich
    library for enhanced visual display and supports multiple concurrent progress tasks.

    The progress bar is registered as a static PyTree node to ensure compatibility
    with JAX transformations like jit, vmap, and distributed operations.

    Example:
        >>> from jax_grid_search import ProgressBar, optimize
        >>> import optax
        >>> import jax.numpy as jnp
        >>>
        >>> with ProgressBar() as p:
        ...     result = optimize(
        ...         jnp.array([0.0]),
        ...         lambda x: x**2,
        ...         optax.adam(0.1),
        ...         max_iter=100,
        ...         tol=1e-6,
        ...         progress=p
        ...     )

    Note:
        This class uses JAX callbacks which are compatible with JIT compilation
        but may add overhead. For production code, consider disabling progress
        tracking for maximum performance.
    """

    def __init__(self: Self, *args: None, **kwargs: None) -> None:
        self.tasks: Dict[TaskID, Task] = {}
        self.progress = Progress(*args, **kwargs)
        self.progress.start()

    def create_task(self: Self, id: int, total: int) -> None:
        """
        Create a new progress tracking task.

        Args:
            id: Unique identifier for this progress task.
            total: Total number of steps for this task.

        Note:
            Uses JAX callback system for compatibility with JIT compilation.
            Multiple tasks can run concurrently with different IDs.
        """

        def _create_task(id: Array, total: Array) -> None:
            id = int(id)  # type: ignore[assignment]
            if id not in self.tasks:
                self.tasks[id] = self.progress.add_task(f"Running {id}...", total=total)
            else:
                # Reset
                self.progress.reset(self.tasks[id], total=total, start=True)

        return callback(_create_task, id, total, ordered=True)

    def update(
        self: Self,
        idx: int,
        arguments: PyTree[Array],
        desc_cb: Callable[[int, Any], Optional[str]] = _base_cb,
        total: int = 100,
        n: int = 1,
    ) -> None:
        """
        Update progress for a specific task.

        Args:
            idx: Task ID to update.
            arguments: Arguments passed to the description callback function.
            desc_cb: Function to format progress description (optional).
            total: Total steps (used if task doesn't exist yet).
            n: Number of steps to advance (default: 1).

        Note:
            Creates task automatically if it doesn't exist.
            Compatible with JAX transformations via callback system.
        """

        def _update_task(idx: int, total: int, arguments: PyTree[Array]) -> None:
            idx = int(idx)
            if idx not in self.tasks:
                self.create_task(idx, total)
            desc = desc_cb(idx, arguments)
            self.progress.update(self.tasks[idx], advance=n, description=desc)

        return callback(_update_task, idx, total, arguments, ordered=True)

    def finish(self: Self, id: int, total: int) -> None:
        """
        Mark a progress task as completed.

        Args:
            id: Task ID to finish.
            total: Total steps (used if task doesn't exist yet).

        Note:
            Sets progress to completion and stops further updates for this task.
            Creates task if it doesn't exist to ensure clean completion.
        """

        def _finish_task(id: int, total: int) -> None:
            id = int(id)
            if id not in self.tasks:
                self.create_task(id, total)
            self.progress.update(self.tasks[id], completed=total)

        return callback(_finish_task, id, total, ordered=True)

    def close(self) -> None:
        self.progress.stop()

    def __enter__(self: Self) -> Self:
        self.progress.__enter__()
        return self

    def __exit__(self: Self, exc_type: type[RuntimeError], exc_value: RuntimeError, traceback: Any) -> Any:
        return self.progress.__exit__(exc_type, exc_value, traceback)
