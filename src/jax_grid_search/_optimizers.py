from typing import Any


def optimize(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "The 'optimize' function has been deprecated and removed from jax-grid-search. "
        "Please use 'furax_cs.minimize' from https://github.com/CMBSciPol/furax-cs instead."
    )
