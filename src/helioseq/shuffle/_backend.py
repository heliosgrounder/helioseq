"""Backend selection for the shuffling core.

The compiled extension is used when it is importable. Otherwise the pure-Python
port in :mod:`helioseq.shuffle._purepy` takes over, so ``import
helioseq.shuffle`` works on any platform, with or without a wheel and with or
without a compiler. The two produce byte-identical output for a given seed; the
fallback is roughly 100x slower.

Set ``HELIOSEQ_BACKEND=python`` to force the fallback, or ``=c`` to make a
missing extension an error instead of a warning. The test suite runs the whole
thing both ways.
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any

__all__ = ["ShufflerImpl", "memory_estimate", "backend", "core_version"]

_CHOICES = ("", "auto", "c", "python")

_requested = os.environ.get("HELIOSEQ_BACKEND", "").strip().lower()
if _requested not in _CHOICES:
    raise ValueError(
        "HELIOSEQ_BACKEND must be one of 'c', 'python' or 'auto', got %r" % _requested
    )

_impl: Any = None
_backend_name = ""

if _requested != "python":
    try:
        # Imported by full dotted path rather than `from . import _core`: this
        # module runs while helioseq.shuffle is still initialising, and the
        # `from` form would report a missing extension as a confusing
        # "partially initialized module / circular import" error.
        _impl = importlib.import_module(__name__.rsplit(".", 1)[0] + "._core")
        _backend_name = "c"
    except ImportError as exc:  # pragma: no cover - platform dependent
        if _requested == "c":
            raise ImportError(
                "HELIOSEQ_BACKEND=c was requested but the compiled extension is "
                "not available: %s" % exc
            ) from exc
        warnings.warn(
            "helioseq: the compiled extension is unavailable (%s), falling back "
            "to the pure-Python backend, which is roughly 100x slower. Install a "
            "wheel for your platform, or a C compiler, to get the fast path."
            % exc,
            RuntimeWarning,
            stacklevel=2,
        )

if _impl is None:
    from . import _purepy as _impl  # type: ignore[no-redef]

    _backend_name = "python"

ShufflerImpl = _impl.ShufflerImpl
memory_estimate = _impl.memory_estimate


def backend() -> str:
    """``"c"`` or ``"python"`` -- which implementation is actually running."""
    return _backend_name


def core_version() -> str:
    """Version of the underlying libushuffle C core."""
    getter = getattr(_impl, "core_version", None)
    return getter() if getter is not None else "2.0.0 (pure python)"
