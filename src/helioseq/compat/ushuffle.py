"""Drop-in replacement for the ``ushuffle`` 1.x API.

For scripts written against ``ushuffle`` 1.1.x. Change

    import ushuffle

to

    from helioseq.compat import ushuffle

and the rest should keep working, including the ``bytes``-only behaviour that
1.x had.

What is deliberately *not* reproduced: the bugs. Two live ``Shuffler`` objects
no longer corrupt each other, ``set_seed`` no longer downgrades the random
source permanently, and out-of-memory raises ``MemoryError`` instead of killing
the interpreter. Output will therefore not match 1.x for a given seed -- it
could not, since 1.x used the platform's ``rand()``.

New code should use :mod:`helioseq.shuffle` directly: it accepts ``str``,
seeds per object, and has the rest of the toolkit behind it.
"""

from __future__ import annotations

import warnings
from typing import Optional

from ..seq.types import as_bytes
from ..shuffle import shuffler as _shuffler

__all__ = ["shuffle", "Shuffler", "set_seed"]

_DEPRECATION = (
    "helioseq.compat.ushuffle reproduces the uShuffle 1.x API for migration. "
    "Use helioseq.shuffle.%s instead, which accepts str as well as bytes and "
    "takes a per-object seed."
)


def shuffle(sequence, let_size: int = 2) -> bytes:
    """Shuffle ``sequence``, preserving ``let_size``-let counts. Returns bytes.

    >>> result = shuffle(b"ACGTACGTAGCTAGCT", 2)
    >>> sorted(result) == sorted(b"ACGTACGTAGCTAGCT")
    True
    """
    warnings.warn(_DEPRECATION % "shuffle", DeprecationWarning, stacklevel=2)
    return _shuffler.Shuffler(as_bytes(sequence), let_size).shuffle()


def set_seed(seed: int) -> None:
    """Fix the seeds handed to later unseeded shufflers.

    In 1.x this replaced the algorithm's random source with ``srand``/``rand``
    for the rest of the process and could not be undone. Here it only fixes the
    stream of seeds, and :func:`helioseq.shuffle.set_seed(None)` restores
    non-deterministic seeding.
    """
    warnings.warn(_DEPRECATION % "set_seed", DeprecationWarning, stacklevel=2)
    _shuffler.set_seed(seed)


class Shuffler:
    """A sequence shuffler with k-let size preservation.

    >>> sh = Shuffler(b"ACGTACGTAGCTAGCT", 2)
    >>> len(sh.shuffle()) == 16
    True
    """

    def __init__(self, sequence, let_size: int = 2, *, seed: Optional[int] = None):
        warnings.warn(_DEPRECATION % "Shuffler", DeprecationWarning, stacklevel=2)
        self._impl = _shuffler.Shuffler(as_bytes(sequence), let_size, seed=seed)

    @property
    def sequence(self) -> bytes:
        return self._impl.sequence

    @property
    def let_size(self) -> int:
        """1.x spelling of ``k``."""
        return self._impl.k

    @property
    def length(self) -> int:
        return self._impl.length

    def shuffle(self) -> bytes:
        return self._impl.shuffle()
