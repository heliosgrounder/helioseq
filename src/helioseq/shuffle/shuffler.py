"""The shuffling API.

Everything here preserves exact k-let (k-mer substring) counts, and every
shuffle is drawn uniformly from the set of sequences that have those counts.
For ``k = 2`` on DNA that means dinucleotide composition -- and therefore GC and
CpG content -- is identical to the input, which is what makes these sequences
usable as a background for motif and structure analyses.

Reproducibility
---------------
Pass ``seed=`` for a result that is identical on every platform and every
Python version, or call :func:`set_seed` once to make a whole script
reproducible. Unlike uShuffle 1.x this does not swap the algorithm's random
source for a weaker one, and it is not a one-way door.
"""

from __future__ import annotations

import os
import random
import threading
from typing import Any, Iterator, List, Optional, Sequence

from ..seq.types import SeqLike, coerce
from ._backend import ShufflerImpl, backend, memory_estimate

__all__ = [
    "Shuffler",
    "shuffle",
    "shuffle_many",
    "shuffle_batch",
    "set_seed",
    "backend",
    "memory_estimate",
]

_MASK64 = (1 << 64) - 1

_seed_lock = threading.Lock()
_seed_source = random.Random()


def set_seed(seed: Optional[int]) -> None:
    """Make every later unseeded shuffle reproducible.

    Each shuffler still owns its random state; this only fixes the stream of
    seeds handed out when ``seed=None``. Pass ``None`` to go back to
    non-deterministic seeding.
    """
    with _seed_lock:
        if seed is None:
            _seed_source.seed(os.urandom(32))
        else:
            _seed_source.seed(int(seed))


def _next_seed() -> int:
    with _seed_lock:
        return _seed_source.getrandbits(64)


def _validate_k(k: int) -> int:
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1, got %d" % k)
    return k


class Shuffler:
    """Reusable shuffler for one sequence.

    Building the graph is the expensive part, so shuffling the same sequence
    many times is much cheaper through a ``Shuffler`` than through repeated
    :func:`shuffle` calls::

        sh = Shuffler(seq, k=2, seed=42)
        background = sh.shuffle_many(1000)

    Instances are independent: two live shufflers never interfere. (In uShuffle
    1.x they did, and the symptom was heap corruption rather than a wrong
    answer.) One instance is *not* safe to share between threads -- give each
    thread its own, or use :func:`shuffle_batch`.
    """

    __slots__ = ("_impl", "_restore", "_seq", "_k", "_seed", "_stream")

    def __init__(
        self,
        sequence: SeqLike,
        k: int = 2,
        *,
        seed: Optional[int] = None,
        stream: int = 0,
    ) -> None:
        raw, restore = coerce(sequence)
        k = _validate_k(k)
        self._seed = _next_seed() if seed is None else int(seed) & _MASK64
        self._stream = int(stream) & _MASK64
        self._impl = ShufflerImpl(seed=self._seed, stream=self._stream)
        self._impl.prepare(raw, k)
        self._restore = restore
        self._seq = raw
        self._k = k

    # -- introspection ----------------------------------------------------

    @property
    def sequence(self) -> Any:
        """The prepared sequence, in the type it was given in."""
        return self._restore(self._seq)

    @property
    def k(self) -> int:
        return self._k

    @property
    def length(self) -> int:
        return len(self._seq)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def n_vertices(self) -> int:
        """Distinct (k-1)-lets, i.e. vertices in the underlying graph."""
        return self._impl.n_vertices

    def __len__(self) -> int:
        return len(self._seq)

    def __repr__(self) -> str:
        return "Shuffler(length=%d, k=%d, seed=%d)" % (
            len(self._seq),
            self._k,
            self._seed,
        )

    # -- shuffling --------------------------------------------------------

    def shuffle(self) -> Any:
        """One shuffle, in the input's type."""
        return self._restore(self._impl.generate())

    def shuffle_raw(self) -> bytes:
        """One shuffle as raw ``bytes``, skipping the type conversion.

        Used by the modules that stitch segments together and by the batch
        helpers, where converting each intermediate result would be wasted work.
        """
        return self._impl.generate()

    def shuffle_many(self, n: int) -> List[Any]:
        """``n`` shuffles as a list."""
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0, got %d" % n)
        restore = self._restore
        generate = self._impl.generate
        return [restore(generate()) for _ in range(n)]

    def __iter__(self) -> Iterator[Any]:
        """An endless stream of shuffles -- pair it with ``itertools.islice``."""
        restore = self._restore
        generate = self._impl.generate
        while True:
            yield restore(generate())

    def reseed(self, seed: int, stream: int = 0) -> None:
        """Rewind to the start of a random stream.

        After this, the shuffler emits exactly what a freshly constructed
        ``Shuffler(sequence, k, seed=seed, stream=stream)`` would. That needs
        more than resetting the generator: each draw permutes the graph's edge
        lists in place (which is part of what makes successive draws
        independent), so the graph is rebuilt as well. The cost is one
        ``prepare``, which is why this is a separate call rather than something
        ``shuffle`` does.
        """
        self._seed = int(seed) & _MASK64
        self._stream = int(stream) & _MASK64
        self._impl.seed(self._seed, self._stream)
        self._impl.prepare(self._seq, self._k)


def shuffle(sequence: SeqLike, k: int = 2, *, seed: Optional[int] = None) -> Any:
    """Shuffle ``sequence``, preserving the counts of every k-let.

    Returns the same type it was given::

        >>> shuffle("ACGTACGTAGCTAGCT", k=2, seed=1)   # doctest: +SKIP
        'ACGTAGCTACGTAGCT'

    ``k=1`` is a plain uniform permutation (composition preserved, nothing
    else). ``k >= len(sequence)`` returns the input unchanged, because that is
    the only sequence with those k-let counts.
    """
    return Shuffler(sequence, k, seed=seed).shuffle()


def shuffle_many(
    sequence: SeqLike, k: int = 2, n: int = 1, *, seed: Optional[int] = None
) -> List[Any]:
    """``n`` independent shuffles of one sequence, analysing it only once."""
    return Shuffler(sequence, k, seed=seed).shuffle_many(n)


def shuffle_batch(
    sequences: Sequence[SeqLike],
    k: int = 2,
    *,
    n: int = 1,
    seed: Optional[int] = None,
    threads: int = 1,
) -> List[List[Any]]:
    """Shuffle many sequences, optionally on several threads.

    Returns one list of ``n`` shuffles per input sequence, in input order.

    Sequence *i* is shuffled with stream *i* of the same seed, so the output
    does not depend on how work was scheduled: ``threads=8`` and ``threads=1``
    give byte-identical results. The compiled backend releases the GIL, so
    threads give real parallelism; the pure-Python fallback will not scale.
    """
    k = _validate_k(k)
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0, got %d" % n)
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be >= 1, got %d" % threads)

    base_seed = _next_seed() if seed is None else int(seed) & _MASK64
    items = list(sequences)

    def one(index: int) -> List[Any]:
        return Shuffler(
            items[index], k, seed=base_seed, stream=index
        ).shuffle_many(n)

    if threads == 1 or len(items) < 2:
        return [one(i) for i in range(len(items))]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(threads, len(items))) as pool:
        return list(pool.map(one, range(len(items))))
