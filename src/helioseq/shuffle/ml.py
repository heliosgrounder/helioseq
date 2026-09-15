"""Dinucleotide shuffling for machine-learning pipelines.

Dinucleotide-preserving shuffles are the standard reference distribution for
attribution methods in regulatory genomics: DeepSHAP's ``DeepExplainer`` uses
them as baselines, TF-MoDISco builds on those attributions, and shuffled
sequences are the usual negative set when training a sequence model. The
reference implementation everyone copied (``deeplift.dinuc_shuffle``) is pure
NumPy, and at tens of thousands of sequences times twenty references each it
becomes a real part of the runtime.

:func:`dinuc_shuffle` is a drop-in replacement with the same signature, backed
by the C extension.

There is one deliberate behavioural difference, and it matters for correctness
rather than speed. The common NumPy implementation samples an Eulerian walk by
shuffling each vertex's out-edges independently, which is *not* uniform over
sequences with the given dinucleotide counts -- it over-samples some
arrangements. This module uses Wilson's algorithm (via libushuffle), which is
uniform by construction. Expect slightly different -- better -- reference
distributions, not identical output.

NumPy is optional: it is needed only for the one-hot forms.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from ..seq.types import as_bytes
from .shuffler import Shuffler, _next_seed

__all__ = [
    "dinuc_shuffle",
    "shuffle_onehot",
    "one_hot_encode",
    "one_hot_decode",
    "ALPHABET",
]

ALPHABET = "ACGT"


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "one-hot input needs NumPy; install it, or pass sequences as str"
        ) from exc
    return np


def _seed_from(rng: Any) -> Optional[int]:
    """Derive a seed from whatever the caller passed as ``rng``.

    Accepts an int, ``numpy.random.RandomState``/``Generator``, ``random.Random``
    or ``None``, so the deeplift call style keeps working.
    """
    if rng is None:
        return None
    if isinstance(rng, int):
        return rng
    for attribute in ("randint", "integers"):
        method = getattr(rng, attribute, None)
        if method is not None:
            try:
                return int(method(0, 2**62))
            except TypeError:
                continue
    getrandbits = getattr(rng, "getrandbits", None)
    if getrandbits is not None:
        return int(getrandbits(62))
    raise TypeError(
        "rng must be None, an int, numpy.random.RandomState/Generator or "
        "random.Random, not %s" % type(rng).__name__
    )


def one_hot_encode(sequence: str, alphabet: str = ALPHABET):
    """``(L, len(alphabet))`` float32 one-hot array. Unknown residues are zero."""
    np = _require_numpy()
    index = {residue: position for position, residue in enumerate(alphabet)}
    array = np.zeros((len(sequence), len(alphabet)), dtype="float32")
    for position, residue in enumerate(sequence.upper()):
        column = index.get(residue)
        if column is not None:
            array[position, column] = 1.0
    return array


def one_hot_decode(array, alphabet: str = ALPHABET) -> str:
    """Inverse of :func:`one_hot_encode`; all-zero rows become ``N``."""
    np = _require_numpy()
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != len(alphabet):
        raise ValueError(
            "expected an (L, %d) array, got shape %r" % (len(alphabet), array.shape)
        )
    residues = []
    for row in array:
        if not row.any():
            residues.append("N")
        else:
            residues.append(alphabet[int(row.argmax())])
    return "".join(residues)


def dinuc_shuffle(
    seq: Union[str, bytes, Any],
    num_shufs: Optional[int] = None,
    rng: Any = None,
    *,
    k: int = 2,
    alphabet: str = ALPHABET,
):
    """Dinucleotide-preserving shuffle, compatible with ``deeplift.dinuc_shuffle``.

    ``seq``
        A ``str``/``bytes`` sequence, or an ``(L, 4)`` one-hot array.
    ``num_shufs``
        ``None`` returns a single shuffle of the same type as the input;
        an integer returns a list (or a stacked ``(N, L, 4)`` array for one-hot
        input).
    ``rng``
        ``None``, an ``int``, a NumPy ``RandomState``/``Generator`` or a
        ``random.Random``.
    ``k``
        Exposed as an extension: ``k=3`` preserves trinucleotides.

    >>> from helioseq.shuffle.ml import dinuc_shuffle
    >>> out = dinuc_shuffle("ACGTACGTAGCTAGCT", rng=1)
    >>> sorted(out) == sorted("ACGTACGTAGCTAGCT")
    True
    """
    seed = _seed_from(rng)
    if seed is None:
        seed = _next_seed()

    is_onehot = not isinstance(seq, (str, bytes, bytearray, memoryview))
    if is_onehot:
        text = one_hot_decode(seq, alphabet)
    else:
        text = as_bytes(seq).decode("ascii")

    shuffler = Shuffler(text, k, seed=seed)
    count = 1 if num_shufs is None else int(num_shufs)
    results: List[str] = [shuffler.shuffle() for _ in range(count)]

    if is_onehot:
        np = _require_numpy()
        stacked = np.stack([one_hot_encode(item, alphabet) for item in results])
        return stacked[0] if num_shufs is None else stacked

    if isinstance(seq, (bytes, bytearray, memoryview)):
        encoded = [item.encode("ascii") for item in results]
        return encoded[0] if num_shufs is None else encoded

    return results[0] if num_shufs is None else results


def shuffle_onehot(
    arrays: Sequence[Any],
    *,
    k: int = 2,
    n: int = 1,
    seed: Optional[int] = None,
    threads: int = 1,
    alphabet: str = ALPHABET,
):
    """Shuffle a batch of one-hot arrays, returning ``(batch, n, L, len(alphabet))``.

    Built for the reference-baseline step of an attribution run: one call
    replaces a Python loop over thousands of sequences, and with the compiled
    backend the work spreads across threads because the GIL is released around
    each shuffle.

    Sequence *i* uses stream *i* of ``seed``, so the output does not depend on
    ``threads``.
    """
    np = _require_numpy()
    texts = [one_hot_decode(array, alphabet) for array in arrays]

    from .shuffler import shuffle_batch

    shuffled = shuffle_batch(texts, k, n=n, seed=seed, threads=threads)
    return np.stack(
        [
            np.stack([one_hot_encode(item, alphabet) for item in per_sequence])
            for per_sequence in shuffled
        ]
    )
