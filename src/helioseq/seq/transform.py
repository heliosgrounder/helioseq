"""Sequence transformations that several domains need.

Small on purpose: these are the operations that turn up in
:mod:`helioseq.stats`, :mod:`helioseq.motifs` and the CLI alike, and having
three private copies of reverse-complement is how they drift apart.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .types import SeqLike, coerce

__all__ = ["complement", "reverse_complement", "composition", "gc_fraction"]

# Case is preserved; N/n map to themselves. IUPAC ambiguity codes are
# complemented too, so a soft-masked ambiguous sequence survives intact.
_COMPLEMENT = bytes.maketrans(
    b"ACGTUacgtuRYSWKMBDHVNryswkmbdhvn",
    b"TGCAAtgcaaYRSWMKVHDBNyrswmkvhdbn",
)


def complement(sequence: SeqLike) -> Any:
    """Complement each residue, keeping order, case and unknown characters.

    >>> complement("ACGTn")
    'TGCAn'
    """
    raw, restore = coerce(sequence)
    return restore(raw.translate(_COMPLEMENT))


def reverse_complement(sequence: SeqLike) -> Any:
    """Reverse complement.

    >>> reverse_complement("ACGTTG")
    'CAACGT'
    """
    raw, restore = coerce(sequence)
    return restore(raw.translate(_COMPLEMENT)[::-1])


def composition(sequence: SeqLike, *, upper: bool = True) -> Counter:
    """Residue counts.

    >>> sorted(composition("AACGT").items())
    [('A', 2), ('C', 1), ('G', 1), ('T', 1)]
    """
    raw, _ = coerce(sequence)
    if upper:
        raw = raw.upper()
    return Counter(raw.decode("ascii", "replace"))


def gc_fraction(sequence: SeqLike) -> float:
    """Fraction of G and C among all residues; ``nan`` for an empty sequence.

    >>> gc_fraction("ACGT")
    0.5
    """
    raw, _ = coerce(sequence)
    if not raw:
        return float("nan")
    upper = raw.upper()
    return (upper.count(b"G") + upper.count(b"C")) / len(upper)
