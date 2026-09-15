"""Frame-aware null models for coding sequences.

Shuffling a CDS base by base destroys the reading frame, so the result is not a
credible background for anything about coding sequence: it has no start codon,
stops in the wrong places, and encodes a completely different protein. Three
alternatives, each answering a different question:

:func:`shuffle_codons`
    Rearrange whole codons. Codon usage is preserved exactly; with ``k=2`` the
    codon *pair* counts are preserved too, which is the null model used in
    codon-pair bias work. The protein changes.

:func:`shuffle_synonymous`
    Permute codons within each synonymous family. The protein and the codon
    usage are both preserved exactly; only which synonymous codon sits at which
    position changes. This is the standard control for questions about
    synonymous-site signal: RNA structure in coding regions, codon
    optimisation, exonic splicing elements.

:func:`shuffle_third_positions`
    Redraw each codon from its synonymous family, with replacement. Preserves
    the protein but not the exact codon counts -- for when the null should be
    "codon usage as a distribution" rather than "this exact multiset".

The translation tables themselves live in :mod:`helioseq.seq.code`; this module
only builds null models out of them.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..seq.code import codon_table, split_codons
from ..seq.types import SeqLike, coerce
from .shuffler import Shuffler, _next_seed

__all__ = [
    "shuffle_codons",
    "shuffle_synonymous",
    "shuffle_third_positions",
]


def _amino_acids(codons: List[bytes], table: int) -> List[str]:
    mapping = codon_table(table)
    return [
        mapping.get(codon.decode("ascii", "replace").upper(), "X") for codon in codons
    ]


def shuffle_codons(
    sequence: SeqLike, k: int = 1, *, seed: Optional[int] = None
) -> Any:
    """Rearrange whole codons, preserving k-let counts *over codons*.

    ``k=1`` preserves codon usage; ``k=2`` additionally preserves every codon
    pair count, ``k=3`` every codon triple, and so on.

    Implemented by mapping each distinct codon to a single byte and running the
    ordinary k-let shuffle over that alphabet, so the uniformity guarantee
    carries over unchanged.

    >>> from helioseq.seq import codon_counts
    >>> cds = "ATGGCTGCAGGTTTAAAGCTGTAA"
    >>> codon_counts(shuffle_codons(cds, seed=1)) == codon_counts(cds)
    True
    """
    raw, restore = coerce(sequence)
    codons = split_codons(raw)
    if not codons:
        return restore(raw)

    order: Dict[bytes, int] = {}
    for codon in codons:
        if codon not in order:
            order[codon] = len(order)
    if len(order) > 250:
        raise ValueError(
            "sequence uses %d distinct codons, more than the 250 this mapping "
            "supports; the input is probably not a clean CDS" % len(order)
        )

    alphabet = [bytes([1 + index]) for index in range(len(order))]
    encoded = b"".join(alphabet[order[codon]] for codon in codons)

    shuffled = Shuffler(encoded, k, seed=seed).shuffle_raw()

    inverse = {1 + index: codon for codon, index in order.items()}
    return restore(b"".join(inverse[byte] for byte in shuffled))


def shuffle_synonymous(
    sequence: SeqLike, *, table: int = 1, seed: Optional[int] = None
) -> Any:
    """Permute codons within each synonymous family.

    The translated protein and the codon usage are both exactly unchanged; what
    varies is which synonymous codon occupies which position.

    >>> from helioseq.seq import translate
    >>> cds = "ATGGCTGCAGCCGCGGGTTTAAAGCTGTAA"
    >>> translate(shuffle_synonymous(cds, seed=3)) == translate(cds)
    True
    """
    raw, restore = coerce(sequence)
    codons = split_codons(raw)
    rng = random.Random(_next_seed() if seed is None else int(seed))

    families: Dict[str, List[int]] = {}
    for position, amino in enumerate(_amino_acids(codons, table)):
        families.setdefault(amino, []).append(position)

    out = list(codons)
    for amino in sorted(families):  # sorted so the result depends only on the seed
        positions = families[amino]
        members = [codons[p] for p in positions]
        rng.shuffle(members)
        for position, codon in zip(positions, members):
            out[position] = codon

    return restore(b"".join(out))


def shuffle_third_positions(
    sequence: SeqLike, *, table: int = 1, seed: Optional[int] = None
) -> Any:
    """Redraw each codon from its synonymous family, with replacement.

    >>> from helioseq.seq import translate
    >>> cds = "ATGGCTGCAGCCGCGGGTTTAAAGCTGTAA"
    >>> translate(shuffle_third_positions(cds, seed=3)) == translate(cds)
    True
    """
    raw, restore = coerce(sequence)
    codons = split_codons(raw)
    amino_acids = _amino_acids(codons, table)
    rng = random.Random(_next_seed() if seed is None else int(seed))

    pools: Dict[str, List[bytes]] = {}
    for codon, amino in zip(codons, amino_acids):
        pools.setdefault(amino, []).append(codon)

    return restore(b"".join(rng.choice(pools[amino]) for amino in amino_acids))
