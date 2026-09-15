"""The genetic code: translation tables, codon splitting, translation.

Kept separate from :mod:`helioseq.shuffle.codon`, which uses these tables to
build null models. Translating a CDS and shuffling one are different jobs, and
only the first belongs in the sequence layer.
"""

from __future__ import annotations

from typing import Dict, List

from .types import SeqLike, coerce

__all__ = [
    "GENETIC_CODES",
    "codon_table",
    "split_codons",
    "translate",
    "codon_counts",
]

_BASES = "TCAG"

# NCBI translation tables, as the amino acid string over the 64 codons in
# TTT, TTC, TTA, TTG, TCT, ... order. Add one by pasting the NCBI "AAs" line.
GENETIC_CODES: Dict[int, str] = {
    1: "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    2: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG",
    4: "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    # Table 11 (bacterial/plant plastid) differs from 1 only in its start
    # codons, which are not modelled here.
    11: "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
}

_ALL_CODONS = [a + b + c for a in _BASES for b in _BASES for c in _BASES]


def codon_table(table: int = 1) -> Dict[str, str]:
    """``{"TTT": "F", ...}`` for an NCBI translation table id.

    >>> codon_table(1)["ATG"]
    'M'
    >>> codon_table(2)["TGA"]   # vertebrate mitochondrial: not a stop
    'W'
    """
    try:
        amino_acids = GENETIC_CODES[int(table)]
    except KeyError:
        raise ValueError(
            "translation table %r is not bundled; available: %s. Pass a dict "
            "of your own instead." % (table, sorted(GENETIC_CODES))
        ) from None
    return dict(zip(_ALL_CODONS, amino_acids))


def split_codons(sequence: SeqLike, *, argument: str = "sequence") -> List[bytes]:
    """Split an in-frame sequence into codons, rejecting a broken frame."""
    raw, _ = coerce(sequence, argument=argument)
    if len(raw) % 3:
        raise ValueError(
            "%s has length %d, which is not a multiple of 3; a coding sequence "
            "must be in frame. Trim it first, or treat it as a plain sequence."
            % (argument, len(raw))
        )
    return [raw[i : i + 3] for i in range(0, len(raw), 3)]


def translate(sequence: SeqLike, table: int = 1, *, unknown: str = "X") -> str:
    """Translate an in-frame CDS. Codons outside the table become ``unknown``.

    >>> translate("ATGGCTGCAGGTTTAAAGCTGTAA")
    'MAAGLKL*'
    """
    mapping = codon_table(table)
    return "".join(
        mapping.get(codon.decode("ascii", "replace").upper(), unknown)
        for codon in split_codons(sequence)
    )


def codon_counts(sequence: SeqLike) -> Dict[str, int]:
    """Codon usage of an in-frame CDS.

    >>> codon_counts("ATGATGTAA")["ATG"]
    2
    """
    counts: Dict[str, int] = {}
    for codon in split_codons(sequence):
        key = codon.decode("ascii", "replace").upper()
        counts[key] = counts.get(key, 0) + 1
    return counts
