"""Sequence primitives.

The bottom layer of helioseq: what a sequence *is*, what alphabet it is drawn
from, and the handful of transformations that every other domain needs. This
subpackage imports nothing else from helioseq, so anything here is safe to use
from anywhere.
"""

from __future__ import annotations

from .alphabet import ALPHABETS, DNA, PROTEIN, RNA, Alphabet, detect, get, validate
from .code import (
    GENETIC_CODES,
    codon_counts,
    codon_table,
    split_codons,
    translate,
)
from .transform import complement, composition, gc_fraction, reverse_complement
from .types import SeqLike, as_bytes, coerce, restorer

__all__ = [
    # types
    "SeqLike",
    "as_bytes",
    "coerce",
    "restorer",
    # alphabets
    "Alphabet",
    "DNA",
    "RNA",
    "PROTEIN",
    "ALPHABETS",
    "get",
    "detect",
    "validate",
    # genetic code
    "GENETIC_CODES",
    "codon_table",
    "split_codons",
    "translate",
    "codon_counts",
    # transformations
    "complement",
    "reverse_complement",
    "composition",
    "gc_fraction",
]
