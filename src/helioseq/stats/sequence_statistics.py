"""Ready-made statistics to test against a null distribution.

Each one is a plain callable taking a sequence and returning a float, which is
also the contract for anything you write yourself -- ``null_test`` never needs
to know where the number came from.

Note which of these a k-let shuffle can actually test. ``gc_content`` is
preserved exactly by any ``k >= 1`` shuffle and ``cpg_observed_expected`` by any
``k >= 2``, so against those nulls they are invariants, not tests. That is worth
knowing rather than discovering from a column of ``p = 1``; see
:attr:`helioseq.stats.NullResult.constant_null`.
"""

from __future__ import annotations

from ..seq.transform import gc_fraction, reverse_complement
from ..seq.types import SeqLike, as_bytes
from .nulls import Statistic

__all__ = [
    "gc_content",
    "cpg_observed_expected",
    "kmer_frequency",
    "longest_homopolymer",
]


def gc_content(sequence: SeqLike) -> float:
    """Fraction of G and C among all residues.

    >>> gc_content("ACGT")
    0.5
    """
    return gc_fraction(sequence)


def cpg_observed_expected(sequence: SeqLike) -> float:
    """CpG observed/expected ratio, ``(CG * N) / (C * G)``.

    >>> cpg_observed_expected("CGCGCG")     # 3 CpGs, 3 C, 3 G, length 6
    2.0
    >>> round(cpg_observed_expected("GCGCGC"), 4)   # same composition, 2 CpGs
    1.3333
    """
    raw = as_bytes(sequence).upper()
    if len(raw) < 2:
        return float("nan")
    c = raw.count(b"C")
    g = raw.count(b"G")
    if c == 0 or g == 0:
        return 0.0
    cg = sum(1 for i in range(len(raw) - 1) if raw[i : i + 2] == b"CG")
    return cg * len(raw) / (c * g)


def kmer_frequency(kmer: str, *, both_strands: bool = False) -> Statistic:
    """Build a statistic counting ``kmer`` occurrences per residue.

    >>> round(kmer_frequency("CG")("ACGACG"), 4)
    0.3333
    """
    needle = kmer.upper().encode("ascii")
    if not needle:
        raise ValueError("kmer must not be empty")
    needles = [needle]
    if both_strands:
        needles.append(as_bytes(reverse_complement(needle)))

    def statistic(sequence: SeqLike) -> float:
        raw = as_bytes(sequence).upper()
        if len(raw) < len(needle):
            return 0.0
        total = 0
        for pattern in needles:
            total += sum(
                1
                for i in range(len(raw) - len(pattern) + 1)
                if raw[i : i + len(pattern)] == pattern
            )
        return total / len(raw)

    statistic.__name__ = "kmer_frequency(%s)" % kmer
    return statistic


def longest_homopolymer(sequence: SeqLike) -> float:
    """Length of the longest run of one residue.

    >>> longest_homopolymer("AACCCGT")
    3.0
    """
    raw = as_bytes(sequence).upper()
    best = run = 0
    previous = None
    for byte in raw:
        run = run + 1 if byte == previous else 1
        previous = byte
        best = max(best, run)
    return float(best)
