"""Statistics: empirical null distributions and the statistics to test.

    >>> from helioseq.stats import null_test, kmer_frequency
    >>> result = null_test(sequence, kmer_frequency("GGGCGG"), k=2, n=1000)  # doctest: +SKIP
    >>> print(result.summary())                                             # doctest: +SKIP

``nulls`` holds the machinery and knows nothing about biology;
``sequence_statistics`` holds the ready-made statistics. Anything callable that
maps a sequence to a float works, so a new statistic does not have to live here
at all.
"""

from __future__ import annotations

from .nulls import NullResult, Statistic, null_distribution, null_test
from .sequence_statistics import (
    cpg_observed_expected,
    gc_content,
    kmer_frequency,
    longest_homopolymer,
)

__all__ = [
    "Statistic",
    "NullResult",
    "null_distribution",
    "null_test",
    "gc_content",
    "cpg_observed_expected",
    "kmer_frequency",
    "longest_homopolymer",
]
