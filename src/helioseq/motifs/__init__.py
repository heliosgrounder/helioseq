"""Motifs: position weight matrices and scoring.

    >>> from helioseq import motifs
    >>> pwm = motifs.from_counts([{"A": 40, "C": 2, "G": 2, "T": 2}] * 6)
    >>> motifs.consensus(pwm)
    'AAAAAA'

Combined with :mod:`helioseq.stats`, motif enrichment against a
composition-matched background is one call::

    from helioseq import motifs
    from helioseq.stats import null_test

    result = null_test(sequence, motifs.best_score(pwm), k=2, n=1000)
"""

from __future__ import annotations

from .pwm import PWM, best_score, consensus, from_counts, scan

__all__ = ["PWM", "from_counts", "consensus", "scan", "best_score"]
