"""Empirical null distributions done correctly.

The pattern this module encodes is the reason most people reach for a shuffling
library: compute a statistic on a sequence, compute it again on many shuffles,
and ask how extreme the real value is. It is easy to write and easy to get
wrong.

Three mistakes are near-universal.

* ``p = #{null >= observed} / n``. When no shuffle beats the real sequence this
  gives ``p = 0``, which is not a probability any finite sample can support.
  The right estimator adds one to both sides -- ``(1 + #{null >= observed}) /
  (n + 1)`` -- and is what :class:`NullResult` reports (Davison & Hinkley 1997;
  North, Curtis & Sham 2002). With ``n = 1000`` the smallest reportable value
  is ``1/1001``, and it is honest to write it that way.
* Reporting a z-score as though the null were normal. The z-score here is
  descriptive only; the p-value is the empirical one, and
  :attr:`NullResult.normality_warning` says so when the null is visibly skewed.
* Testing a statistic the null model preserves by construction. GC content and
  CpG counts survive a ``k >= 2`` shuffle untouched, so testing them against a
  dinucleotide null produces a tidy table of ``p = 1`` rather than an error.
  :attr:`NullResult.constant_null` catches it.

This module is deliberately independent of how the shuffles are produced: pass
any ``shuffle_fn``, or none to use the default k-let shuffle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any, Callable, List, Optional

from ..seq.types import SeqLike

__all__ = ["Statistic", "NullResult", "null_distribution", "null_test"]

Statistic = Callable[[Any], float]

_ALTERNATIVES = ("greater", "less", "two-sided")


@dataclass
class NullResult:
    """The outcome of :func:`null_test`."""

    observed: float
    null: List[float] = field(repr=False)
    alternative: str = "greater"
    k: int = 2

    @property
    def n(self) -> int:
        return len(self.null)

    @property
    def null_mean(self) -> float:
        return fmean(self.null) if self.null else float("nan")

    @property
    def null_sd(self) -> float:
        return pstdev(self.null) if len(self.null) > 1 else 0.0

    @property
    def zscore(self) -> float:
        """Descriptive only -- the reported p-value does not assume normality."""
        sd = self.null_sd
        if sd == 0.0:
            return float("nan")
        return (self.observed - self.null_mean) / sd

    @property
    def n_at_least(self) -> int:
        return sum(1 for value in self.null if value >= self.observed)

    @property
    def n_at_most(self) -> int:
        return sum(1 for value in self.null if value <= self.observed)

    @property
    def pvalue(self) -> float:
        """Empirical p-value with the add-one correction."""
        total = self.n + 1
        if self.alternative == "greater":
            return (1 + self.n_at_least) / total
        if self.alternative == "less":
            return (1 + self.n_at_most) / total
        one_sided = min((1 + self.n_at_least) / total, (1 + self.n_at_most) / total)
        return min(1.0, 2 * one_sided)

    @property
    def resolution(self) -> float:
        """The smallest p-value this many shuffles can produce, ``1/(n+1)``.

        Report it next to a significant result: ``p = 0.001`` from 1000 shuffles
        means "at the floor", not "one in a thousand measured precisely".
        """
        return 1.0 / (self.n + 1)

    @property
    def at_resolution_limit(self) -> bool:
        return math.isclose(self.pvalue, self.resolution)

    def quantile(self, q: float) -> float:
        """Empirical quantile of the null distribution."""
        if not self.null:
            return float("nan")
        ordered = sorted(self.null)
        position = q * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    @property
    def ci95(self) -> tuple:
        """Central 95% interval of the *null*, not of the observed value."""
        return (self.quantile(0.025), self.quantile(0.975))

    @property
    def constant_null(self) -> bool:
        """The statistic takes the same value on every shuffle.

        Usually this means the statistic is an invariant of the null model
        rather than something it can test.
        """
        return self.n > 1 and self.null_sd == 0.0

    @property
    def constant_null_warning(self) -> Optional[str]:
        if not self.constant_null:
            return None
        return (
            "every shuffle gives the same value, so a %d-let shuffle cannot "
            "test this statistic: it is preserved by construction. Try a "
            "smaller k, or a statistic that depends on the arrangement of the "
            "sequence rather than its composition." % self.k
        )

    @property
    def normality_warning(self) -> Optional[str]:
        """Set when the null is skewed enough that the z-score misleads."""
        if len(self.null) < 20 or self.null_sd == 0:
            return None
        mean = self.null_mean
        sd = self.null_sd
        skew = fmean([((value - mean) / sd) ** 3 for value in self.null])
        if abs(skew) > 1.0:
            return (
                "the null distribution is skewed (skewness %.2f); use the "
                "empirical p-value, not the z-score" % skew
            )
        return None

    def summary(self) -> str:
        lines = [
            "observed      %.6g" % self.observed,
            "null          mean %.6g, sd %.6g, 95%% [%.6g, %.6g]"
            % (self.null_mean, self.null_sd, *self.ci95),
            "z-score       %.3f" % self.zscore,
            "p-value       %.4g  (%s, %d shuffles of %d-lets)"
            % (self.pvalue, self.alternative, self.n, self.k),
        ]
        if self.at_resolution_limit:
            lines.append(
                "              at the resolution limit 1/(n+1) = %.4g; "
                "increase n for a smaller p" % self.resolution
            )
        for warning in (self.constant_null_warning, self.normality_warning):
            if warning:
                lines.append("NOTE          " + warning)
        return "\n".join(lines)


def null_distribution(
    sequence: SeqLike,
    statistic: Statistic,
    *,
    k: int = 2,
    n: int = 1000,
    seed: Optional[int] = None,
    threads: int = 1,
    shuffle_fn: Optional[Callable[..., Any]] = None,
) -> List[float]:
    """Evaluate ``statistic`` on ``n`` shuffles of ``sequence``.

    ``shuffle_fn`` swaps in a different null model -- pass
    :func:`helioseq.shuffle.shuffle_masked` or
    :func:`helioseq.shuffle.shuffle_windows` (as a ``functools.partial`` with
    its own options) and everything else stays the same. It is called as
    ``shuffle_fn(sequence, k, seed=...)``.
    """
    from ..shuffle.shuffler import Shuffler, _next_seed

    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1, got %d" % n)
    base_seed = _next_seed() if seed is None else int(seed)

    if shuffle_fn is not None:
        return [
            statistic(shuffle_fn(sequence, k, seed=base_seed + index))
            for index in range(n)
        ]

    if threads <= 1:
        shuffler = Shuffler(sequence, k, seed=base_seed)
        return [statistic(shuffler.shuffle()) for _ in range(n)]

    # Each worker owns a shuffler on its own stream, so the set of values does
    # not depend on the thread count (their order may).
    from concurrent.futures import ThreadPoolExecutor

    workers = min(int(threads), n)
    chunk = (n + workers - 1) // workers

    def run(index: int) -> List[float]:
        local = Shuffler(sequence, k, seed=base_seed, stream=index + 1)
        count = min(chunk, n - index * chunk)
        return [statistic(local.shuffle()) for _ in range(max(0, count))]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [value for part in pool.map(run, range(workers)) for value in part]


def null_test(
    sequence: SeqLike,
    statistic: Statistic,
    *,
    k: int = 2,
    n: int = 1000,
    seed: Optional[int] = None,
    alternative: str = "greater",
    threads: int = 1,
    shuffle_fn: Optional[Callable[..., Any]] = None,
) -> NullResult:
    """Compare ``statistic(sequence)`` against its k-let preserving null.

    >>> from helioseq.stats import null_test, kmer_frequency
    >>> result = null_test("ACGTACGTAAGCTT" * 20, kmer_frequency("CG"),
    ...                    k=1, n=199, seed=1)
    >>> 0 < result.pvalue <= 1
    True

    A worked example -- RNA folding energy, the classic "is this structure
    unusual" test -- needs nothing from this package beyond a callable::

        import RNA
        mfe = lambda s: RNA.fold(s)[1]
        result = null_test(seq, mfe, k=2, n=1000, seed=1, alternative="less")
        print(result.summary())
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            "alternative must be one of %s, got %r"
            % (", ".join(_ALTERNATIVES), alternative)
        )
    values = null_distribution(
        sequence,
        statistic,
        k=k,
        n=n,
        seed=seed,
        threads=threads,
        shuffle_fn=shuffle_fn,
    )
    return NullResult(
        observed=float(statistic(sequence)),
        null=values,
        alternative=alternative,
        k=int(k),
    )
