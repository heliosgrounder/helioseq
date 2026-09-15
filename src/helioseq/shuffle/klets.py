"""k-let counting, verification and degeneracy diagnostics.

The guarantee that makes a shuffle usable as a null model is that the k-let
counts are *exactly* preserved. It costs almost nothing to check, so check it --
especially in a pipeline, where a silent change of composition would move every
p-value downstream.

The second thing worth checking is degeneracy. Some sequences admit only one
Eulerian walk, so "shuffling" returns the input every time. Low-complexity
regions, short sequences and large ``k`` all lead there, and nothing in the
output looks wrong; the background set is simply a copy of the foreground. See
:func:`diagnose`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from ..seq.types import SeqLike, as_bytes
from .shuffler import Shuffler

__all__ = [
    "klet_counts",
    "klets_preserved",
    "check_klets",
    "distinct_count",
    "is_degenerate",
    "Diagnosis",
    "diagnose",
]


def klet_counts(sequence: SeqLike, k: int = 2, *, as_str: bool = True) -> Counter:
    """Count every k-let (length-``k`` substring, overlapping).

    >>> sorted(klet_counts("ACGTACG", k=2).items())
    [('AC', 2), ('CG', 2), ('GT', 1), ('TA', 1)]
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1, got %d" % k)
    raw = as_bytes(sequence)
    if len(raw) < k:
        return Counter()
    if as_str:
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            as_str = False
        else:
            return Counter(text[i : i + k] for i in range(len(text) - k + 1))
    return Counter(raw[i : i + k] for i in range(len(raw) - k + 1))


def klets_preserved(original: SeqLike, shuffled: SeqLike, k: int = 2) -> bool:
    """Whether two sequences have identical k-let counts."""
    return klet_counts(original, k, as_str=False) == klet_counts(
        shuffled, k, as_str=False
    )


def check_klets(original: SeqLike, shuffled: SeqLike, k: int = 2) -> None:
    """Raise ``AssertionError`` with a readable diff if the counts differ.

    Cheap enough to leave switched on in a pipeline for anything but the
    innermost loop.
    """
    before = klet_counts(original, k)
    after = klet_counts(shuffled, k)
    if before == after:
        return

    missing = before - after
    added = after - before
    parts = []
    if missing:
        parts.append(
            "lost " + ", ".join("%s x%d" % kv for kv in sorted(missing.items())[:8])
        )
    if added:
        parts.append(
            "gained " + ", ".join("%s x%d" % kv for kv in sorted(added.items())[:8])
        )
    raise AssertionError("%d-let counts changed: %s" % (k, "; ".join(parts)))


def distinct_count(
    sequence: SeqLike, k: int = 2, n: int = 100, *, seed: Optional[int] = None
) -> int:
    """How many distinct sequences ``n`` shuffles produce.

    A blunt but effective estimate of how much room the shuffle actually has.
    """
    shuffler = Shuffler(sequence, k, seed=seed)
    return len({shuffler.shuffle_raw() for _ in range(int(n))})


def is_degenerate(
    sequence: SeqLike, k: int = 2, n: int = 20, *, seed: Optional[int] = None
) -> bool:
    """``True`` when ``n`` shuffles all return the input unchanged.

    Worth calling before building a background set out of short or
    low-complexity sequences::

        >>> is_degenerate("GCGCGCGCGCGCGCGC", k=4)
        True
    """
    raw = as_bytes(sequence)
    if len(raw) <= k:
        return True
    shuffler = Shuffler(raw, k, seed=seed)
    return all(shuffler.shuffle_raw() == raw for _ in range(int(n)))


@dataclass
class Diagnosis:
    """What :func:`diagnose` found out about one sequence."""

    length: int
    k: int
    n_samples: int
    distinct: int
    identical_to_input: int
    n_vertices: int
    estimated_memory: int

    @property
    def degenerate(self) -> bool:
        return self.distinct <= 1

    @property
    def identical_fraction(self) -> float:
        return self.identical_to_input / self.n_samples if self.n_samples else 0.0

    def warnings(self) -> list:
        """Human-readable problems, empty when the sequence shuffles well."""
        issues = []
        if self.length <= self.k:
            issues.append(
                "sequence is not longer than k=%d, so the only shuffle is the "
                "input itself" % self.k
            )
        elif self.degenerate:
            issues.append(
                "every shuffle is identical to the input: this sequence has "
                "exactly one arrangement with these %d-let counts, so it "
                "cannot serve as its own background" % self.k
            )
        elif self.identical_fraction > 0.1:
            issues.append(
                "%.0f%% of shuffles are identical to the input; the space of "
                "rearrangements is very small" % (100 * self.identical_fraction)
            )
        elif self.distinct < self.n_samples / 4:
            issues.append(
                "only %d distinct results in %d shuffles; consider a smaller k"
                % (self.distinct, self.n_samples)
            )
        return issues

    def summary(self) -> str:
        lines = [
            "length=%d  k=%d  distinct (k-1)-lets=%d"
            % (self.length, self.k, self.n_vertices),
            "%d distinct results in %d shuffles, %d identical to the input"
            % (self.distinct, self.n_samples, self.identical_to_input),
            "estimated peak memory: %.1f MB" % (self.estimated_memory / 1048576.0),
        ]
        lines.extend("WARNING: " + w for w in self.warnings())
        return "\n".join(lines)


def diagnose(
    sequence: SeqLike, k: int = 2, n: int = 100, *, seed: Optional[int] = None
) -> Diagnosis:
    """Sample ``n`` shuffles and report how well this sequence shuffles at all."""
    from ._backend import memory_estimate

    raw = as_bytes(sequence)
    n = int(n)
    shuffler = Shuffler(raw, k, seed=seed)
    draws = [shuffler.shuffle_raw() for _ in range(n)]
    return Diagnosis(
        length=len(raw),
        k=int(k),
        n_samples=n,
        distinct=len(set(draws)),
        identical_to_input=sum(1 for d in draws if d == raw),
        n_vertices=shuffler.n_vertices,
        estimated_memory=memory_estimate(len(raw), int(k)),
    )
