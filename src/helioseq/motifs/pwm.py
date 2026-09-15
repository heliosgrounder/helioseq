"""Position weight matrices and motif scoring.

A PWM here is the plainest possible thing: a list with one ``{residue: score}``
mapping per motif position, holding log-odds. That is what you get from a
JASPAR or MEME count matrix after converting counts to log-odds, and it keeps
this module free of any particular file format.

Scoring returns a statistic compatible with :mod:`helioseq.stats`, so motif
enrichment against a dinucleotide-preserving background is one call.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from ..seq.types import SeqLike, as_bytes

__all__ = ["PWM", "from_counts", "consensus", "best_score", "scan"]

PWM = Sequence[Dict[str, float]]

_DNA = "ACGT"
_PAIRS = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}


def from_counts(
    counts: Sequence[Dict[str, float]],
    *,
    background: Dict[str, float] = None,
    pseudocount: float = 0.25,
) -> List[Dict[str, float]]:
    """Convert a count (or frequency) matrix to log-odds.

    ``pseudocount`` keeps an unobserved residue from scoring ``-inf`` and
    dominating everything; 0.25 is the usual choice for DNA.

    >>> pwm = from_counts([{"A": 40, "C": 0, "G": 0, "T": 0}])
    >>> pwm[0]["A"] > pwm[0]["C"]
    True
    """
    if background is None:
        background = {residue: 0.25 for residue in _DNA}

    matrix = []
    for position in counts:
        total = sum(position.values()) + pseudocount * len(position)
        row = {}
        for residue, count in position.items():
            probability = (count + pseudocount) / total
            row[residue] = math.log2(probability / background.get(residue, 0.25))
        matrix.append(row)
    return matrix


def consensus(pwm: PWM) -> str:
    """The highest-scoring residue at each position.

    >>> consensus([{"A": 2.0, "C": -1.0}, {"A": -1.0, "C": 2.0}])
    'AC'
    """
    return "".join(max(position, key=position.get) for position in pwm)


def _reverse_complement_matrix(pwm: PWM) -> List[Dict[str, float]]:
    return [
        {_PAIRS.get(residue, residue): score for residue, score in position.items()}
        for position in reversed(list(pwm))
    ]


def scan(
    pwm: PWM, sequence: SeqLike, *, both_strands: bool = True
) -> List[float]:
    """Score every position; ``-inf`` where a residue is not in the matrix.

    The returned list has one entry per start position on the forward strand,
    holding the better of the two strands when ``both_strands`` is set.
    """
    width = len(pwm)
    if width == 0:
        raise ValueError("pwm must have at least one position")

    text = as_bytes(sequence).upper().decode("ascii", "replace")
    if len(text) < width:
        return []

    tables = [list(pwm)]
    if both_strands:
        tables.append(_reverse_complement_matrix(pwm))

    scores = []
    for start in range(len(text) - width + 1):
        best = float("-inf")
        for table in tables:
            total = 0.0
            for offset, position in enumerate(table):
                total += position.get(text[start + offset], float("-inf"))
                if total == float("-inf"):
                    break
            if total > best:
                best = total
        scores.append(best)
    return scores


def best_score(pwm: PWM, *, both_strands: bool = True):
    """A :mod:`helioseq.stats` statistic returning the best PWM score.

    >>> motif = [{"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0}] * 2
    >>> round(best_score(motif, both_strands=False)("CCAACC"), 2)
    4.0
    """
    width = len(pwm)
    if width == 0:
        raise ValueError("pwm must have at least one position")

    def statistic(sequence: SeqLike) -> float:
        scores = scan(pwm, sequence, both_strands=both_strands)
        return max(scores) if scores else float("-inf")

    statistic.__name__ = "pwm_best_score(width=%d)" % width
    return statistic
