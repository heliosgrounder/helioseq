"""Shuffling real genomic FASTA without destroying what is written into it.

Genome sequence carries two annotations in-band. Runs of ``N`` mark assembly
gaps and unsequenced regions; lowercase marks repeats, from RepeatMasker or
Dust. A naive shuffle treats both as ordinary letters, which produces a
background set where the gaps have moved, the repeat layout is gone, and the
``N`` runs have been smeared through the sequence as if they were bases.

The functions here keep those annotations where they are and shuffle only what
is actually sequence.

The honest caveat: holding positions fixed splits the sequence into independent
pieces, so k-lets that would straddle a frozen position are not preserved and
each piece is shuffled only within itself. That is a weaker null model than a
global shuffle, and a stronger one than pretending ``N`` is a nucleotide.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from ..seq.types import SeqLike, coerce
from .windows import shuffle_segments

__all__ = [
    "free_spans",
    "case_profile",
    "apply_case",
    "shuffle_masked",
]

Span = Tuple[int, int]

_CASE_MODES = ("preserve", "split", "ignore")


def free_spans(
    raw: bytes, keep: bytes = b"N", *, case_sensitive: bool = False
) -> List[Span]:
    """Maximal runs of positions that are *not* in ``keep``.

    >>> free_spans(b"ACGTNNNNACGT")
    [(0, 4), (8, 12)]
    """
    if case_sensitive:
        frozen = set(keep)
    else:
        frozen = set(keep.upper()) | set(keep.lower())

    spans: List[Span] = []
    start = None
    for position, byte in enumerate(raw):
        if byte in frozen:
            if start is not None:
                spans.append((start, position))
                start = None
        elif start is None:
            start = position
    if start is not None:
        spans.append((start, len(raw)))
    return spans


def case_profile(raw: bytes) -> bytes:
    """A per-position record of which residues were lowercase."""
    return bytes(1 if 97 <= b <= 122 else 0 for b in raw)


def apply_case(raw: bytes, profile: bytes) -> bytes:
    """Re-impose a :func:`case_profile` on a sequence of the same length."""
    if len(raw) != len(profile):
        raise ValueError(
            "case profile has length %d but the sequence has length %d"
            % (len(profile), len(raw))
        )
    out = bytearray(raw.upper())
    for position, lower in enumerate(profile):
        if lower:
            out[position] += 32
    return bytes(out)


def _split_by_case(raw: bytes, spans: List[Span]) -> List[Span]:
    """Subdivide each span at every upper/lower case transition."""
    result: List[Span] = []
    for start, end in spans:
        run_start = start
        for position in range(start + 1, end):
            was_lower = 97 <= raw[position - 1] <= 122
            is_lower = 97 <= raw[position] <= 122
            if was_lower != is_lower:
                result.append((run_start, position))
                run_start = position
        result.append((run_start, end))
    return result


def shuffle_masked(
    sequence: SeqLike,
    k: int = 2,
    *,
    keep: str = "N",
    case: str = "preserve",
    seed: Optional[int] = None,
) -> Any:
    """Shuffle a masked sequence, leaving the mask in place.

    ``keep``
        Characters frozen at their positions. ``"N"`` by default, matched
        case-insensitively; pass ``""`` to freeze nothing, or e.g. ``"N-"`` to
        hold alignment gaps too.

    ``case``
        What to do about soft masking.

        ``"preserve"`` (default)
            Shuffle the uppercased sequence, then put the original pattern of
            upper and lower case back, position by position. The repeat
            *annotation* is unchanged; which base carries it is not.
        ``"split"``
            Treat upper- and lowercase stretches as separate segments and
            shuffle each within itself, so repeat-derived bases stay inside
            repeats. The most faithful option, and the most conservative.
        ``"ignore"``
            Uppercase the output and forget the soft masking.

    >>> out = shuffle_masked("acgtACGTACGTNNNNacgtacgt", k=2, seed=7)
    >>> out[12:16]
    'NNNN'
    >>> [c.islower() for c in out] == [c.islower() for c in "acgtACGTACGTNNNNacgtacgt"]
    True
    """
    if case not in _CASE_MODES:
        raise ValueError(
            "case must be one of %s, got %r" % (", ".join(_CASE_MODES), case)
        )

    raw, restore = coerce(sequence)
    keep_bytes = keep.encode("ascii") if isinstance(keep, str) else bytes(keep)

    spans = free_spans(raw, keep_bytes) if keep_bytes else [(0, len(raw))]

    if case == "split":
        spans = _split_by_case(raw, spans)
        shuffled = shuffle_segments(raw, k, spans=spans, seed=seed)
        return restore(shuffled)

    profile = case_profile(raw) if case == "preserve" else None
    shuffled = shuffle_segments(raw.upper(), k, spans=spans, seed=seed)
    if profile is not None:
        shuffled = apply_case(shuffled, profile)
    return restore(shuffled)
