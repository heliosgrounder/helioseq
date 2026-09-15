"""Local shuffling: windows and arbitrary segments.

A global shuffle preserves k-let counts over the whole sequence, which also
means it happily moves a GC-rich stretch to the other end. When composition
varies along the sequence -- isochores, a CpG island inside a promoter, a
structured domain inside a long transcript -- a global shuffle is a weaker null
model than it looks, because the statistic under test may be sensitive to the
*arrangement* of composition rather than its total.

Shuffling inside windows keeps the local profile and only rearranges within it.
The trade-off is explicit: smaller windows mean a more conservative null and
less randomisation, and k-lets that straddle a boundary are not preserved. Both
functions here report boundaries honestly rather than hiding the seam.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ..seq.types import SeqLike, coerce
from .shuffler import Shuffler, _next_seed

__all__ = ["shuffle_segments", "shuffle_windows", "tile_spans", "window_report"]

Span = Tuple[int, int]


def tile_spans(length: int, window: int, *, min_window: int = 0) -> List[Span]:
    """Split ``[0, length)`` into consecutive windows.

    A trailing window shorter than ``min_window`` is merged into the previous
    one, so a 1000 bp sequence tiled at 300 bp does not end with a 100 bp
    fragment that barely shuffles at all.
    """
    length = int(length)
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1, got %d" % window)
    if length <= 0:
        return []

    spans = [(start, min(start + window, length)) for start in range(0, length, window)]
    if len(spans) > 1 and min_window > 0:
        last_start, last_end = spans[-1]
        if last_end - last_start < min_window:
            prev_start, _ = spans[-2]
            spans[-2:] = [(prev_start, last_end)]
    return spans


def shuffle_segments(
    sequence: SeqLike,
    k: int = 2,
    *,
    spans: Iterable[Span],
    seed: Optional[int] = None,
) -> Any:
    """Shuffle each span independently; leave everything outside them untouched.

    Spans must be sorted and non-overlapping. This is the primitive that
    :func:`shuffle_windows` and :mod:`helioseq.shuffle.masking` are built on, and it is
    also the hook for callers with their own annotation: shuffle the exons and
    leave the introns alone, shuffle outside the known motif occurrences, and so
    on.
    """
    raw, restore = coerce(sequence)
    base_seed = _next_seed() if seed is None else int(seed)

    out = bytearray(raw)
    previous_end = 0
    for index, (start, end) in enumerate(spans):
        start, end = int(start), int(end)
        if start < previous_end:
            raise ValueError(
                "spans must be sorted and non-overlapping; span %d starts at %d "
                "but the previous one ends at %d" % (index, start, previous_end)
            )
        if not 0 <= start <= end <= len(raw):
            raise ValueError(
                "span %d = (%d, %d) is outside the sequence of length %d"
                % (index, start, end, len(raw))
            )
        previous_end = end
        if end - start < 2:
            continue
        piece = Shuffler(raw[start:end], k, seed=base_seed, stream=index)
        out[start:end] = piece.shuffle_raw()

    return restore(bytes(out))


def shuffle_windows(
    sequence: SeqLike,
    k: int = 2,
    *,
    window: int = 100,
    min_window: Optional[int] = None,
    seed: Optional[int] = None,
) -> Any:
    """Shuffle inside consecutive windows of ``window`` residues.

    >>> out = shuffle_windows("ACGT" * 50, k=2, window=40, seed=1)
    >>> len(out)
    200

    ``min_window`` defaults to half the window size.
    """
    raw, restore = coerce(sequence)
    if min_window is None:
        min_window = max(1, int(window) // 2)
    spans = tile_spans(len(raw), window, min_window=min_window)
    return restore(shuffle_segments(raw, k, spans=spans, seed=seed))


def window_report(length: int, window: int, k: int) -> str:
    """One line describing what windowing costs, for CLI output and logs."""
    spans: Sequence[Span] = tile_spans(length, window)
    boundaries = max(0, len(spans) - 1)
    return (
        "%d windows of up to %d residues; %d boundaries, so %d of the %d "
        "%d-lets are not preserved"
        % (
            len(spans),
            window,
            boundaries,
            boundaries * (k - 1),
            max(0, length - k + 1),
            k,
        )
    )
