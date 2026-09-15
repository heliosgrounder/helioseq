"""Streaming sequence file I/O.

Deliberately small and dependency-free. Biopython does this better and does far
more, but making it a hard requirement of a shuffling toolkit is the kind of
dependency that turns ``pip install`` into an afternoon.

Everything streams: a 30 GB FASTQ passes through in constant memory, one record
at a time. Compression is detected from magic bytes rather than the file
extension.

    >>> from helioseq import seqio
    >>> for record in seqio.read_sequences("peaks.fa.gz"):   # doctest: +SKIP
    ...     print(record.id, len(record))

Adding a format means a new module here plus one line in ``_READERS`` and
``_WRITERS``; nothing outside this subpackage needs to change.
"""

from __future__ import annotations

import sys
from typing import IO, Iterator, Optional

from ._open import PathLike, guess_format, open_maybe_compressed
from .fasta import read_fasta, write_fasta
from .fastq import read_fastq, write_fastq
from .records import Record

__all__ = [
    "Record",
    "PathLike",
    "FORMATS",
    "open_maybe_compressed",
    "guess_format",
    "read_fasta",
    "read_fastq",
    "read_sequences",
    "write_fasta",
    "write_fastq",
    "write_record",
]

_READERS = {"fasta": read_fasta, "fastq": read_fastq}

# Each writer declares which keyword options it understands, so write_record()
# can reject a typo instead of silently dropping it -- `wrap=0` quietly ignored
# on FASTQ output would be a confusing afternoon.
_WRITERS = {
    "fasta": (write_fasta, frozenset({"wrap"})),
    "fastq": (write_fastq, frozenset()),
}

FORMATS = tuple(sorted(_READERS))


def read_sequences(
    path: Optional[PathLike], fmt: Optional[str] = None
) -> Iterator[Record]:
    """Stream records from a path, autodetecting format and compression.

    Reading from ``"-"`` requires an explicit ``fmt``, because detection would
    have to consume the stream first.
    """
    if fmt is None:
        if path is None or str(path) == "-":
            raise ValueError(
                "format cannot be detected on standard input; pass "
                "fmt='fasta' or fmt='fastq'"
            )
        fmt = guess_format(path)

    fmt = fmt.lower()
    if fmt not in _READERS:
        raise ValueError(
            "format must be one of %s, got %r" % (", ".join(FORMATS), fmt)
        )

    handle = open_maybe_compressed(path, "rt")
    try:
        for record in _READERS[fmt](handle):
            yield record
    finally:
        if handle is not sys.stdin:
            handle.close()


def write_record(handle: IO, record: Record, fmt: str = "fasta", **options) -> None:
    """Write one record in ``fmt``.

    Options a format does not understand are dropped, but only if some other
    bundled format does understand them -- so a caller can pass ``wrap=60``
    for a run that emits both FASTA and FASTQ, while a genuine typo still
    raises.
    """
    fmt = fmt.lower()
    if fmt not in _WRITERS:
        raise ValueError(
            "format must be one of %s, got %r" % (", ".join(FORMATS), fmt)
        )
    writer, accepted = _WRITERS[fmt]

    known = set().union(*(names for _, names in _WRITERS.values()))
    unknown = set(options) - known
    if unknown:
        raise TypeError(
            "unknown write option(s) %s; no bundled format accepts them"
            % ", ".join(sorted(repr(name) for name in unknown))
        )
    writer(handle, record, **{k: v for k, v in options.items() if k in accepted})
