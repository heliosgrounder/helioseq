"""FASTA reading and writing."""

from __future__ import annotations

from typing import IO, Iterable, Iterator

from ._open import split_header
from .records import Record

__all__ = ["read_fasta", "write_fasta"]


def read_fasta(handle: Iterable[str]) -> Iterator[Record]:
    """Stream FASTA records from an open text handle.

    Multi-line sequences are joined; blank lines are ignored. Records are
    yielded one at a time, so a file larger than memory is fine.
    """
    identifier = None
    description = ""
    chunks: list = []

    for line in handle:
        line = line.rstrip("\r\n")
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                yield Record(identifier, "".join(chunks), description)
            identifier, description = split_header(line)
            chunks = []
        else:
            if identifier is None:
                raise ValueError("FASTA data before the first '>' header line")
            chunks.append(line.strip())

    if identifier is not None:
        yield Record(identifier, "".join(chunks), description)


def write_fasta(handle: IO, record: Record, *, wrap: int = 60) -> None:
    """Write one record as FASTA, wrapping at ``wrap`` columns (0 = no wrap)."""
    handle.write(">%s\n" % record.header)
    sequence = record.sequence
    if wrap and wrap > 0:
        for start in range(0, len(sequence), wrap):
            handle.write(sequence[start : start + wrap])
            handle.write("\n")
        if not sequence:
            handle.write("\n")
    else:
        handle.write(sequence)
        handle.write("\n")
