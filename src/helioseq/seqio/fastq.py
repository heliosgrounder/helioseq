"""FASTQ reading and writing."""

from __future__ import annotations

from typing import IO, Iterable, Iterator

from ._open import split_header
from .records import Record

__all__ = ["read_fastq", "write_fastq"]


def read_fastq(handle: Iterable[str]) -> Iterator[Record]:
    """Stream FASTQ records (four lines each, sequences not wrapped).

    Malformed input raises with the line number, because "your FASTQ is broken
    somewhere" is not an actionable error message for a 40 GB file.
    """
    lines = iter(handle)
    line_number = 0
    while True:
        try:
            header = next(lines)
        except StopIteration:
            return
        line_number += 1
        if not header.strip():
            continue
        if not header.startswith("@"):
            raise ValueError(
                "line %d: expected a FASTQ header starting with '@', got %r"
                % (line_number, header[:40])
            )
        try:
            sequence = next(lines).rstrip("\r\n")
            separator = next(lines)
            quality = next(lines).rstrip("\r\n")
        except StopIteration:
            raise ValueError(
                "truncated FASTQ record starting at line %d" % line_number
            ) from None
        line_number += 3
        if not separator.startswith("+"):
            raise ValueError(
                "line %d: expected '+' separator, got %r"
                % (line_number - 1, separator[:40])
            )
        if len(sequence) != len(quality):
            raise ValueError(
                "line %d: sequence and quality differ in length (%d vs %d)"
                % (line_number, len(sequence), len(quality))
            )
        identifier, description = split_header(header)
        yield Record(identifier, sequence, description, quality)


def write_fastq(handle: IO, record: Record) -> None:
    """Write one record as FASTQ. Missing quality is filled with ``I`` (Q40)."""
    quality = record.quality
    if quality is None or len(quality) != len(record.sequence):
        quality = "I" * len(record.sequence)
    handle.write("@%s\n%s\n+\n%s\n" % (record.header, record.sequence, quality))
