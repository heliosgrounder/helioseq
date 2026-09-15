"""The record type every reader yields and every writer accepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["Record"]


@dataclass
class Record:
    """One sequence record. ``quality`` is set only for FASTQ."""

    id: str
    sequence: str
    description: str = ""
    quality: Optional[str] = None

    @property
    def header(self) -> str:
        return "%s %s" % (self.id, self.description) if self.description else self.id

    def __len__(self) -> int:
        return len(self.sequence)

    def replaced(self, sequence: str, *, id: Optional[str] = None) -> "Record":
        """A copy carrying a new sequence.

        Quality is dropped when the length changes, because per-base quality
        no longer describes the new bases. Keeping it would produce a FASTQ
        file that parses but lies.
        """
        quality = self.quality
        if quality is not None and len(quality) != len(sequence):
            quality = None
        return Record(
            id=self.id if id is None else id,
            sequence=sequence,
            description=self.description,
            quality=quality,
        )
