"""Opening files: compression and format detection.

Compression is detected from the file's magic bytes rather than its extension,
because a gzipped file named ``.fa`` is common enough that failing on it is
just an annoyance.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import os
import sys
from typing import IO, Optional, Union

__all__ = ["PathLike", "open_maybe_compressed", "guess_format"]

PathLike = Union[str, "os.PathLike[str]"]

_MAGIC = (
    (b"\x1f\x8b", gzip.open),
    (b"BZh", bz2.open),
    (b"\xfd7zXZ\x00", lzma.open),
)

_BY_SUFFIX = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}


def open_maybe_compressed(path: Optional[PathLike], mode: str = "rt") -> IO:
    """Open a path, transparently handling gzip/bzip2/xz.

    ``None`` or ``"-"`` means standard input or output. When writing, the
    compression is chosen from the extension (nothing to sniff yet).
    """
    writing = "w" in mode or "a" in mode

    if path is None or str(path) == "-":
        stream = sys.stdout if writing else sys.stdin
        return stream if "b" not in mode else stream.buffer

    if writing:
        name = str(path).lower()
        for suffix, opener in _BY_SUFFIX.items():
            if name.endswith(suffix):
                return opener(path, mode)
        return open(path, mode)

    with open(path, "rb") as probe:
        head = probe.read(8)
    for magic, opener in _MAGIC:
        if head.startswith(magic):
            return opener(path, mode)
    return open(path, mode)


def guess_format(path: Optional[PathLike]) -> str:
    """``"fasta"`` or ``"fastq"``, decided by the first non-blank character."""
    handle = open_maybe_compressed(path, "rt")
    try:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                return "fasta"
            if stripped.startswith("@"):
                return "fastq"
            raise ValueError(
                "%s does not start with '>' or '@'; it is neither FASTA nor FASTQ"
                % (path or "<stdin>")
            )
        return "fasta"  # an empty file is a valid empty FASTA
    finally:
        if handle not in (sys.stdin, sys.stdout):
            handle.close()


def split_header(line: str) -> tuple:
    """Split a ``>``/``@`` header into (id, description)."""
    header = line[1:].strip()
    if not header:
        return "", ""
    parts = header.split(None, 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")
