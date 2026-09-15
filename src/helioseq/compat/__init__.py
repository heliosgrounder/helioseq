"""Shims for code written against other libraries' APIs.

These modules exist so that existing scripts keep running while they are
migrated; they are not where new code should start. Each one documents the
helioseq call it forwards to.

    >>> from helioseq.compat import ushuffle
    >>> out = ushuffle.shuffle(b"ACGTACGTAGCTAGCT", 2)
"""

from __future__ import annotations

__all__ = ["ushuffle"]
