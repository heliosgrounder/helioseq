"""Sequence types: accept what callers have, return what they gave.

Every public function in helioseq takes ``str``, ``bytes``, ``bytearray`` or
``memoryview`` and returns the same type it was given. That rule lives here so
it is stated once rather than reimplemented per module.

It is also a bug fix carried over from uShuffle 1.x, which accepted only
``bytes``: a ``str`` raised ``TypeError``, and a list comprehension over ``str``
sequences produced a baffling ``UnicodeDecodeError`` (guma44/ushuffle#4).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Union

__all__ = ["SeqLike", "as_bytes", "restorer", "coerce"]

SeqLike = Union[str, bytes, bytearray, memoryview]

_Restore = Callable[[bytes], Any]


def _restore_str(raw: bytes) -> str:
    return raw.decode("ascii")


def _restore_bytes(raw: bytes) -> bytes:
    return raw


def _restore_bytearray(raw: bytes) -> bytearray:
    return bytearray(raw)


def coerce(seq: SeqLike, *, argument: str = "sequence") -> Tuple[bytes, _Restore]:
    """Return ``(raw_bytes, restore)`` for ``seq``.

    ``restore`` converts any ``bytes`` result back to the caller's input type,
    so a ``str`` in means a ``str`` out.
    """
    if isinstance(seq, str):
        try:
            return seq.encode("ascii"), _restore_str
        except UnicodeEncodeError as exc:
            raise ValueError(
                "%s contains non-ASCII characters at position %d; biological "
                "sequences are expected to be ASCII. Pass bytes if you really "
                "mean to shuffle arbitrary binary data." % (argument, exc.start)
            ) from None
    if isinstance(seq, bytes):
        return seq, _restore_bytes
    if isinstance(seq, bytearray):
        return bytes(seq), _restore_bytearray
    if isinstance(seq, memoryview):
        return seq.tobytes(), _restore_bytes
    raise TypeError(
        "%s must be str, bytes, bytearray or memoryview, not %s"
        % (argument, type(seq).__name__)
    )


def as_bytes(seq: SeqLike, *, argument: str = "sequence") -> bytes:
    """Just the bytes, when the caller does not need to convert anything back."""
    return coerce(seq, argument=argument)[0]


def restorer(seq: SeqLike) -> _Restore:
    """Just the conversion function for ``seq``'s type."""
    return coerce(seq)[1]
