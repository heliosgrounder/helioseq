"""One-hot sequence tensors for PyTorch models and training pipelines.

This mirrors :mod:`helioseq.shuffle.ml`'s NumPy one-hot helpers, but returns
``torch.Tensor`` objects so encoded sequences can be fed straight into a
PyTorch model, loss, or ``DataLoader`` without an extra conversion step.

torch is optional: it is imported lazily, only when one of these functions is
called, so importing ``helioseq.shuffle`` never requires it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .ml import ALPHABET

__all__ = [
    "one_hot_encode_torch",
    "one_hot_decode_torch",
    "one_hot_encode_batch_torch",
    "ALPHABET",
]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "one-hot torch helpers need PyTorch; install helioseq[torch], "
            "or pass sequences as str"
        ) from exc
    return torch


def one_hot_encode_torch(
    sequence: str,
    alphabet: str = ALPHABET,
    *,
    dtype: Any = None,
    device: Any = None,
):
    """``(L, len(alphabet))`` float32 tensor. Unknown residues are zero.

    >>> from helioseq.shuffle.ml_torch import one_hot_decode_torch, one_hot_encode_torch
    >>> tensor = one_hot_encode_torch("ACGT")
    >>> tensor.shape
    torch.Size([4, 4])
    >>> one_hot_decode_torch(tensor)
    'ACGT'
    """
    torch = _require_torch()
    index = {residue: position for position, residue in enumerate(alphabet)}
    array = torch.zeros(
        (len(sequence), len(alphabet)), dtype=dtype or torch.float32, device=device
    )
    for position, residue in enumerate(sequence.upper()):
        column = index.get(residue)
        if column is not None:
            array[position, column] = 1.0
    return array


def one_hot_decode_torch(tensor, alphabet: str = ALPHABET) -> str:
    """Inverse of :func:`one_hot_encode_torch`; all-zero rows become ``N``."""
    torch = _require_torch()
    tensor = torch.as_tensor(tensor)
    if tensor.ndim != 2 or tensor.shape[-1] != len(alphabet):
        raise ValueError(
            "expected an (L, %d) tensor, got shape %r"
            % (len(alphabet), tuple(tensor.shape))
        )
    residues = []
    for row in tensor:
        if not bool(row.any()):
            residues.append("N")
        else:
            residues.append(alphabet[int(row.argmax())])
    return "".join(residues)


def one_hot_encode_batch_torch(
    sequences: Sequence[str],
    alphabet: str = ALPHABET,
    *,
    pad_to: Optional[int] = None,
    dtype: Any = None,
    device: Any = None,
):
    """Stack sequences into an ``(N, L, len(alphabet))`` tensor, for a batch.

    With ``pad_to=None`` (the default) every sequence must be the same
    length. With ``pad_to`` set, shorter sequences are zero-padded at the end
    (the same "unknown residue -> all-zero row" convention as
    :func:`one_hot_encode_torch`); a sequence longer than ``pad_to`` is an
    error.
    """
    torch = _require_torch()

    if pad_to is None:
        lengths = {len(sequence) for sequence in sequences}
        if len(lengths) > 1:
            raise ValueError(
                "sequences have mismatched lengths %s; pass pad_to= to pad them"
                % sorted(lengths)
            )
        length = next(iter(lengths), 0)
    else:
        length = pad_to
        for sequence in sequences:
            if len(sequence) > length:
                raise ValueError(
                    "sequence of length %d exceeds pad_to=%d" % (len(sequence), length)
                )

    batch = torch.zeros(
        (len(sequences), length, len(alphabet)),
        dtype=dtype or torch.float32,
        device=device,
    )
    for row, sequence in enumerate(sequences):
        encoded = one_hot_encode_torch(sequence, alphabet, dtype=dtype, device=device)
        batch[row, : encoded.shape[0]] = encoded
    return batch
