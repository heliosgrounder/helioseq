"""One-hot PyTorch tensor helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="one-hot torch helpers need torch")


class TestOneHotTorch:
    def test_encode_decode_round_trip(self):
        from helioseq.shuffle.ml_torch import one_hot_decode_torch, one_hot_encode_torch

        sequence = "ACGTACGT"
        assert one_hot_decode_torch(one_hot_encode_torch(sequence)) == sequence

    def test_unknown_residues_decode_to_n(self):
        from helioseq.shuffle.ml_torch import one_hot_decode_torch, one_hot_encode_torch

        assert one_hot_decode_torch(one_hot_encode_torch("ACNT")) == "ACNT"

    def test_wrong_shape_is_rejected(self):
        from helioseq.shuffle.ml_torch import one_hot_decode_torch

        with pytest.raises(ValueError, match="expected an"):
            one_hot_decode_torch(torch.zeros((4, 7)))

    def test_dtype_defaults_to_float32(self):
        from helioseq.shuffle.ml_torch import one_hot_encode_torch

        assert one_hot_encode_torch("ACGT").dtype == torch.float32

    def test_dtype_is_passed_through(self):
        from helioseq.shuffle.ml_torch import one_hot_encode_torch

        assert one_hot_encode_torch("ACGT", dtype=torch.float64).dtype == torch.float64

    def test_batch_stacks_equal_length_sequences(self):
        from helioseq.shuffle.ml_torch import one_hot_encode_batch_torch

        sequences = ["ACGT", "TGCA", "AAAA"]
        batch = one_hot_encode_batch_torch(sequences)
        assert batch.shape == (3, 4, 4)

    def test_batch_pad_to_pads_shorter_sequences(self):
        from helioseq.shuffle.ml_torch import (
            one_hot_decode_torch,
            one_hot_encode_batch_torch,
        )

        batch = one_hot_encode_batch_torch(["AC", "ACGT"], pad_to=4)
        assert batch.shape == (2, 4, 4)
        assert one_hot_decode_torch(batch[0]) == "ACNN"
        assert one_hot_decode_torch(batch[1]) == "ACGT"

    def test_batch_pad_to_rejects_too_long_sequence(self):
        from helioseq.shuffle.ml_torch import one_hot_encode_batch_torch

        with pytest.raises(ValueError, match="exceeds pad_to"):
            one_hot_encode_batch_torch(["ACGTA"], pad_to=4)

    def test_batch_rejects_length_mismatch_without_pad_to(self):
        from helioseq.shuffle.ml_torch import one_hot_encode_batch_torch

        with pytest.raises(ValueError, match="mismatched lengths"):
            one_hot_encode_batch_torch(["AC", "ACGT"])
