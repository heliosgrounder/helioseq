"""k-let diagnostics and the machine-learning helpers."""

from __future__ import annotations

import pytest

from helioseq.shuffle import (
    check_klets,
    diagnose,
    distinct_count,
    is_degenerate,
    klet_counts,
    klets_preserved,
)


class TestKletDiagnostics:
    def test_counts(self):
        assert dict(klet_counts("ACGTACG", 2)) == {"AC": 2, "CG": 2, "GT": 1, "TA": 1}
        assert klet_counts("AC", 5) == {}

    def test_counts_reject_bad_k(self):
        with pytest.raises(ValueError, match="k must be"):
            klet_counts("ACGT", 0)

    def test_check_klets_passes_silently(self):
        check_klets("ACGTACGT", "ACGTACGT", 2)

    def test_check_klets_explains_the_difference(self):
        with pytest.raises(AssertionError, match="2-let counts changed"):
            check_klets("ACGTACGT", "AAAACCCC", 2)

    def test_klets_preserved_works_on_bytes(self):
        assert klets_preserved(b"ACGTACGT", b"ACGTACGT", 2)

    def test_degenerate_repeat(self):
        assert is_degenerate("GCGCGCGCGCGCGCGC", 4)
        assert is_degenerate("AAAAAAAA", 2)
        assert is_degenerate("AC", 2)

    def test_non_degenerate(self):
        assert not is_degenerate("ACGTACGTAGCTAGCTAAGGCCTT", 2, seed=1)

    def test_distinct_count(self):
        assert distinct_count("GCGCGCGCGC", 2, 50, seed=1) == 1
        assert distinct_count("ACGTACGTAGCTAGCTAAGG", 2, 50, seed=1) > 5

    def test_diagnosis_warns_about_degeneracy(self):
        report = diagnose("GCGCGCGCGCGC", 4, 20, seed=1)
        assert report.degenerate
        assert any("every shuffle is identical" in w for w in report.warnings())
        assert "WARNING" in report.summary()

    def test_diagnosis_is_quiet_for_a_healthy_sequence(self):
        report = diagnose("ACGTACGTAGCTAGCTAAGGCCTTACGT", 2, 100, seed=1)
        assert not report.warnings()
        assert report.distinct > 1
        assert report.identical_fraction < 0.1


class TestMl:
    def test_dinuc_shuffle_preserves_dinucleotides(self):
        from helioseq.shuffle.ml import dinuc_shuffle

        sequence = "ACGTACGTAGCTAGCTAAGGCCTT"
        result = dinuc_shuffle(sequence, rng=1)
        assert klets_preserved(sequence, result, 2)
        assert isinstance(result, str)

    def test_num_shufs_returns_a_list(self):
        from helioseq.shuffle.ml import dinuc_shuffle

        results = dinuc_shuffle("ACGTACGTAGCTAGCT", 5, rng=1)
        assert isinstance(results, list) and len(results) == 5

    def test_bytes_input(self):
        from helioseq.shuffle.ml import dinuc_shuffle

        assert isinstance(dinuc_shuffle(b"ACGTACGTAGCT", rng=1), bytes)

    def test_rng_accepts_several_shapes(self):
        import random

        from helioseq.shuffle.ml import dinuc_shuffle

        sequence = "ACGTACGTAGCTAGCTAAGG"
        assert dinuc_shuffle(sequence, rng=7) == dinuc_shuffle(sequence, rng=7)
        assert isinstance(dinuc_shuffle(sequence, rng=random.Random(3)), str)

    def test_rng_rejects_nonsense(self):
        from helioseq.shuffle.ml import dinuc_shuffle

        with pytest.raises(TypeError, match="rng must be"):
            dinuc_shuffle("ACGT", rng="tomorrow")

    def test_k_can_be_raised(self):
        from helioseq.shuffle.ml import dinuc_shuffle

        sequence = "ACGTACGTAGCTAGCTAAGGCCTTACGT"
        assert klets_preserved(sequence, dinuc_shuffle(sequence, rng=1, k=3), 3)


numpy = pytest.importorskip("numpy", reason="one-hot helpers need NumPy")


class TestOneHot:
    def test_encode_decode_round_trip(self):
        from helioseq.shuffle.ml import one_hot_decode, one_hot_encode

        sequence = "ACGTACGT"
        assert one_hot_decode(one_hot_encode(sequence)) == sequence

    def test_unknown_residues_decode_to_n(self):
        from helioseq.shuffle.ml import one_hot_decode, one_hot_encode

        assert one_hot_decode(one_hot_encode("ACNT")) == "ACNT".replace("N", "N")

    def test_dinuc_shuffle_on_one_hot(self):
        from helioseq.shuffle.ml import dinuc_shuffle, one_hot_decode, one_hot_encode

        sequence = "ACGTACGTAGCTAGCTAAGG"
        array = one_hot_encode(sequence)
        result = dinuc_shuffle(array, rng=1)
        assert result.shape == array.shape
        assert klets_preserved(sequence, one_hot_decode(result), 2)

    def test_stacked_output(self):
        from helioseq.shuffle.ml import dinuc_shuffle, one_hot_encode

        array = one_hot_encode("ACGTACGTAGCTAGCT")
        result = dinuc_shuffle(array, 4, rng=1)
        assert result.shape == (4, 16, 4)

    def test_shuffle_onehot_batch(self):
        from helioseq.shuffle.ml import one_hot_encode, shuffle_onehot

        arrays = [one_hot_encode("ACGTACGTAGCTAGCT") for _ in range(3)]
        result = shuffle_onehot(arrays, n=2, seed=1)
        assert result.shape == (3, 2, 16, 4)

    def test_wrong_shape_is_rejected(self):
        from helioseq.shuffle.ml import one_hot_decode

        with pytest.raises(ValueError, match="expected an"):
            one_hot_decode(numpy.zeros((4, 7)))
