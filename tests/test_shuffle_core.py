"""The invariants the whole package rests on."""

from __future__ import annotations

import pytest

import helioseq
from helioseq.shuffle import Shuffler, klet_counts, klets_preserved, shuffle
from helioseq.shuffle import shuffle_batch, shuffle_many


class TestKletInvariant:
    """Preserving k-let counts is the contract; check it from every angle."""

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_counts_are_identical(self, dna_sequences, k):
        for sequence in dna_sequences:
            result = shuffle(sequence, k, seed=1)
            assert klets_preserved(sequence, result, k), (sequence, k)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_lower_order_counts_follow(self, dna_sequences, k):
        """Preserving k-lets implies preserving every j-let for j < k."""
        for sequence in dna_sequences:
            result = shuffle(sequence, k, seed=2)
            for j in range(1, k):
                assert klets_preserved(sequence, result, j), (sequence, k, j)

    def test_works_on_protein(self, protein_sequence):
        result = shuffle(protein_sequence, 2, seed=3)
        assert klets_preserved(protein_sequence, result, 2)

    def test_length_is_unchanged(self, dna_sequences):
        for sequence in dna_sequences:
            assert len(shuffle(sequence, 2, seed=4)) == len(sequence)

    def test_prefix_and_suffix_are_fixed(self):
        """uShuffle keeps the first (k-1)-let; the last is then determined."""
        sequence = "ACGTTGCAACGTAGCTAGCTTTACG"
        for k in (2, 3, 4):
            shuffler = Shuffler(sequence, k, seed=5)
            for _ in range(50):
                result = shuffler.shuffle()
                assert result[: k - 1] == sequence[: k - 1]
                assert result[-(k - 1) :] == sequence[-(k - 1) :]


class TestSequenceTypes:
    """uShuffle 1.x only accepted bytes; str raised TypeError (issue #4)."""

    def test_str_in_str_out(self):
        assert isinstance(shuffle("ACGTACGT", 2, seed=1), str)

    def test_bytes_in_bytes_out(self):
        assert isinstance(shuffle(b"ACGTACGT", 2, seed=1), bytes)

    def test_bytearray_round_trips(self):
        result = shuffle(bytearray(b"ACGTACGT"), 2, seed=1)
        assert isinstance(result, bytearray)

    def test_memoryview_is_accepted(self):
        result = shuffle(memoryview(b"ACGTACGT"), 2, seed=1)
        assert isinstance(result, bytes)

    def test_str_and_bytes_agree(self):
        sequence = "ACGTACGTAGCTAGCTAAGG"
        assert shuffle(sequence, 2, seed=9).encode() == shuffle(
            sequence.encode(), 2, seed=9
        )

    def test_list_comprehension_over_str(self):
        """The exact shape that used to raise UnicodeDecodeError."""
        sequences = ["ACGTACGT", "GGCCTTAA", "ACACACGT"]
        assert [shuffle(s, 2, seed=1) for s in sequences] == [
            shuffle(s, 2, seed=1) for s in sequences
        ]

    def test_non_ascii_is_rejected_clearly(self):
        with pytest.raises(ValueError, match="non-ASCII"):
            shuffle("ACGTé", 2)

    def test_wrong_type_is_rejected(self):
        with pytest.raises(TypeError, match="str, bytes"):
            shuffle(["A", "C"], 2)

    def test_embedded_nul_is_not_a_terminator(self):
        """1.x used strlen() and silently truncated at the first NUL byte."""
        sequence = b"AC\x00GTAC\x00GT"
        result = shuffle(sequence, 2, seed=1)
        assert len(result) == len(sequence)
        assert klets_preserved(sequence, result, 2)


class TestEdgeCases:
    @pytest.mark.parametrize("sequence", ["", "A", "AC"])
    def test_tiny_sequences(self, sequence):
        assert shuffle(sequence, 2, seed=1) == sequence

    def test_k_at_or_above_length_returns_the_input(self):
        assert shuffle("ACGT", 4, seed=1) == "ACGT"
        assert shuffle("ACGT", 99, seed=1) == "ACGT"

    def test_k_one_is_a_plain_permutation(self):
        sequence = "AAAACCCCGGGGTTTT"
        result = shuffle(sequence, 1, seed=1)
        assert sorted(result) == sorted(sequence)
        assert result != sequence  # astronomically unlikely to tie

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_below_one_is_rejected(self, k):
        with pytest.raises(ValueError, match="k must be >= 1"):
            shuffle("ACGT", k)

    def test_homopolymer_has_exactly_one_arrangement(self):
        assert shuffle("AAAAAAAA", 2, seed=1) == "AAAAAAAA"


class TestReproducibility:
    def test_same_seed_same_result(self, dna_sequences):
        for sequence in dna_sequences:
            assert shuffle(sequence, 2, seed=42) == shuffle(sequence, 2, seed=42)

    def test_different_seeds_differ(self):
        sequence = "ACGTACGTAGCTAGCTAAGGCCTTACGT"
        results = {shuffle(sequence, 2, seed=s) for s in range(20)}
        assert len(results) > 10

    def test_streams_are_independent(self):
        sequence = "ACGTACGTAGCTAGCTAAGGCCTTACGT"
        a = Shuffler(sequence, 2, seed=1, stream=0).shuffle_many(20)
        b = Shuffler(sequence, 2, seed=1, stream=1).shuffle_many(20)
        assert a != b

    def test_set_seed_makes_a_script_reproducible(self):
        sequence = "ACGTACGTAGCTAGCTAAGG"
        helioseq.shuffle.set_seed(7)
        first = [shuffle(sequence, 2) for _ in range(5)]
        helioseq.shuffle.set_seed(7)
        second = [shuffle(sequence, 2) for _ in range(5)]
        assert first == second
        helioseq.shuffle.set_seed(None)

    def test_reseed_keeps_the_prepared_graph(self):
        shuffler = Shuffler("ACGTACGTAGCTAGCTAAGG", 2, seed=1)
        first = shuffler.shuffle_many(5)
        shuffler.reseed(1)
        assert shuffler.shuffle_many(5) == first


class TestShufflerIsolation:
    """Regression tests for guma44/ushuffle#7: 1.x kept the whole graph in C
    globals, so a second Shuffler corrupted the first one's heap."""

    def test_two_shufflers_do_not_interfere(self):
        short = Shuffler("ACGTACGTAC", 2, seed=1)
        long = Shuffler("GGGGCCCCGGGGCCCCGGGGCCCC", 2, seed=2)
        for _ in range(100):
            a = short.shuffle()
            b = long.shuffle()
            assert len(a) == 10 and klets_preserved("ACGTACGTAC", a, 2)
            assert len(b) == 24

    def test_interleaved_construction_and_use(self):
        first = Shuffler("ACGTACGTAGCTAGCT", 2, seed=1)
        baseline = first.shuffle()
        first.reseed(1)
        Shuffler("TTTTAAAACCCCGGGG", 3, seed=9).shuffle()
        assert first.shuffle() == baseline

    def test_many_live_shufflers(self):
        sequences = ["ACGT" * (i + 2) for i in range(30)]
        shufflers = [Shuffler(s, 2, seed=i) for i, s in enumerate(sequences)]
        for sequence, shuffler in zip(sequences, shufflers):
            result = shuffler.shuffle()
            assert len(result) == len(sequence)
            assert klets_preserved(sequence, result, 2)


class TestShufflerApi:
    def test_properties(self):
        shuffler = Shuffler("ACGTACGTAGCT", 3, seed=11)
        assert shuffler.k == 3
        assert shuffler.length == 12 == len(shuffler)
        assert shuffler.sequence == "ACGTACGTAGCT"
        assert shuffler.seed == 11
        assert shuffler.n_vertices > 0

    def test_iteration(self):
        import itertools

        shuffler = Shuffler("ACGTACGTAGCTAGCT", 2, seed=1)
        assert len(list(itertools.islice(shuffler, 7))) == 7

    def test_shuffle_many_matches_repeated_shuffle(self):
        a = Shuffler("ACGTACGTAGCTAGCT", 2, seed=3).shuffle_many(5)
        shuffler = Shuffler("ACGTACGTAGCTAGCT", 2, seed=3)
        assert a == [shuffler.shuffle() for _ in range(5)]

    def test_shuffle_many_rejects_negative(self):
        with pytest.raises(ValueError):
            shuffle_many("ACGT", 2, -1)

    def test_shuffle_raw_returns_bytes(self):
        assert isinstance(Shuffler("ACGTACGT", 2, seed=1).shuffle_raw(), bytes)


class TestBatch:
    def test_shape(self):
        result = shuffle_batch(["ACGTACGT", "GGCCTTAA"], 2, n=3, seed=1)
        assert len(result) == 2
        assert all(len(item) == 3 for item in result)

    def test_threads_do_not_change_the_result(self):
        sequences = ["".join("ACGT"[(i * j) % 4] for j in range(60)) for i in range(20)]
        single = shuffle_batch(sequences, 2, n=2, seed=5, threads=1)
        many = shuffle_batch(sequences, 2, n=2, seed=5, threads=8)
        assert single == many

    def test_every_output_preserves_klets(self):
        sequences = ["ACGTACGTAGCTAGCT", "AACAGATAACAGAT", "TTTTAAAACCCC"]
        for sequence, outputs in zip(
            sequences, shuffle_batch(sequences, 2, n=4, seed=1)
        ):
            for result in outputs:
                assert klets_preserved(sequence, result, 2)

    def test_empty_input(self):
        assert shuffle_batch([], 2) == []

    def test_threads_must_be_positive(self):
        with pytest.raises(ValueError):
            shuffle_batch(["ACGT"], 2, threads=0)


def test_klet_counts_are_readable():
    assert dict(klet_counts("ACGTACG", 2)) == {"AC": 2, "CG": 2, "GT": 1, "TA": 1}


def test_backend_is_reported():
    assert helioseq.shuffle.backend() in ("c", "python")
    assert isinstance(helioseq.shuffle.core_version(), str)


def test_memory_estimate_grows_with_length():
    assert helioseq.shuffle.memory_estimate(1_000_000, 2) > helioseq.shuffle.memory_estimate(1000, 2)
