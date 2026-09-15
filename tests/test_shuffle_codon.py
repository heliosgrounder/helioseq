"""Frame-aware null models."""

from __future__ import annotations

import pytest

from helioseq.seq import codon_counts, codon_table, translate
from helioseq.shuffle.codon import (
    shuffle_codons,
    shuffle_synonymous,
    shuffle_third_positions,
)

CDS = "ATGGCTGCAGCCGCGGGTTTAAAGCTGCGTCGCCGGTAA"[:39]


class TestGeneticCode:
    def test_standard_table_spot_checks(self):
        table = codon_table(1)
        assert table["ATG"] == "M"
        assert table["TGG"] == "W"
        assert table["TAA"] == table["TAG"] == table["TGA"] == "*"
        assert table["TTT"] == table["TTC"] == "F"
        assert {table[c] for c in ("CGT", "CGC", "CGA", "CGG", "AGA", "AGG")} == {"R"}

    def test_table_is_complete(self):
        assert len(codon_table(1)) == 64

    def test_mitochondrial_differences(self):
        standard, mito = codon_table(1), codon_table(2)
        assert standard["TGA"] == "*" and mito["TGA"] == "W"
        assert standard["ATA"] == "I" and mito["ATA"] == "M"
        assert standard["AGA"] == "R" and mito["AGA"] == "*"

    def test_unknown_table(self):
        with pytest.raises(ValueError, match="not bundled"):
            codon_table(999)

    def test_translate(self):
        assert translate("ATGGCTGCAGGTTTAAAGCTGTAA") == "MAAGLKL*"

    def test_out_of_frame_is_rejected(self):
        with pytest.raises(ValueError, match="multiple of 3"):
            translate("ATGGC")


class TestShuffleCodons:
    def test_codon_usage_is_preserved(self):
        result = shuffle_codons(CDS, 1, seed=1)
        assert codon_counts(result) == codon_counts(CDS)

    def test_length_and_frame_are_preserved(self):
        result = shuffle_codons(CDS, 1, seed=1)
        assert len(result) == len(CDS) and len(result) % 3 == 0

    def test_k_two_preserves_codon_pairs(self):
        from collections import Counter

        def pairs(sequence):
            codons = [sequence[i : i + 3] for i in range(0, len(sequence), 3)]
            return Counter(zip(codons, codons[1:]))

        sequence = "ATGGCTGCAGCCGCGGGTTTAAAGCTGGCTGCAGCCTAA"
        result = shuffle_codons(sequence, 2, seed=2)
        assert pairs(result) == pairs(sequence)

    def test_out_of_frame_is_rejected(self):
        with pytest.raises(ValueError, match="multiple of 3"):
            shuffle_codons("ATGGC", 1)

    def test_bytes_round_trip(self):
        assert isinstance(shuffle_codons(CDS.encode(), 1, seed=1), bytes)


class TestShuffleSynonymous:
    def test_protein_is_identical(self):
        for seed in range(10):
            assert translate(shuffle_synonymous(CDS, seed=seed)) == translate(CDS)

    def test_codon_usage_is_identical(self):
        assert codon_counts(shuffle_synonymous(CDS, seed=1)) == codon_counts(CDS)

    def test_it_actually_moves_something(self):
        sequence = "ATGGCTGCAGCCGCGCGTCGCCGGAGAAGGTAA"
        results = {shuffle_synonymous(sequence, seed=s) for s in range(30)}
        assert len(results) > 1

    def test_reproducible(self):
        assert shuffle_synonymous(CDS, seed=5) == shuffle_synonymous(CDS, seed=5)


class TestShuffleThirdPositions:
    def test_protein_is_identical(self):
        for seed in range(10):
            assert translate(shuffle_third_positions(CDS, seed=seed)) == translate(CDS)

    def test_length_is_preserved(self):
        assert len(shuffle_third_positions(CDS, seed=1)) == len(CDS)

    def test_reproducible(self):
        assert shuffle_third_positions(CDS, seed=2) == shuffle_third_positions(
            CDS, seed=2
        )
