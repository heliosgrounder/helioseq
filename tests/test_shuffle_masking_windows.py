"""Masking and windowed shuffling: the annotations must survive."""

from __future__ import annotations

import pytest

from helioseq.shuffle import klets_preserved
from helioseq.shuffle.masking import apply_case, case_profile, free_spans, shuffle_masked
from helioseq.shuffle.windows import (
    shuffle_segments,
    shuffle_windows,
    tile_spans,
    window_report,
)

SOFT_MASKED = "acgtacgtACGTACGTNNNNNacgtACGTACGTAAGGCC"


class TestFreeSpans:
    def test_finds_runs_between_frozen_positions(self):
        assert free_spans(b"ACGTNNNNACGT") == [(0, 4), (8, 12)]

    def test_case_insensitive_by_default(self):
        assert free_spans(b"ACGTnnnnACGT") == [(0, 4), (8, 12)]

    def test_case_sensitive_mode(self):
        assert free_spans(b"ACGTnnnnACGT", case_sensitive=True) == [(0, 12)]

    def test_leading_and_trailing_runs(self):
        assert free_spans(b"NNACGTNN") == [(2, 6)]
        assert free_spans(b"NNNN") == []
        assert free_spans(b"ACGT") == [(0, 4)]


class TestCaseProfile:
    def test_round_trips(self):
        profile = case_profile(SOFT_MASKED.encode())
        restored = apply_case(SOFT_MASKED.upper().encode(), profile)
        assert restored.decode() == SOFT_MASKED

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="length"):
            apply_case(b"ACGT", b"\x00\x00")


class TestShuffleMasked:
    def test_n_runs_stay_put(self):
        result = shuffle_masked(SOFT_MASKED, 2, keep="N", seed=1)
        for position, residue in enumerate(SOFT_MASKED):
            if residue.upper() == "N":
                assert result[position] == residue

    def test_no_n_is_introduced_or_lost(self):
        result = shuffle_masked(SOFT_MASKED, 2, keep="N", seed=1)
        assert result.upper().count("N") == SOFT_MASKED.upper().count("N")

    def test_preserve_keeps_the_masking_pattern(self):
        result = shuffle_masked(SOFT_MASKED, 2, case="preserve", seed=2)
        assert [c.islower() for c in result] == [c.islower() for c in SOFT_MASKED]

    def test_split_keeps_residues_inside_their_case_run(self):
        result = shuffle_masked(SOFT_MASKED, 2, case="split", seed=2)
        assert [c.islower() for c in result] == [c.islower() for c in SOFT_MASKED]
        # masked and unmasked composition are each preserved exactly
        assert sorted(c for c in result if c.islower()) == sorted(
            c for c in SOFT_MASKED if c.islower()
        )

    def test_ignore_uppercases(self):
        result = shuffle_masked(SOFT_MASKED, 2, case="ignore", seed=2)
        assert result == result.upper()

    def test_length_and_composition_are_unchanged(self):
        for case in ("preserve", "split", "ignore"):
            result = shuffle_masked(SOFT_MASKED, 2, case=case, seed=3)
            assert len(result) == len(SOFT_MASKED)
            assert sorted(result.upper()) == sorted(SOFT_MASKED.upper())

    def test_invalid_case_mode(self):
        with pytest.raises(ValueError, match="case must be one of"):
            shuffle_masked("ACGT", 2, case="nonsense")

    def test_keep_nothing_is_a_plain_shuffle(self):
        sequence = "ACGTACGTAGCTAGCTAAGG"
        result = shuffle_masked(sequence, 2, keep="", case="ignore", seed=4)
        assert klets_preserved(sequence, result, 2)


class TestTileSpans:
    def test_exact_division(self):
        assert tile_spans(9, 3) == [(0, 3), (3, 6), (6, 9)]

    def test_short_tail_is_merged(self):
        assert tile_spans(11, 5, min_window=3) == [(0, 5), (5, 11)]

    def test_short_tail_is_kept_without_min_window(self):
        assert tile_spans(11, 5) == [(0, 5), (5, 10), (10, 11)]

    def test_empty_and_invalid(self):
        assert tile_spans(0, 5) == []
        with pytest.raises(ValueError, match="window must be"):
            tile_spans(10, 0)


class TestShuffleWindows:
    def test_length_and_composition(self):
        sequence = "".join("ACGT"[(i * 7) % 4] for i in range(200))
        result = shuffle_windows(sequence, 2, window=40, seed=1)
        assert len(result) == len(sequence)
        assert sorted(result) == sorted(sequence)

    def test_local_composition_is_preserved(self):
        """The reason to window at all: a GC-rich half stays GC-rich."""
        sequence = "AT" * 100 + "GC" * 100
        result = shuffle_windows(sequence, 1, window=50, seed=1)
        assert set(result[:200]) == {"A", "T"}
        assert set(result[200:]) == {"G", "C"}

    def test_global_shuffle_would_mix_them(self):
        from helioseq.shuffle import shuffle

        sequence = "AT" * 100 + "GC" * 100
        assert set(shuffle(sequence, 1, seed=1)[:200]) > {"A", "T"}

    def test_report_is_honest_about_boundaries(self):
        text = window_report(200, 40, 2)
        assert "4 boundaries" in text


class TestShuffleSegments:
    def test_untouched_outside_spans(self):
        sequence = "AAAACCCCGGGGTTTT"
        result = shuffle_segments(sequence, 1, spans=[(4, 12)], seed=1)
        assert result[:4] == "AAAA"
        assert result[12:] == "TTTT"
        assert sorted(result[4:12]) == sorted("CCCCGGGG")

    def test_overlapping_spans_are_rejected(self):
        with pytest.raises(ValueError, match="sorted and non-overlapping"):
            shuffle_segments("ACGTACGT", 2, spans=[(0, 5), (3, 8)])

    def test_out_of_range_spans_are_rejected(self):
        with pytest.raises(ValueError, match="outside the sequence"):
            shuffle_segments("ACGT", 2, spans=[(0, 99)])

    def test_bytes_round_trip(self):
        result = shuffle_segments(b"AAAACCCCGGGG", 1, spans=[(0, 12)], seed=1)
        assert isinstance(result, bytes)
