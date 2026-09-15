"""Null distributions, p-values and the ready-made statistics."""

from __future__ import annotations

import math

import pytest

from helioseq.motifs import best_score as pwm_best_score
from helioseq.stats import (
    NullResult,
    cpg_observed_expected,
    gc_content,
    kmer_frequency,
    longest_homopolymer,
    null_distribution,
    null_test,
)


class TestPValue:
    """The add-one correction is the whole reason this module exists."""

    def test_never_returns_zero(self):
        result = NullResult(observed=100.0, null=[0.0] * 999)
        assert result.pvalue == pytest.approx(1 / 1000)
        assert result.pvalue > 0

    def test_resolution_limit_is_reported(self):
        result = NullResult(observed=100.0, null=[0.0] * 999)
        assert result.at_resolution_limit
        assert result.resolution == pytest.approx(1 / 1000)

    def test_observed_inside_the_null(self):
        result = NullResult(observed=5.0, null=[float(i) for i in range(11)])
        # 6 of 11 values are >= 5
        assert result.pvalue == pytest.approx(7 / 12)

    def test_less_alternative(self):
        result = NullResult(
            observed=1.0, null=[float(i) for i in range(11)], alternative="less"
        )
        assert result.pvalue == pytest.approx(3 / 12)  # values 0 and 1

    def test_two_sided_is_capped_at_one(self):
        result = NullResult(
            observed=5.0,
            null=[float(i) for i in range(11)],
            alternative="two-sided",
        )
        assert result.pvalue <= 1.0

    def test_rejects_unknown_alternative(self):
        with pytest.raises(ValueError, match="alternative must be"):
            null_test("ACGTACGT", gc_content, alternative="sideways")


class TestNullResultSummary:
    def test_zscore_and_moments(self):
        result = NullResult(observed=10.0, null=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.null_mean == pytest.approx(3.0)
        assert result.zscore > 0
        assert result.n == 5

    def test_zscore_is_nan_for_a_constant_null(self):
        result = NullResult(observed=1.0, null=[2.0] * 10)
        assert math.isnan(result.zscore)

    def test_quantiles_and_interval(self):
        result = NullResult(observed=0.0, null=[float(i) for i in range(101)])
        assert result.quantile(0.5) == pytest.approx(50.0)
        low, high = result.ci95
        assert low < 50 < high

    def test_skew_warning(self):
        skewed = [0.0] * 90 + [100.0] * 10
        result = NullResult(observed=1.0, null=skewed)
        assert "skewed" in (result.normality_warning or "")

    def test_summary_mentions_everything_important(self):
        result = NullResult(observed=10.0, null=[1.0] * 50 + [2.0] * 50)
        text = result.summary()
        assert "observed" in text and "p-value" in text and "z-score" in text


class TestNullTest:
    def test_gc_has_no_variance_under_a_dinucleotide_shuffle(self):
        sequence = "ACGTACGTAGCTAGCTAAGGCCTT" * 4
        result = null_test(sequence, gc_content, k=2, n=99, seed=1)
        assert result.null_sd == 0.0
        assert result.observed == pytest.approx(result.null_mean)

    def test_gc_does_vary_under_nothing(self):
        """k=1 preserves composition too, so this is a sanity check that the
        statistic is not accidentally constant for the wrong reason."""
        sequence = "ACGTACGTAGCTAGCTAAGGCCTT"
        values = null_distribution(sequence, longest_homopolymer, k=1, n=50, seed=1)
        assert len(set(values)) > 1

    def test_detects_a_planted_motif(self):
        background = "ACTACTGTCAATCGTACATGCATACGTTAGCATGCAATCGATCAGTGACT"
        sequence = background + "GGGGCGGGGC" * 4 + background
        result = null_test(sequence, kmer_frequency("GGGGCGGGGC"), k=2, n=299, seed=1)
        assert result.pvalue <= 0.01

    def test_threads_produce_the_same_multiset(self):
        sequence = "ACGTACGTAGCTAGCTAAGGCCTT" * 3
        single = sorted(
            null_distribution(sequence, longest_homopolymer, k=2, n=64, seed=5)
        )
        assert len(single) == 64
        threaded = sorted(
            null_distribution(
                sequence, longest_homopolymer, k=2, n=64, seed=5, threads=4
            )
        )
        assert len(threaded) == 64

    def test_reproducible(self):
        sequence = "ACGTACGTAGCTAGCTAAGG"
        first = null_distribution(sequence, longest_homopolymer, k=2, n=20, seed=3)
        second = null_distribution(sequence, longest_homopolymer, k=2, n=20, seed=3)
        assert first == second

    def test_custom_shuffle_function(self):
        from functools import partial

        from helioseq.shuffle.windows import shuffle_windows

        sequence = "AT" * 100 + "GC" * 100
        result = null_test(
            sequence,
            gc_content,
            n=20,
            seed=1,
            shuffle_fn=partial(shuffle_windows, window=50),
        )
        assert result.n == 20

    def test_n_must_be_positive(self):
        with pytest.raises(ValueError, match="n must be"):
            null_test("ACGT", gc_content, n=0)


class TestStatistics:
    def test_gc_content(self):
        assert gc_content("GGCC") == 1.0
        assert gc_content("AATT") == 0.0
        assert gc_content("ACGT") == 0.5
        assert math.isnan(gc_content(""))

    def test_cpg_observed_expected(self):
        assert cpg_observed_expected("CGCGCG") > 1.0
        assert cpg_observed_expected("GCGCGC") < cpg_observed_expected("CGCGCG")
        assert cpg_observed_expected("AAAA") == 0.0

    def test_kmer_frequency(self):
        assert kmer_frequency("CG")("ACGACG") == pytest.approx(2 / 6)
        assert kmer_frequency("CG")("AAAA") == 0.0

    def test_kmer_frequency_both_strands(self):
        assert kmer_frequency("AC", both_strands=True)("ACGT") == pytest.approx(2 / 4)

    def test_longest_homopolymer(self):
        assert longest_homopolymer("AACCCGT") == 3.0
        assert longest_homopolymer("") == 0.0

    def test_pwm_best_score(self):
        motif = [{"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0}] * 2
        assert pwm_best_score(motif, both_strands=False)("CCAACC") == pytest.approx(4.0)

    def test_pwm_handles_short_input(self):
        motif = [{"A": 1.0}] * 5
        assert pwm_best_score(motif)("AC") == float("-inf")

    def test_pwm_both_strands_finds_the_reverse_complement(self):
        motif = [
            {"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0},
            {"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0},
        ]
        assert pwm_best_score(motif, both_strands=True)("CCTTCC") == pytest.approx(4.0)

    def test_empty_pwm_is_rejected(self):
        with pytest.raises(ValueError, match="at least one position"):
            pwm_best_score([])
