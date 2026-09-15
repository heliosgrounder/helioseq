"""Position weight matrices and motif scoring."""

from __future__ import annotations

import pytest

from helioseq.motifs import best_score, consensus, from_counts, scan

# A strong AA motif: A scores well, everything else badly.
AA = [{"A": 2.0, "C": -1.0, "G": -1.0, "T": -1.0}] * 2


class TestFromCounts:
    def test_converts_counts_to_log_odds(self):
        pwm = from_counts([{"A": 40, "C": 0, "G": 0, "T": 0}])
        assert pwm[0]["A"] > 0 > pwm[0]["C"]

    def test_pseudocount_prevents_minus_infinity(self):
        pwm = from_counts([{"A": 40, "C": 0, "G": 0, "T": 0}])
        assert all(score != float("-inf") for score in pwm[0].values())

    def test_uniform_counts_score_about_zero(self):
        pwm = from_counts([{"A": 25, "C": 25, "G": 25, "T": 25}])
        assert all(abs(score) < 0.05 for score in pwm[0].values())

    def test_background_shifts_the_scores(self):
        at_rich = {"A": 0.35, "C": 0.15, "G": 0.15, "T": 0.35}
        counts = [{"A": 25, "C": 25, "G": 25, "T": 25}]
        assert from_counts(counts, background=at_rich)[0]["C"] > 0


class TestConsensus:
    def test_picks_the_best_residue_per_position(self):
        assert consensus([{"A": 2.0, "C": -1.0}, {"A": -1.0, "C": 2.0}]) == "AC"

    def test_round_trips_through_from_counts(self):
        counts = [
            {"A": 40, "C": 1, "G": 1, "T": 1},
            {"A": 1, "C": 1, "G": 40, "T": 1},
        ]
        assert consensus(from_counts(counts)) == "AG"


class TestScan:
    def test_one_score_per_start_position(self):
        assert len(scan(AA, "ACGTACGT", both_strands=False)) == 7

    def test_short_input_gives_no_scores(self):
        assert scan(AA, "A") == []

    def test_finds_the_motif(self):
        scores = scan(AA, "CCAACC", both_strands=False)
        assert max(scores) == pytest.approx(4.0)
        assert scores.index(max(scores)) == 2

    def test_unknown_residues_score_minus_infinity(self):
        assert scan(AA, "NN", both_strands=False) == [float("-inf")]

    def test_empty_pwm_is_rejected(self):
        with pytest.raises(ValueError, match="at least one position"):
            scan([], "ACGT")


class TestBestScore:
    def test_is_a_statistic(self):
        statistic = best_score(AA, both_strands=False)
        assert statistic("CCAACC") == pytest.approx(4.0)
        assert "pwm_best_score" in statistic.__name__

    def test_both_strands_finds_the_reverse_complement(self):
        assert best_score(AA, both_strands=True)("CCTTCC") == pytest.approx(4.0)

    def test_one_strand_does_not(self):
        assert best_score(AA, both_strands=False)("CCTTCC") < 4.0

    def test_short_input(self):
        assert best_score([{"A": 1.0}] * 5)("AC") == float("-inf")

    def test_works_as_a_null_test_statistic(self):
        from helioseq.stats import null_test

        planted = "ACTGTCAATCGTACATGCAT" + "AA" * 8 + "TCGATCAGTGACTGCATGCA"
        result = null_test(planted, best_score(AA), k=1, n=99, seed=1)
        assert 0 < result.pvalue <= 1
