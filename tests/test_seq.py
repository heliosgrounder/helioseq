"""Sequence primitives: types, alphabets, the genetic code, transformations."""

from __future__ import annotations

import pytest

from helioseq.seq import (
    DNA,
    PROTEIN,
    RNA,
    as_bytes,
    codon_counts,
    codon_table,
    coerce,
    complement,
    composition,
    detect,
    gc_fraction,
    get,
    reverse_complement,
    split_codons,
    translate,
    validate,
)


class TestTypes:
    @pytest.mark.parametrize(
        "value,expected",
        [("ACGT", str), (b"ACGT", bytes), (bytearray(b"ACGT"), bytearray)],
    )
    def test_round_trip_preserves_the_input_type(self, value, expected):
        raw, restore = coerce(value)
        assert isinstance(restore(raw), expected)

    def test_memoryview_becomes_bytes(self):
        raw, restore = coerce(memoryview(b"ACGT"))
        assert isinstance(restore(raw), bytes)

    def test_as_bytes(self):
        assert as_bytes("ACGT") == b"ACGT"

    def test_non_ascii_is_rejected_with_the_position(self):
        with pytest.raises(ValueError, match="position 4"):
            coerce("ACGTé")

    def test_wrong_type_names_what_is_accepted(self):
        with pytest.raises(TypeError, match="str, bytes, bytearray or memoryview"):
            coerce(42)

    def test_argument_name_appears_in_the_error(self):
        with pytest.raises(TypeError, match="motif"):
            coerce(42, argument="motif")


class TestAlphabet:
    def test_detects_dna_rna_protein(self):
        assert detect("ACGTACGTACGT") is DNA
        assert detect("ACGUACGUACGU") is RNA
        assert detect("MKVLAAGIVGLNLWWYYHHQQ") is PROTEIN

    def test_returns_none_for_junk(self):
        assert detect("!!!!!!!!!!") is None
        assert detect("") is None

    def test_get_by_name(self):
        assert get("DNA") is DNA
        with pytest.raises(ValueError, match="unknown alphabet"):
            get("klingon")

    def test_validate_accepts_ambiguity_codes(self):
        assert validate("ACGTNNRYACGT", DNA) is DNA

    def test_validate_can_refuse_ambiguity(self):
        with pytest.raises(ValueError, match="outside the dna alphabet"):
            validate("ACGTNN", DNA, ambiguous=False)

    def test_validate_reports_the_position(self):
        with pytest.raises(ValueError, match="at 4"):
            validate("ACGT@CGT", DNA)

    def test_validate_gaps(self):
        with pytest.raises(ValueError):
            validate("ACGT--ACGT", DNA)
        assert validate("ACGT--ACGT", DNA, gaps=True) is DNA

    def test_validate_detects_when_not_told(self):
        assert validate("ACGTACGT") is DNA

    def test_validate_refuses_to_guess_at_junk(self):
        with pytest.raises(ValueError, match="could not detect"):
            validate("@@@@@@@@")

    def test_lowercase_is_accepted(self):
        assert validate("acgtACGT", DNA) is DNA


class TestGeneticCode:
    def test_standard_table_spot_checks(self):
        table = codon_table(1)
        assert table["ATG"] == "M"
        assert table["TGG"] == "W"
        assert table["TAA"] == table["TAG"] == table["TGA"] == "*"
        assert {table[c] for c in ("CGT", "CGC", "CGA", "CGG", "AGA", "AGG")} == {"R"}

    def test_table_is_complete(self):
        assert len(codon_table(1)) == 64

    def test_mitochondrial_differences(self):
        standard, mito = codon_table(1), codon_table(2)
        assert standard["TGA"] == "*" and mito["TGA"] == "W"
        assert standard["ATA"] == "I" and mito["ATA"] == "M"
        assert standard["AGA"] == "R" and mito["AGA"] == "*"

    def test_table_11_matches_table_1(self):
        assert codon_table(11) == codon_table(1)

    def test_unknown_table(self):
        with pytest.raises(ValueError, match="not bundled"):
            codon_table(999)

    def test_translate(self):
        assert translate("ATGGCTGCAGGTTTAAAGCTGTAA") == "MAAGLKL*"

    def test_translate_is_case_insensitive(self):
        assert translate("atggctgcataa") == translate("ATGGCTGCATAA")

    def test_unknown_codons_become_x(self):
        assert translate("ATGNNNTAA") == "M X *".replace(" ", "")

    def test_out_of_frame_is_rejected(self):
        with pytest.raises(ValueError, match="multiple of 3"):
            translate("ATGGC")

    def test_split_codons(self):
        assert split_codons("ATGGCT") == [b"ATG", b"GCT"]

    def test_codon_counts(self):
        assert codon_counts("ATGATGTAA") == {"ATG": 2, "TAA": 1}


class TestTransform:
    def test_complement_keeps_order_and_case(self):
        assert complement("ACGTn") == "TGCAn"

    def test_reverse_complement(self):
        assert reverse_complement("ACGTTG") == "CAACGT"

    def test_reverse_complement_is_an_involution(self):
        sequence = "ACGTTGCAnnACGT"
        assert reverse_complement(reverse_complement(sequence)) == sequence

    def test_ambiguity_codes_are_complemented(self):
        assert complement("RYSWKM") == "YRSWMK"

    def test_type_is_preserved(self):
        assert isinstance(reverse_complement(b"ACGT"), bytes)
        assert isinstance(reverse_complement("ACGT"), str)

    def test_composition(self):
        assert dict(composition("AACGT")) == {"A": 2, "C": 1, "G": 1, "T": 1}

    def test_composition_case_folding(self):
        assert dict(composition("aA")) == {"A": 2}
        assert dict(composition("aA", upper=False)) == {"a": 1, "A": 1}

    def test_gc_fraction(self):
        assert gc_fraction("ACGT") == 0.5
        assert gc_fraction("GGCC") == 1.0
        assert gc_fraction("AATT") == 0.0

    def test_gc_fraction_of_empty_is_nan(self):
        import math

        assert math.isnan(gc_fraction(""))
