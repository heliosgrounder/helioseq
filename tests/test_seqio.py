"""Streaming FASTA/FASTQ reading and writing."""

from __future__ import annotations

import gzip
import io

import pytest

from helioseq.seqio import (
    FORMATS,
    Record,
    guess_format,
    read_fasta,
    read_fastq,
    read_sequences,
    write_fasta,
    write_fastq,
    write_record,
)


class TestFastaReading:
    def test_multiline_records_are_joined(self, fasta_file):
        records = list(read_sequences(fasta_file))
        assert len(records) == 3
        assert records[0].id == "seq1"
        assert records[0].description == "first record"
        assert records[0].sequence == "ACGTACGTAGCTAGCTAAGGCCTTACGTACGT"

    def test_records_without_a_description(self, fasta_file):
        assert list(read_sequences(fasta_file))[2].description == ""

    def test_blank_lines_are_ignored(self):
        text = ">a\n\nACGT\n\n>b\nTTTT\n"
        records = list(read_fasta(io.StringIO(text)))
        assert [r.sequence for r in records] == ["ACGT", "TTTT"]

    def test_data_before_a_header_is_an_error(self):
        with pytest.raises(ValueError, match="before the first"):
            list(read_fasta(io.StringIO("ACGT\n>a\nACGT\n")))

    def test_empty_input(self):
        assert list(read_fasta(io.StringIO(""))) == []

    def test_gzip_is_detected_by_magic_not_extension(self, tmp_path):
        path = tmp_path / "compressed.fa"  # deliberately not .gz
        with gzip.open(path, "wt") as handle:
            handle.write(">a\nACGTACGT\n")
        assert [r.sequence for r in read_sequences(path)] == ["ACGTACGT"]


class TestFastqReading:
    def test_reads_records(self, fastq_file):
        records = list(read_sequences(fastq_file))
        assert len(records) == 2
        assert records[0].id == "read1"
        assert records[0].quality == "I" * 16

    def test_length_mismatch_is_an_error(self):
        text = "@a\nACGT\n+\nII\n"
        with pytest.raises(ValueError, match="differ in length"):
            list(read_fastq(io.StringIO(text)))

    def test_truncated_record_is_an_error(self):
        with pytest.raises(ValueError, match="truncated"):
            list(read_fastq(io.StringIO("@a\nACGT\n+\n")))

    def test_missing_separator_is_an_error(self):
        with pytest.raises(ValueError, match="separator"):
            list(read_fastq(io.StringIO("@a\nACGT\n-\nIIII\n")))


class TestFormatDetection:
    def test_fasta(self, fasta_file):
        assert guess_format(fasta_file) == "fasta"

    def test_fastq(self, fastq_file):
        assert guess_format(fastq_file) == "fastq"

    def test_garbage(self, tmp_path):
        path = tmp_path / "junk.txt"
        path.write_text("this is not a sequence file\n")
        with pytest.raises(ValueError, match="neither FASTA nor FASTQ"):
            guess_format(path)

    def test_stdin_needs_an_explicit_format(self):
        with pytest.raises(ValueError, match="standard input"):
            list(read_sequences("-"))


class TestWriting:
    def test_fasta_wrapping(self):
        handle = io.StringIO()
        write_fasta(handle, Record("a", "ACGT" * 30), wrap=60)
        lines = handle.getvalue().splitlines()
        assert lines[0] == ">a"
        assert all(len(line) <= 60 for line in lines[1:])
        assert "".join(lines[1:]) == "ACGT" * 30

    def test_fasta_without_wrapping(self):
        handle = io.StringIO()
        write_fasta(handle, Record("a", "ACGT" * 30), wrap=0)
        assert len(handle.getvalue().splitlines()) == 2

    def test_fastq_fills_missing_quality(self):
        handle = io.StringIO()
        write_fastq(handle, Record("a", "ACGT"))
        assert handle.getvalue() == "@a\nACGT\n+\nIIII\n"

    def test_replaced_drops_stale_quality(self):
        record = Record("a", "ACGT", quality="IIII")
        assert record.replaced("ACG").quality is None
        assert record.replaced("TGCA").quality == "IIII"


class TestWriteRecord:
    """The format-dispatching writer that the CLI and future formats use."""

    def test_dispatches_by_format(self):
        handle = io.StringIO()
        write_record(handle, Record("a", "ACGT"), "fasta")
        assert handle.getvalue().startswith(">a")

        handle = io.StringIO()
        write_record(handle, Record("a", "ACGT"), "fastq")
        assert handle.getvalue().startswith("@a")

    def test_an_option_one_format_ignores_is_still_accepted(self):
        """`wrap` means something to FASTA and nothing to FASTQ; a caller
        emitting both should not have to branch."""
        handle = io.StringIO()
        write_record(handle, Record("a", "ACGT"), "fastq", wrap=0)
        assert handle.getvalue() == "@a\nACGT\n+\nIIII\n"

    def test_a_genuine_typo_still_raises(self):
        with pytest.raises(TypeError, match="unknown write option"):
            write_record(io.StringIO(), Record("a", "ACGT"), "fasta", wrapp=60)

    def test_unknown_format(self):
        with pytest.raises(ValueError, match="format must be one of"):
            write_record(io.StringIO(), Record("a", "ACGT"), "embl")

    def test_formats_are_listed(self):
        assert set(FORMATS) == {"fasta", "fastq"}

