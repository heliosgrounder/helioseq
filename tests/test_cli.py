"""The ``helioseq`` command: dispatch, subcommands, end-to-end behaviour."""

from __future__ import annotations

import pytest

from helioseq.cli import COMMANDS, main
from helioseq.seqio import read_sequences
from helioseq.shuffle import klets_preserved


class TestCli:

    def test_shuffle_writes_every_record(self, fasta_file, tmp_path, capsys):
        out = tmp_path / "out.fa"
        assert main(["shuffle", "-i", str(fasta_file), "-o", str(out),
                     "-k", "2", "-n", "3", "--seed", "1"]) == 0
        records = list(read_sequences(out))
        assert len(records) == 9  # 3 records x 3 shuffles

    def test_name_template(self, fasta_file, tmp_path):
        out = tmp_path / "out.fa"
        main(["shuffle", "-i", str(fasta_file), "-o", str(out), "-n", "2",
              "--seed", "1", "--name-template", "{id}|bg{n}|k{k}"])
        ids = [r.id for r in read_sequences(out)]
        assert ids[:2] == ["seq1|bg1|k2", "seq1|bg2|k2"]

    def test_klets_are_preserved_end_to_end(self, fasta_file, tmp_path):
        out = tmp_path / "out.fa"
        main(["shuffle", "-i", str(fasta_file), "-o", str(out), "--seed", "2",
              "--check"])
        originals = {r.id: r.sequence for r in read_sequences(fasta_file)}
        for record in read_sequences(out):
            source = originals[record.id.rsplit("_shuf", 1)[0]]
            assert klets_preserved(source, record.sequence, 2)

    def test_deterministic_with_a_seed(self, fasta_file, tmp_path):
        first, second = tmp_path / "a.fa", tmp_path / "b.fa"
        for target in (first, second):
            main(["shuffle", "-i", str(fasta_file), "-o", str(target),
                  "--seed", "7", "-n", "2"])
        assert first.read_text() == second.read_text()

    def test_threads_do_not_change_output(self, fasta_file, tmp_path):
        first, second = tmp_path / "a.fa", tmp_path / "b.fa"
        main(["shuffle", "-i", str(fasta_file), "-o", str(first), "--seed", "7"])
        main(["shuffle", "-i", str(fasta_file), "-o", str(second), "--seed", "7",
              "--threads", "4"])
        assert first.read_text() == second.read_text()

    def test_fastq_round_trip(self, fastq_file, tmp_path):
        out = tmp_path / "out.fastq"
        main(["shuffle", "-i", str(fastq_file), "-o", str(out),
              "--out-format", "fastq", "--seed", "1"])
        records = list(read_sequences(out, "fastq"))
        assert len(records) == 2
        assert all(len(r.quality) == len(r.sequence) for r in records)

    def test_preserve_n(self, tmp_path):
        source = tmp_path / "in.fa"
        source.write_text(">a\nACGTACGTNNNNACGTACGTAAGG\n")
        out = tmp_path / "out.fa"
        main(["shuffle", "-i", str(source), "-o", str(out), "--preserve-n",
              "--seed", "1"])
        assert list(read_sequences(out))[0].sequence[8:12] == "NNNN"

    def test_window_mode(self, fasta_file, tmp_path):
        out = tmp_path / "out.fa"
        assert main(["shuffle", "-i", str(fasta_file), "-o", str(out),
                     "--window", "8", "--seed", "1"]) == 0
        assert len(list(read_sequences(out))) == 3

    def test_codon_mode(self, tmp_path):
        source = tmp_path / "cds.fa"
        source.write_text(">cds\nATGGCTGCAGCCGCGGGTTTAAAGCTGTAA\n")
        out = tmp_path / "out.fa"
        main(["shuffle", "-i", str(source), "-o", str(out), "--synonymous",
              "--seed", "1"])
        from helioseq.seq import translate

        assert translate(list(read_sequences(out))[0].sequence) == translate(
            "ATGGCTGCAGCCGCGGGTTTAAAGCTGTAA"
        )

    def test_klets_subcommand(self, fasta_file, capsys):
        assert main(["klets", "-i", str(fasta_file), "-k", "2", "--top", "3"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "record\tklet\tcount"
        # header + 3 + 3 + 2: seq3 is a GC repeat with only two distinct
        # dinucleotides, so --top 3 cannot produce three rows for it
        assert len(lines) == 9
        assert [line.split("\t")[0] for line in lines[1:]].count("seq3") == 2

    def test_klets_aggregate(self, fasta_file, capsys):
        main(["klets", "-i", str(fasta_file), "--aggregate", "--top", "2"])
        lines = capsys.readouterr().out.splitlines()
        assert all(line.startswith("*") for line in lines[1:])

    def test_check_flags_degenerate_records(self, fasta_file, capsys):
        assert main(["check", "-i", str(fasta_file), "-n", "20", "--seed", "1"]) == 0
        output = capsys.readouterr().out
        assert "seq3" in output  # GC repeat, cannot be shuffled

    def test_check_strict_exits_non_zero(self, fasta_file):
        assert main(["check", "-i", str(fasta_file), "-n", "20", "--seed", "1",
                     "--strict", "-q"]) == 1

    def test_null_subcommand(self, fasta_file, capsys):
        assert main(["null", "-i", str(fasta_file), "-n", "49", "--seed", "1",
                     "-k", "1", "--statistic", "homopolymer"]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("record\tobserved")
        assert len(lines) == 4

    def test_null_refuses_to_pretend_a_constant_null_is_a_test(self, fasta_file,
                                                               capsys):
        """A k=2 shuffle preserves CpG counts exactly, so testing them against
        it is vacuous. It must be an error, not a tidy table of p = 1."""
        assert main(["null", "-i", str(fasta_file), "-n", "49", "--seed", "1",
                     "-k", "2", "--statistic", "cpg"]) == 3
        assert "zero variance" in capsys.readouterr().err

    def test_null_rejects_an_unknown_statistic(self, fasta_file):
        with pytest.raises(SystemExit, match="unknown statistic"):
            main(["null", "-i", str(fasta_file), "--statistic", "wat"])

    def test_info(self, capsys):
        assert main(["info", "--length", "1000000", "-k", "2"]) == 0
        output = capsys.readouterr().out
        assert "backend" in output and "MB" in output

    def test_missing_file_is_reported_not_traced(self, tmp_path):
        assert main(["klets", "-i", str(tmp_path / "nope.fa")]) == 2


class TestDispatcher:
    """The registry that makes adding a subcommand a one-line change."""

    def test_every_registered_command_is_complete(self):
        import importlib

        for name, module_name in COMMANDS.items():
            module = importlib.import_module("helioseq.cli." + module_name)
            assert module.NAME == name
            assert isinstance(module.HELP, str) and module.HELP
            assert callable(module.add_arguments)
            assert callable(module.run)

    def test_help_lists_every_command(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        output = capsys.readouterr().out
        for name in COMMANDS:
            assert name in output

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            main(["--version"])
        assert "helioseq" in capsys.readouterr().out

    def test_no_command_is_an_error(self):
        with pytest.raises(SystemExit):
            main([])

    def test_command_modules_do_not_import_domains_eagerly(self):
        """argparse forces every command module to load on every invocation,
        so they must not drag in the extension or NumPy at import time. See
        docs/architecture.md."""
        import subprocess
        import sys

        script = (
            "import sys\n"
            "import helioseq.cli\n"
            "helioseq.cli.build_parser()\n"
            "leaked = [m for m in ('numpy', 'helioseq.shuffle._core',\n"
            "                      'helioseq.shuffle._purepy')\n"
            "          if m in sys.modules]\n"
            "sys.exit(1 if leaked else 0)\n"
        )
        result = subprocess.run([sys.executable, "-c", script], check=False)
        assert result.returncode == 0

    def test_shared_flags_mean_the_same_thing_everywhere(self):
        """-i and -k are defined once in cli/_args.py; check they really are
        shared rather than redefined per command."""
        from helioseq.cli import build_parser

        parser = build_parser()
        actions = {
            action.dest
            for action in parser._subparsers._group_actions[0]
            .choices["shuffle"]
            ._actions
        }
        assert {"input", "output", "k", "seed", "quiet"} <= actions

