"""Argument fragments shared by several subcommands.

Keeping these in one place is what makes ``-i``, ``--format`` and ``-k`` mean
the same thing everywhere, which is most of what makes a multi-command tool
feel like one tool.

Nothing here imports a domain package: see the note in :mod:`helioseq.cli`.
"""

from __future__ import annotations

import argparse

__all__ = ["add_input", "add_output", "add_k", "add_seed", "add_quiet", "positive_int"]


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1, got %d" % value)
    return value


def add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i", "--input", default="-", help="input file, or '-' for stdin"
    )
    parser.add_argument(
        "--format",
        choices=("fasta", "fastq"),
        help="input format (detected from the file when not given)",
    )


def add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--output", default="-", help="output file, or '-' for stdout"
    )


def add_k(parser: argparse.ArgumentParser, default: int = 2) -> None:
    parser.add_argument(
        "-k",
        type=positive_int,
        default=default,
        help="k-let size (default %d)" % default,
    )


def add_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seed", type=int, help="seed, for output that is reproducible everywhere"
    )


def add_quiet(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress progress notes on stderr"
    )
