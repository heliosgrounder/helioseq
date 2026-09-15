"""Shared fixtures.

Two things worth noting about how this suite is set up.

* It runs against whichever backend is active. CI runs it twice -- once
  normally and once with ``HELIOSEQ_BACKEND=python`` -- so both
  implementations are covered by the same assertions.
* ``golden.json`` is produced by the C harness in ``tests/c``. Checking the
  pure-Python port against it is what keeps the two implementations from
  drifting apart, which matters because they are supposed to be
  byte-for-byte identical for a given seed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def golden():
    with open(DATA / "golden.json") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def dna_sequences():
    """A reproducible spread of shapes: plain, repetitive, skewed, long."""
    rng = random.Random(20260915)
    sequences = [
        "ACGTACGTAGCTAGCTAAGGCCTT",
        "AACAGATAACAG",
        "AAAACCCCGGGGTTTT",
        "ACGT",
        "A",
        "",
        "GCGCGCGCGCGCGCGCGC",
        "".join(rng.choice("ACGT") for _ in range(500)),
        "".join(rng.choice("AACGT") for _ in range(1000)),
        "".join(rng.choice("AT") for _ in range(300)),
    ]
    return sequences


@pytest.fixture(scope="session")
def protein_sequence():
    return "MKVLAAGIVGLNLGGKVAAMKVLAAGIVGLNLGGKVAAWWYYCCHHQQ"


@pytest.fixture
def fasta_file(tmp_path):
    path = tmp_path / "input.fa"
    path.write_text(
        ">seq1 first record\n"
        "ACGTACGTAGCTAGCT\n"
        "AAGGCCTTACGTACGT\n"
        ">seq2 second record\n"
        "AACAGATAACAGATAACAGAT\n"
        ">seq3\n"
        "GCGCGCGCGCGC\n"
    )
    return path


@pytest.fixture
def fastq_file(tmp_path):
    path = tmp_path / "input.fastq"
    path.write_text(
        "@read1 desc\n"
        "ACGTACGTAGCTAGCT\n"
        "+\n"
        "IIIIIIIIIIIIIIII\n"
        "@read2\n"
        "AACAGATAACAGATAA\n"
        "+\n"
        "JJJJJJJJJJJJJJJJ\n"
    )
    return path
