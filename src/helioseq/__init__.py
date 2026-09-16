"""helioseq -- a toolkit for working with biological sequences in Python.

The package is organised by domain. Each subpackage owns one area and depends
only on the ones below it, so a new area can be added without touching the
others:

``helioseq.seq``
    Sequence primitives: type handling, alphabets, the genetic code,
    reverse complement. Depends on nothing else here.
``helioseq.seqio``
    Streaming FASTA/FASTQ reading and writing, with transparent compression.
``helioseq.stats``
    Empirical null distributions and the statistics to test against them.
``helioseq.motifs``
    Position weight matrices and motif scoring.
``helioseq.shuffle``
    k-let preserving sequence shuffling -- the uShuffle algorithm, plus the
    null models built on it (masking-aware, windowed, codon-aware) and helpers
    for machine-learning pipelines.
``helioseq.cli``
    The ``helioseq`` command. One module per subcommand.
``helioseq.compat``
    Shims for code written against other libraries' APIs.

``docs/architecture.md`` describes what a new subpackage has to provide.

Getting started
---------------

    >>> from helioseq.shuffle import shuffle
    >>> background = shuffle("ACGTACGTAGCTAGCT", k=2, seed=42)

Importing ``helioseq`` itself is deliberately cheap: subpackages load on first
use, so a script that only needs FASTA parsing never pays for the shuffling
extension, and NumPy is imported only by the code that needs it.

The shuffling algorithm is from Jiang, Anderson, Gillespie & Mayne, *uShuffle:
a useful tool for shuffling biological sequences while preserving the k-let
counts*, BMC Bioinformatics 2008, 9:192. Please cite it if you use
``helioseq.shuffle``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.5"

# Subpackages, resolved lazily by __getattr__ below.
#
# Nothing else may be re-exported here under one of these names: binding
# `helioseq.shuffle` to a function would shadow the subpackage and break
# `import helioseq.shuffle.klets`. tests/test_package.py enforces that.
_SUBPACKAGES = frozenset(
    {"seq", "seqio", "shuffle", "stats", "motifs", "cli", "compat"}
)

__all__ = ["__version__", *sorted(_SUBPACKAGES)]

if TYPE_CHECKING:  # make the subpackages visible to type checkers and IDEs
    from . import cli, compat, motifs, seq, seqio, shuffle, stats


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(__all__)
