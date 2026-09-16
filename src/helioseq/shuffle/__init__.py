"""k-let preserving sequence shuffling, and the null models built on it.

A shuffle that keeps every k-mer count identical to the input is the standard
null model for sequence analysis: it controls for composition (GC, CpG,
dinucleotide bias) while destroying everything else, so whatever signal
survives is not an artefact of composition.

    >>> from helioseq.shuffle import shuffle, klet_counts
    >>> background = shuffle("ACGTACGTAGCTAGCT", k=2, seed=42)
    >>> klet_counts(background) == klet_counts("ACGTACGTAGCTAGCT")
    True

The algorithm -- a uniformly random Eulerian walk, sampled via Wilson's
algorithm -- is from Jiang, Anderson, Gillespie & Mayne, *uShuffle: a useful
tool for shuffling biological sequences while preserving the k-let counts*,
BMC Bioinformatics 2008, 9:192. **Please cite it.**

What is where
-------------
``shuffler``  :func:`shuffle`, :class:`Shuffler`, :func:`shuffle_batch`
``klets``     counting, verification, degeneracy diagnostics
``masking``   keep ``N`` runs and soft masking in place
``windows``   shuffle locally, or within your own annotation
``codon``     frame-aware null models for coding sequences
``ml``        ``dinuc_shuffle`` for attribution pipelines (needs NumPy for
              one-hot input; the plain string form does not)
``ml_torch``  one-hot sequence tensors for PyTorch models/training (needs
              torch)

Neither ``ml`` nor ``ml_torch`` is imported here, so NumPy and torch are
never pulled in unless you ask for them: ``from helioseq.shuffle import ml``
or ``from helioseq.shuffle import ml_torch``.
"""

from __future__ import annotations

from ._backend import backend, core_version, memory_estimate
from .codon import shuffle_codons, shuffle_synonymous, shuffle_third_positions
from .klets import (
    Diagnosis,
    check_klets,
    diagnose,
    distinct_count,
    is_degenerate,
    klet_counts,
    klets_preserved,
)
from .masking import shuffle_masked
from .shuffler import (
    Shuffler,
    set_seed,
    shuffle,
    shuffle_batch,
    shuffle_many,
)
from .windows import shuffle_segments, shuffle_windows, tile_spans, window_report

__all__ = [
    # shuffling
    "shuffle",
    "Shuffler",
    "shuffle_many",
    "shuffle_batch",
    "set_seed",
    # diagnostics
    "klet_counts",
    "klets_preserved",
    "check_klets",
    "distinct_count",
    "is_degenerate",
    "diagnose",
    "Diagnosis",
    # alternative null models
    "shuffle_masked",
    "shuffle_windows",
    "shuffle_segments",
    "tile_spans",
    "window_report",
    "shuffle_codons",
    "shuffle_synonymous",
    "shuffle_third_positions",
    # environment
    "backend",
    "core_version",
    "memory_estimate",
]
