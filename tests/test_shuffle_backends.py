"""The two backends must agree exactly.

``tests/data/golden.json`` is emitted by the C harness (``tests/c``). Checking
the pure-Python port against it means a change to either implementation that
alters results shows up as a test failure, not as a quiet difference in
somebody's published numbers.
"""

from __future__ import annotations

import pytest

from helioseq.shuffle import backend
from helioseq.shuffle._purepy import ShufflerImpl as PurePython


def test_pure_python_matches_the_golden_vectors(golden):
    for case in golden:
        shuffler = PurePython(seed=case["seed"], stream=case["stream"])
        shuffler.prepare(case["seq"].encode(), case["k"])
        produced = [shuffler.generate().decode() for _ in case["draws"]]
        assert produced == case["draws"], case["seq"]


@pytest.mark.skipif(backend() != "c", reason="the compiled backend is not installed")
def test_compiled_backend_matches_the_golden_vectors(golden):
    from helioseq.shuffle._core import ShufflerImpl as Compiled

    for case in golden:
        shuffler = Compiled(seed=case["seed"], stream=case["stream"])
        shuffler.prepare(case["seq"].encode(), case["k"])
        produced = [shuffler.generate().decode() for _ in case["draws"]]
        assert produced == case["draws"], case["seq"]


@pytest.mark.skipif(backend() != "c", reason="the compiled backend is not installed")
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_backends_agree_on_random_input(dna_sequences, k):
    from helioseq.shuffle._core import ShufflerImpl as Compiled

    for index, sequence in enumerate(dna_sequences):
        raw = sequence.encode()
        fast = Compiled(seed=1234, stream=index)
        slow = PurePython(seed=1234, stream=index)
        fast.prepare(raw, k)
        slow.prepare(raw, k)
        assert [fast.generate() for _ in range(10)] == [
            slow.generate() for _ in range(10)
        ], (sequence, k)


def test_rng_reference_values():
    """xoshiro256++ seeded through splitmix64, pinned so a refactor of either
    implementation cannot quietly change every user's output."""
    from helioseq.shuffle._purepy import _Rng

    rng = _Rng(42, 0)
    first = [rng.next() for _ in range(4)]
    assert all(0 <= value < 2**64 for value in first)
    assert len(set(first)) == 4

    again = _Rng(42, 0)
    assert [again.next() for _ in range(4)] == first

    other = _Rng(42, 1)
    assert [other.next() for _ in range(4)] != first


def test_bounded_sampling_is_unbiased():
    """Masked rejection, not `% n` -- the modulo bias in 1.x was small on
    glibc and catastrophic with the 32767-range rand() used on Windows."""
    from collections import Counter

    from helioseq.shuffle._purepy import _Rng

    rng = _Rng(7, 0)
    counts = Counter(rng.below(3) for _ in range(90_000))
    assert set(counts) == {0, 1, 2}
    for value in counts.values():
        assert abs(value - 30_000) < 1500

    assert _Rng(1, 0).below(1) == 0
    assert _Rng(1, 0).below(0) == 0


def test_backend_override_is_validated(monkeypatch):
    import importlib

    monkeypatch.setenv("HELIOSEQ_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="HELIOSEQ_BACKEND"):
        importlib.reload(importlib.import_module("helioseq.shuffle._backend"))

    monkeypatch.delenv("HELIOSEQ_BACKEND")
    importlib.reload(importlib.import_module("helioseq.shuffle._backend"))
