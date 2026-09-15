"""The uShuffle 1.x compatibility shim.

It exists so a script written against `ushuffle` 1.1.x keeps running after
changing one import line. What it does *not* reproduce is 1.x's bugs, so the
tests here assert the old API surface and the new correctness.
"""

from __future__ import annotations

import pytest

from helioseq.shuffle import klets_preserved

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_shuffle_keeps_the_1x_signature_and_returns_bytes():
    from helioseq.compat import ushuffle

    result = ushuffle.shuffle(b"ACGTACGTAGCTAGCT", 2)
    assert isinstance(result, bytes)
    assert klets_preserved(b"ACGTACGTAGCTAGCT", result, 2)


def test_shuffler_keeps_the_1x_attribute_names():
    from helioseq.compat import ushuffle

    shuffler = ushuffle.Shuffler(b"ACGTACGTAGCTAGCT", 2, seed=1)
    assert shuffler.let_size == 2  # 1.x spelling of k
    assert shuffler.length == 16
    assert shuffler.sequence == b"ACGTACGTAGCTAGCT"
    assert isinstance(shuffler.shuffle(), bytes)


def test_set_seed_makes_later_shuffles_reproducible():
    from helioseq.compat import ushuffle

    ushuffle.set_seed(7)
    first = [ushuffle.shuffle(b"ACGTACGTAGCTAGCTAAGG", 2) for _ in range(5)]
    ushuffle.set_seed(7)
    assert [ushuffle.shuffle(b"ACGTACGTAGCTAGCTAAGG", 2) for _ in range(5)] == first

    from helioseq.shuffle import set_seed

    set_seed(None)


def test_str_is_accepted_even_though_1x_refused_it():
    from helioseq.compat import ushuffle

    assert isinstance(ushuffle.shuffle("ACGTACGTAGCT", 2), bytes)


def test_two_live_shufflers_do_not_interfere():
    """The 1.x bug this shim deliberately does not reproduce."""
    from helioseq.compat import ushuffle

    short = ushuffle.Shuffler(b"ACGTACGTAC", 2, seed=1)
    long = ushuffle.Shuffler(b"GGGGCCCCGGGGCCCCGGGGCCCC", 2, seed=2)
    for _ in range(50):
        a, b = short.shuffle(), long.shuffle()
        assert len(a) == 10 and len(b) == 24
        assert klets_preserved(b"ACGTACGTAC", a, 2)


def test_everything_warns_about_the_migration():
    from helioseq.compat import ushuffle

    for call in (
        lambda: ushuffle.shuffle(b"ACGTACGT", 2),
        lambda: ushuffle.Shuffler(b"ACGTACGT", 2),
        lambda: ushuffle.set_seed(1),
    ):
        with pytest.warns(DeprecationWarning, match="helioseq.shuffle"):
            call()
    from helioseq.shuffle import set_seed

    set_seed(None)
