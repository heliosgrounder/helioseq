"""Statistical tests.

"Preserves k-let counts" is necessary but not sufficient -- a sampler could
preserve them and still favour some arrangements over others, which would bias
every p-value computed against it. uShuffle's claim is that the shuffle is
drawn *uniformly* from the set of sequences with those counts, and that claim
is testable: for a short sequence the support can be enumerated exhaustively
and compared with the observed frequencies.

The bounds here are deliberately loose (6 standard deviations) so the suite
never flakes, and the sequences are chosen so the test still has real power.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from helioseq.shuffle import Shuffler, klet_counts


def enumerate_support(sequence: str, k: int) -> set:
    """Every sequence uShuffle could legitimately produce.

    That is: the same multiset of characters, the same k-let counts, and the
    same first (k-1) characters -- which the algorithm holds fixed, and which
    then forces the last (k-1) characters too.

    Built by backtracking with k-let pruning rather than by filtering
    ``itertools.permutations``: the factorial version is unusable past about
    nine residues, and the sequences worth testing are longer than that.
    """
    reference = klet_counts(sequence, k)
    length = len(sequence)
    letters = sorted(set(sequence))
    found: set = set()

    def recurse(built: list, remaining: Counter, used: Counter) -> None:
        depth = len(built)
        if depth == length:
            if used == reference:
                found.add("".join(built))
            return
        for letter in letters:
            if remaining[letter] == 0:
                continue
            if depth < k - 1 and letter != sequence[depth]:
                continue
            built.append(letter)
            klet = "".join(built[-k:]) if depth + 1 >= k else None
            if klet is not None and used[klet] + 1 > reference[klet]:
                built.pop()
                continue
            remaining[letter] -= 1
            if klet is not None:
                used[klet] += 1
            recurse(built, remaining, used)
            if klet is not None:
                used[klet] -= 1
                if used[klet] == 0:
                    del used[klet]
            remaining[letter] += 1
            built.pop()

    recurse([], Counter(sequence), Counter())
    return found


@pytest.mark.slow
@pytest.mark.parametrize(
    "sequence,k,draws",
    [
        ("AACAGATAACAG", 2, 60_000),  # 180 outcomes
        ("AACCAACCAA", 2, 20_000),  # 30 outcomes
        ("ACAGCTACAGCT", 2, 20_000),  # 18 outcomes, four letters
        ("GAGAAACGGGGACAAG", 3, 20_000),  # 24 outcomes, k > 2
    ],
)
def test_shuffles_are_uniform(sequence, k, draws):
    support = enumerate_support(sequence, k)
    assert len(support) > 5, "pick a sequence with a larger support"

    shuffler = Shuffler(sequence, k, seed=20260915)
    observed = Counter(shuffler.shuffle() for _ in range(draws))

    # Nothing outside the support may ever appear.
    assert set(observed) <= support, set(observed) - support

    expected = draws / len(support)
    chi2 = sum(
        (observed.get(candidate, 0) - expected) ** 2 / expected
        for candidate in support
    )
    df = len(support) - 1
    bound = df + 6 * math.sqrt(2 * df)
    assert chi2 < bound, "chi2=%.1f exceeds %.1f (df=%d)" % (chi2, bound, df)


@pytest.mark.slow
def test_every_reachable_sequence_is_reached():
    """Coverage, not just balance: a sampler that never emits some valid
    arrangement is biased even if the ones it does emit look uniform."""
    sequence = "AACAGATAACAG"
    support = enumerate_support(sequence, 2)
    shuffler = Shuffler(sequence, 2, seed=1)
    seen = {shuffler.shuffle() for _ in range(40_000)}
    assert seen == support


@pytest.mark.slow
def test_no_positional_bias():
    """Every residue should be about equally likely at every position, given
    the constraints. A generator with a truncated range -- the Windows rand()
    problem in uShuffle 1.x -- fails this badly."""
    sequence = "".join("ACGT"[i % 4] for i in range(40)) + "AACCGGTT" * 5
    shuffler = Shuffler(sequence, 1, seed=7)  # k=1: a plain permutation
    draws = 4000
    counts = [Counter() for _ in range(len(sequence))]
    for _ in range(draws):
        for position, residue in enumerate(shuffler.shuffle()):
            counts[position][residue] += 1

    composition = Counter(sequence)
    for position, seen in enumerate(counts):
        for residue, total in composition.items():
            expected = draws * total / len(sequence)
            sd = math.sqrt(expected * (1 - total / len(sequence)))
            deviation = abs(seen.get(residue, 0) - expected)
            assert deviation < 6 * sd, (
                "residue %r at position %d: %d vs expected %.1f"
                % (residue, position, seen.get(residue, 0), expected)
            )


@pytest.mark.slow
def test_shuffling_destroys_order_but_not_composition():
    """The point of the whole exercise, stated as a test: a real motif
    disappears from the null while composition stays put."""
    from helioseq.stats import gc_content, kmer_frequency, null_test

    motif = "GGGGCGGGGC"
    background = "ACTACTGTCAATCGTACATGCATACGTTAGCATGCAATCGATCAGTGACT"
    sequence = background + motif * 4 + background

    result = null_test(sequence, kmer_frequency(motif), k=2, n=499, seed=3)
    assert result.pvalue <= 0.01, result.summary()

    gc = null_test(sequence, gc_content, k=2, n=99, seed=3)
    assert gc.null_sd == 0.0  # a k>=2 shuffle preserves GC exactly
    assert gc.pvalue == 1.0
