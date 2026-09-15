"""Benchmarks: is the fast path actually fast, and do threads actually help?

    python benchmarks/bench.py
    python benchmarks/bench.py --quick

Three questions, because each one corresponds to a claim in the README and a
claim that is not measured is a claim that is not true.

1. How much does preparing once and drawing many times save? This is the
   difference between ``Shuffler`` and calling ``shuffle`` in a loop, and it is
   the single biggest win available to anyone building a null distribution.
2. Do threads scale? They only can because the extension releases the GIL, so
   this measures whether that actually works.
3. How far behind is the pure-Python fallback? The README says "roughly 100x";
   this is where that number comes from.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time


def make_sequence(length: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def timed(function, repeats: int = 3) -> float:
    """Best-of-N wall time in seconds; best-of is the right summary for a
    benchmark competing with the OS scheduler."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def bench_prepare_vs_reuse(lengths, draws: int) -> None:
    from helioseq.shuffle import Shuffler, shuffle

    print("\n1. preparing once vs. re-analysing every time (%d draws)" % draws)
    print("   %-10s %12s %12s %8s" % ("length", "shuffle()", "Shuffler", "speedup"))
    for length in lengths:
        sequence = make_sequence(length)

        naive = timed(lambda: [shuffle(sequence, 2) for _ in range(draws)])
        reused = timed(lambda: Shuffler(sequence, 2).shuffle_many(draws))
        print(
            "   %-10d %11.3fs %11.3fs %7.1fx"
            % (length, naive, reused, naive / reused if reused else float("nan"))
        )


def bench_threads(length: int, count: int, draws: int) -> None:
    from helioseq.shuffle import backend, shuffle_batch

    print(
        "\n2. thread scaling (%d sequences x %d bp x %d draws, backend=%s)"
        % (count, length, draws, backend())
    )
    sequences = [make_sequence(length, seed=i) for i in range(count)]
    baseline = None
    for threads in (1, 2, 4, 8):
        if threads > (os.cpu_count() or 1):
            break
        elapsed = timed(
            lambda t=threads: shuffle_batch(
                sequences, 2, n=draws, seed=1, threads=t
            ),
            repeats=2,
        )
        baseline = baseline or elapsed
        print("   %d thread(s): %7.3fs  (%.2fx)" % (threads, elapsed, baseline / elapsed))
    if backend() == "python":
        print("   (the pure-Python backend holds the GIL, so this will not scale)")


def bench_backends(length: int, draws: int) -> None:
    print("\n3. compiled backend vs. pure-Python fallback")
    sequence = make_sequence(length)

    results = {}
    for name in ("c", "python"):
        elapsed = _time_backend(name, sequence, draws)
        if elapsed is None:
            print("   %-8s not available" % name)
            continue
        results[name] = elapsed
        print(
            "   %-8s %8.3fs for %d draws of %d bp  (%.1f us/draw)"
            % (name, elapsed, draws, length, elapsed / draws * 1e6)
        )
    if len(results) == 2:
        print("   ratio:   %.0fx" % (results["python"] / results["c"]))


def _time_backend(name: str, sequence: str, draws: int):
    """Time one backend in a subprocess, since the choice is made at import."""
    import subprocess

    script = (
        "import time, os\n"
        "from helioseq.shuffle import Shuffler, backend\n"
        "if backend() != %r: raise SystemExit(3)\n"
        "s = %r\n"
        "sh = Shuffler(s, 2, seed=1)\n"
        "start = time.perf_counter()\n"
        "sh.shuffle_many(%d)\n"
        "print(time.perf_counter() - start)\n" % (name, sequence, draws)
    )
    environment = dict(os.environ, HELIOSEQ_BACKEND=name)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return float(completed.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller workloads")
    args = parser.parse_args()

    import helioseq

    print("helioseq %s, backend=%s, python %s"
          % (helioseq.__version__, helioseq.shuffle.backend(), sys.version.split()[0]))

    if args.quick:
        bench_prepare_vs_reuse([200, 2_000], draws=200)
        bench_threads(2_000, count=16, draws=4)
        bench_backends(1_000, draws=200)
    else:
        bench_prepare_vs_reuse([200, 2_000, 20_000], draws=1_000)
        bench_threads(10_000, count=64, draws=8)
        bench_backends(5_000, draws=2_000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
