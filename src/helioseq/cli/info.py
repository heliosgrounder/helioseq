"""``helioseq info`` -- versions, active backend, memory estimates."""

from __future__ import annotations

import sys

from ._args import add_k

NAME = "info"
HELP = "versions, active backend and memory estimates"

__all__ = ["NAME", "HELP", "add_arguments", "run"]


def add_arguments(parser) -> None:
    parser.add_argument(
        "--length", type=int, help="estimate peak memory for this sequence length"
    )
    add_k(parser)


def run(args) -> int:
    from .. import __version__
    from ..shuffle import backend, core_version, memory_estimate

    print("helioseq        %s" % __version__)
    print("libushuffle     %s" % core_version())
    print("backend         %s" % backend())
    print("python          %s" % sys.version.split()[0])

    if backend() == "python":
        print(
            "\nNOTE: running the pure-Python fallback, roughly 100x slower than\n"
            "the compiled extension. Install a wheel for this platform to fix it."
        )
    if args.length:
        print(
            "\nestimated peak memory for %d residues at k=%d: %.1f MB"
            % (args.length, args.k, memory_estimate(args.length, args.k) / 1048576.0)
        )
    return 0
