"""``helioseq check`` -- find sequences that cannot be meaningfully shuffled.

Some sequences admit only one arrangement with their k-let counts, so
"shuffling" returns the input every time and the background set is a copy of
the foreground. Nothing about the output looks wrong, which is what makes this
worth a dedicated command rather than a footnote.
"""

from __future__ import annotations

import sys

from ._args import add_input, add_k, add_quiet, add_seed

NAME = "check"
HELP = "report sequences that cannot be meaningfully shuffled"

__all__ = ["NAME", "HELP", "add_arguments", "run"]


def add_arguments(parser) -> None:
    add_input(parser)
    add_k(parser)
    parser.add_argument("-n", type=int, default=100, help="shuffles to sample")
    add_seed(parser)
    parser.add_argument(
        "--strict", action="store_true", help="exit non-zero if anything is flagged"
    )
    add_quiet(parser)


def run(args) -> int:
    from ..seqio import read_sequences
    from ..shuffle import diagnose

    print("record\tlength\tdistinct\tidentical\tmemory_mb\twarning")
    flagged = 0
    for record in read_sequences(args.input, args.format):
        report = diagnose(record.sequence, args.k, args.n, seed=args.seed)
        warnings = report.warnings()
        if warnings:
            flagged += 1
        print(
            "%s\t%d\t%d\t%d\t%.2f\t%s"
            % (
                record.id,
                report.length,
                report.distinct,
                report.identical_to_input,
                report.estimated_memory / 1048576.0,
                warnings[0] if warnings else "",
            )
        )
    if flagged and not args.quiet:
        print(
            "%d record(s) shuffle poorly -- see the warning column" % flagged,
            file=sys.stderr,
        )
    return 1 if (flagged and args.strict) else 0
