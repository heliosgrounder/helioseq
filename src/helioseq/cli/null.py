"""``helioseq null`` -- empirical p-value against a k-let preserving null."""

from __future__ import annotations

import sys

from ._args import add_input, add_k, add_seed, positive_int

NAME = "null"
HELP = "empirical p-value against a k-let preserving null"

__all__ = ["NAME", "HELP", "add_arguments", "run"]

STATISTICS = ("gc", "cpg", "homopolymer", "kmer:<SEQ>")


def add_arguments(parser) -> None:
    add_input(parser)
    add_k(parser)
    parser.add_argument("-n", type=int, default=1000, help="shuffles per record")
    add_seed(parser)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument(
        "--statistic",
        default="cpg",
        help="one of: %s" % ", ".join(STATISTICS),
    )
    parser.add_argument(
        "--alternative",
        choices=("greater", "less", "two-sided"),
        default="greater",
    )


def _resolve(name: str):
    from ..stats import (
        cpg_observed_expected,
        gc_content,
        kmer_frequency,
        longest_homopolymer,
    )

    if name == "gc":
        return gc_content
    if name == "cpg":
        return cpg_observed_expected
    if name == "homopolymer":
        return longest_homopolymer
    if name.startswith("kmer:"):
        return kmer_frequency(name.split(":", 1)[1])
    raise SystemExit(
        "unknown statistic %r; use one of: %s" % (name, ", ".join(STATISTICS))
    )


def run(args) -> int:
    from ..seqio import read_sequences
    from ..stats import null_test

    statistic = _resolve(args.statistic)

    print("record\tobserved\tnull_mean\tnull_sd\tzscore\tpvalue\tat_limit")
    constant = 0
    total = 0
    for record in read_sequences(args.input, args.format):
        result = null_test(
            record.sequence,
            statistic,
            k=args.k,
            n=args.n,
            seed=args.seed,
            alternative=args.alternative,
            threads=args.threads,
        )
        total += 1
        constant += result.constant_null
        print(
            "%s\t%.6g\t%.6g\t%.6g\t%.3f\t%.4g\t%s"
            % (
                record.id,
                result.observed,
                result.null_mean,
                result.null_sd,
                result.zscore,
                result.pvalue,
                "yes" if result.at_resolution_limit else "no",
            )
        )

    # A statistic the null model preserves by construction produces a neat
    # table of p = 1 rather than an error, so say so loudly. GC content and CpG
    # counts are both invariant under k >= 2.
    if total and constant == total:
        print(
            "\nWARNING: the null distribution has zero variance for every "
            "record, so a %d-let shuffle cannot test '%s' -- this statistic is "
            "preserved by construction. Try a smaller k (e.g. -k 1), or a "
            "statistic that depends on the arrangement of the sequence rather "
            "than its composition." % (args.k, args.statistic),
            file=sys.stderr,
        )
        return 3
    return 0
