"""``helioseq klets`` -- count k-lets per record or across a whole file."""

from __future__ import annotations

from ._args import add_input, add_k

NAME = "klets"
HELP = "count k-lets"

__all__ = ["NAME", "HELP", "add_arguments", "run"]


def add_arguments(parser) -> None:
    add_input(parser)
    add_k(parser)
    parser.add_argument("--top", type=int, help="only the N most frequent")
    parser.add_argument(
        "--aggregate", action="store_true", help="pool all records together"
    )


def _top(counts, limit):
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items if not limit else items[:limit]


def run(args) -> int:
    from ..seqio import read_sequences
    from ..shuffle import klet_counts

    totals: dict = {}
    per_record = not args.aggregate

    print("record\tklet\tcount")
    for record in read_sequences(args.input, args.format):
        counts = klet_counts(record.sequence, args.k)
        if per_record:
            for klet, count in _top(counts, args.top):
                print("%s\t%s\t%d" % (record.id, klet, count))
        else:
            for klet, count in counts.items():
                totals[klet] = totals.get(klet, 0) + count

    if not per_record:
        for klet, count in _top(totals, args.top):
            print("*\t%s\t%d" % (klet, count))
    return 0
