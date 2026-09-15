"""``helioseq shuffle`` -- shuffle every record in a FASTA/FASTQ file."""

from __future__ import annotations

import sys
from typing import Callable, Iterator, List

from ._args import add_input, add_k, add_output, add_quiet, add_seed, positive_int

NAME = "shuffle"
HELP = "shuffle every record in a FASTA/FASTQ file"

__all__ = ["NAME", "HELP", "add_arguments", "run"]

_DEFAULT_TEMPLATE = "{id}_shuf{n}"


def add_arguments(parser) -> None:
    add_input(parser)
    add_output(parser)
    parser.add_argument(
        "--out-format", choices=("fasta", "fastq"), help="output format"
    )
    add_k(parser)
    parser.add_argument(
        "-n", type=int, default=1, help="shuffles per record (default 1)"
    )
    add_seed(parser)
    parser.add_argument(
        "--threads", type=positive_int, default=1, help="worker threads (default 1)"
    )
    parser.add_argument(
        "--chunk",
        type=positive_int,
        default=256,
        help="records held in memory per batch",
    )
    parser.add_argument(
        "--name-template",
        default=_DEFAULT_TEMPLATE,
        help="output record id; placeholders {id} {desc} {n} {k} {seed}",
    )
    parser.add_argument(
        "--wrap", type=int, default=60, help="FASTA line width, 0 for one line"
    )

    masking = parser.add_argument_group("masking")
    masking.add_argument(
        "--keep", default="", help="characters frozen in place, e.g. N for gaps"
    )
    masking.add_argument(
        "--preserve-n", action="store_true", help="shorthand for --keep N"
    )
    masking.add_argument(
        "--case",
        choices=("preserve", "split", "ignore"),
        default="ignore",
        help="soft masking: keep the pattern, shuffle within case runs, or drop it",
    )

    alternative = parser.add_argument_group("alternative null models")
    alternative.add_argument(
        "--window", type=positive_int, help="shuffle within windows of this size"
    )
    alternative.add_argument(
        "--codon", action="store_true", help="shuffle whole codons (CDS input)"
    )
    alternative.add_argument(
        "--codon-k",
        type=positive_int,
        default=1,
        help="k over codons: 1 keeps codon usage, 2 keeps codon pairs",
    )
    alternative.add_argument(
        "--synonymous",
        action="store_true",
        help="permute codons within synonymous families, preserving the protein",
    )
    alternative.add_argument(
        "--table", type=int, default=1, help="NCBI translation table (default 1)"
    )

    parser.add_argument(
        "--check", action="store_true", help="verify k-let counts on every record"
    )
    add_quiet(parser)


def _chunks(records: Iterator, size: int) -> Iterator[List]:
    batch: List = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _plain_path(args) -> bool:
    """True when the fast batched path (a plain global shuffle) applies."""
    return not (
        args.codon or args.synonymous or args.window or args.keep
    ) and args.case == "ignore"


def _make_shuffle_fn(args) -> Callable[[str, int], str]:
    """Build the per-sequence callable the options ask for."""
    from ..shuffle import Shuffler, shuffle_masked, shuffle_windows
    from ..shuffle.codon import shuffle_codons, shuffle_synonymous

    if args.synonymous:
        return lambda seq, seed: shuffle_synonymous(seq, table=args.table, seed=seed)
    if args.codon:
        return lambda seq, seed: shuffle_codons(seq, args.codon_k, seed=seed)
    if args.window:
        return lambda seq, seed: shuffle_windows(
            seq, args.k, window=args.window, seed=seed
        )
    if args.keep or args.case != "ignore":
        return lambda seq, seed: shuffle_masked(
            seq, args.k, keep=args.keep, case=args.case, seed=seed
        )
    return lambda seq, seed: Shuffler(seq, args.k, seed=seed).shuffle()


def _format_name(template: str, record, index: int, args) -> str:
    try:
        return template.format(
            id=record.id,
            desc=record.description,
            n=index,
            k=args.k,
            seed="auto" if args.seed is None else args.seed,
        )
    except KeyError as exc:
        raise SystemExit(
            "unknown placeholder %s in --name-template; available: {id} {desc} "
            "{n} {k} {seed}" % exc
        ) from None


def run(args) -> int:
    from ..seqio import open_maybe_compressed, read_sequences, write_record
    from ..shuffle import klets_preserved, shuffle_batch

    if args.preserve_n and "N" not in args.keep:
        args.keep += "N"

    out_format = args.out_format or args.format or "fasta"
    handle = open_maybe_compressed(args.output, "wt")
    written = 0
    problems = 0
    saw_lowercase = False

    def emit(record) -> None:
        write_record(handle, record, out_format, wrap=args.wrap)

    try:
        records = read_sequences(args.input, args.format)
        if _plain_path(args):
            for chunk_index, batch in enumerate(_chunks(records, args.chunk)):
                texts = [record.sequence for record in batch]
                saw_lowercase = saw_lowercase or any(t != t.upper() for t in texts)
                # Offset the seed per chunk, otherwise records at the same
                # position in different chunks would share a random stream.
                chunk_seed = None if args.seed is None else args.seed + chunk_index
                shuffled = shuffle_batch(
                    texts, args.k, n=args.n, seed=chunk_seed, threads=args.threads
                )
                for record, outputs in zip(batch, shuffled):
                    for index, sequence in enumerate(outputs, start=1):
                        if args.check and not klets_preserved(
                            record.sequence, sequence, args.k
                        ):
                            problems += 1
                        name = _format_name(args.name_template, record, index, args)
                        emit(record.replaced(sequence, id=name))
                        written += 1
        else:
            shuffle_one = _make_shuffle_fn(args)
            for record_index, record in enumerate(records):
                for index in range(1, args.n + 1):
                    seed = (
                        None
                        if args.seed is None
                        else args.seed + record_index * args.n + index
                    )
                    sequence = shuffle_one(record.sequence, seed)
                    if args.check and not klets_preserved(
                        record.sequence, sequence, args.k
                    ):
                        problems += 1
                    name = _format_name(args.name_template, record, index, args)
                    emit(record.replaced(sequence, id=name))
                    written += 1
    finally:
        if handle is not sys.stdout:
            handle.close()

    if not args.quiet:
        print("wrote %d records" % written, file=sys.stderr)
        if saw_lowercase:
            print(
                "note: the input is soft-masked (mixed case) and --case was not "
                "given, so upper- and lowercase residues were treated as "
                "different letters. Use --case preserve to shuffle the sequence "
                "and keep the masking pattern, or --case split to shuffle "
                "inside masked and unmasked stretches separately.",
                file=sys.stderr,
            )
    if problems:
        print(
            "ERROR: %d records did not preserve their %d-let counts; please "
            "report this" % (problems, args.k),
            file=sys.stderr,
        )
        return 1
    return 0
