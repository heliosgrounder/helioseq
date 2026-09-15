"""The ``helioseq`` command.

One module per subcommand, listed in :data:`COMMANDS`. Each module exposes

    NAME: str            the subcommand word
    HELP: str            the one-line description in ``helioseq --help``
    add_arguments(p)     configure that subcommand's ArgumentParser
    run(args) -> int     do the work, return an exit code

Adding a subcommand therefore means writing one module and adding one entry to
``COMMANDS``; nothing else in the package changes.

argparse needs every subparser defined before it can parse anything, so these
modules are all imported at startup. They are kept thin for exactly that
reason: a command module declares its arguments at import time and imports its
domain packages **inside** ``run()``. So ``helioseq --help``, and
``helioseq klets``, never load the shuffling extension or NumPy. Startup time
is a tax on every invocation, including the ones inside a pipeline loop, and it
only grows as the toolkit does. ``tests/test_cli.py`` checks this holds.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional, Sequence

from .. import __version__

__all__ = ["COMMANDS", "main", "build_parser"]

# Subcommand word -> module in this package. Order is the order in --help.
COMMANDS = {
    "shuffle": "shuffle",
    "check": "check",
    "klets": "klets",
    "null": "null",
    "info": "info",
}


def _load(name: str):
    return importlib.import_module("." + COMMANDS[name], __name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helioseq",
        description="A toolkit for working with biological sequences.",
        epilog="Run 'helioseq <command> --help' for a command's options.",
    )
    parser.add_argument(
        "--version", action="version", version="helioseq " + __version__
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    for name in COMMANDS:
        module = _load(name)
        subparser = subparsers.add_parser(name, help=module.HELP, description=module.HELP)
        module.add_arguments(subparser)
        subparser.set_defaults(_run=module.run)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args._run(args)
    except BrokenPipeError:  # `... | head` should be silent
        return 0
    except KeyboardInterrupt:
        print("helioseq: interrupted", file=sys.stderr)
        return 130
    except (ValueError, OSError) as exc:
        print("helioseq: %s" % exc, file=sys.stderr)
        return 2
