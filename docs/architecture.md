# Architecture

helioseq is organised by domain. This document is the contract a new domain has
to satisfy, so that adding one is a local change rather than a refactor.

## The layers

```
        cli/            one module per subcommand
         │
   ┌─────┼──────┬──────────┐
   │     │      │          │
shuffle/ │  motifs/     (your new domain)
   │     │      │          │
   └──── stats/ ┘          │
         │                 │
       seqio/ ─────────────┘
         │
        seq/              primitives: types, alphabets, genetic code
```

The rule is one-directional: **a module may import from the layers below it and
never from the layers above.** `seq` imports nothing else from helioseq;
`seqio` may use `seq`; `stats`, `motifs` and `shuffle` may use both; `cli` may
use everything. Sibling domains do not import each other.

There is one deliberate exception, and it is worth knowing about.
`helioseq.stats.nulls` needs a default shuffler, so it imports
`helioseq.shuffle` — but **inside the function**, not at module level. The
dependency is real (a null distribution needs shuffles) and the lazy import
keeps the import graph acyclic and `import helioseq.stats` cheap. If a new
domain needs the same trick, do it the same way and say why in a comment.

## What a domain looks like

```
src/helioseq/<domain>/
    __init__.py       the public API: re-exports, __all__, a docstring that
                      says what the domain is for and what lives where
    <topic>.py        one module per coherent topic
    _<private>.py     helpers nobody outside the domain should import
```

Rules that matter:

- **`__init__.py` re-exports, it does not implement.** Someone reading
  `__init__.py` should be able to see the whole surface of the domain at once.
- **`__all__` is explicit** in every module. It is what `from x import *`
  honours and what tooling reads.
- **Optional dependencies are imported inside the function that needs them**,
  never at module level. NumPy is used only by `shuffle/ml.py`, and importing
  `helioseq.shuffle` must not pull it in. A domain that needs a heavy
  dependency should isolate it in its own module, not spread it across the
  domain.
- **Every public function accepts `str`, `bytes`, `bytearray` or `memoryview`
  and returns the input's type.** Use `helioseq.seq.types.coerce` — do not
  reimplement it.
- **Docstring examples are tested.** `tests/test_docs.py` runs doctest over
  every module, so an example that drifts out of date is a failing test.

## Adding a domain

1. Create `src/helioseq/<domain>/` with an `__init__.py` as above.
2. Add the name to `_SUBPACKAGES` in `src/helioseq/__init__.py`. That is what
   makes `helioseq.<domain>` resolve lazily, and it is the only place the top
   level learns about it.
3. Nothing may be re-exported at top level under a subpackage's name — binding
   `helioseq.shuffle` to a function would shadow the subpackage and break
   `import helioseq.shuffle.klets`. `tests/test_package.py` enforces this.
4. Add `tests/test_<domain>.py`. `tests/test_docs.py` and
   `tests/test_package.py` pick the new domain up automatically.

## Adding a CLI subcommand

Write `src/helioseq/cli/<name>.py` exposing four things:

```python
NAME = "thing"                 # the subcommand word
HELP = "do the thing"          # one line, shown in `helioseq --help`

def add_arguments(parser): ...  # configure the subparser
def run(args) -> int: ...       # do the work, return an exit code
```

Then add one entry to `COMMANDS` in `src/helioseq/cli/__init__.py`.

Two constraints:

- **Import domain packages inside `run()`, not at module level.** argparse needs
  every subparser defined before it can parse anything, so all command modules
  are imported on every invocation. Keeping them thin is what stops
  `helioseq klets` from loading the shuffling extension and NumPy.
  `tests/test_cli.py` checks this.
- **Reuse the fragments in `cli/_args.py`** (`-i`, `-o`, `-k`, `--seed`,
  `--quiet`) rather than redefining them, so the flags mean the same thing in
  every subcommand.

## Adding a file format

`seqio` keeps a reader and a writer per format in `_READERS` and `_WRITERS`,
with each writer declaring the keyword options it understands. Add a module,
add two entries, and `read_sequences` / `write_record` / the CLI pick it up.
Detection in `guess_format` is separate and may need a line too.

## The C core

`src/libushuffle/` is vendored third-party C (BSD-3, see
`LICENSE.original`) with a re-entrant API added on top; `helioseq/shuffle/_core.c`
is the CPython binding. Both are documented in their own headers. Two rules:

- The binding targets the **limited API** (`Py_LIMITED_API`), so one `abi3`
  wheel per platform serves every Python from 3.9 onward. Anything added to the
  binding must stay inside the limited API, or the wheels stop being
  forward-compatible.
- `helioseq/shuffle/_purepy.py` must keep reproducing the C implementation
  **byte for byte**. `tests/data/golden.json` is generated by the C harness and
  asserted against the Python port; if you change either, run `make golden` and
  `make test-all` before committing.
