"""Build script for the compiled backend.

All project metadata lives in ``pyproject.toml``; this file exists only for the
two decisions the extension needs that static metadata cannot express.

**The stable ABI.** The extension is built against ``Py_LIMITED_API``, so one
``cp39-abi3`` wheel per platform runs on Python 3.9 and on every version after
it. That is the structural fix for what went wrong with uShuffle 1.x: a new
Python release can no longer break installation, because no new wheel is
needed. There is no code generation in the build at all -- nothing to
regenerate, nothing to go stale.

**A failed build is a warning, not an error.** The package still installs and
falls back to :mod:`helioseq.shuffle._purepy`, so ``pip install`` cannot fail because
of a missing compiler. Set ``HELIOSEQ_REQUIRE_EXTENSION=1`` to make it fatal;
the wheel jobs do, so a broken build can never be published as a silently slow
wheel.
"""

import os
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

REQUIRE_EXTENSION = os.environ.get("HELIOSEQ_REQUIRE_EXTENSION", "") not in ("", "0")

# Python 3.9 is the floor; the resulting wheel is forward-compatible.
LIMITED_API_VERSION = 0x03090000

is_cpython = sys.implementation.name == "cpython"

extension = Extension(
    "helioseq.shuffle._core",
    sources=[
        os.path.join("src", "helioseq", "shuffle", "_core.c"),
        os.path.join("src", "libushuffle", "ushuffle.c"),
    ],
    include_dirs=[os.path.join("src", "libushuffle")],
    define_macros=(
        [("Py_LIMITED_API", hex(LIMITED_API_VERSION))] if is_cpython else []
    ),
    # PyPy and other implementations have no stable ABI to target; they build
    # the same source without the limited-API constraint (and are usually
    # better served by the pure-Python backend anyway).
    py_limited_api=is_cpython,
    extra_compile_args=(
        [] if sys.platform == "win32" else ["-O2", "-std=c99", "-Wall", "-Wextra"]
    ),
)


class build_ext(_build_ext):
    """Tolerate a failed compile unless the caller insists otherwise."""

    _already_reported = False

    def run(self):
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 - we really do want everything
            self._handle(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001
            self._handle(exc)

    def _handle(self, exc):
        if REQUIRE_EXTENSION:
            raise exc
        # A failed compile surfaces twice: once here, and again when run()
        # tries to copy the object that was never produced. The first one names
        # the actual cause ("no such file or directory: cc"); the second would
        # only bury it under a confusing message about a missing .so.
        if self._already_reported:
            return
        type(self)._already_reported = True
        sys.stderr.write(
            "\n"
            + "*" * 72
            + "\nhelioseq: the compiled backend could not be built:\n"
            "    %s\n"
            "The package will still install and work, using the pure-Python\n"
            "fallback (roughly 100x slower). Installing a C compiler, or a\n"
            "wheel for this platform, restores the fast path.\n" % exc
            + "*" * 72
            + "\n\n"
        )


setup(
    ext_modules=[extension],
    cmdclass={"build_ext": build_ext},
    options=(
        {"bdist_wheel": {"py_limited_api": "cp39"}} if is_cpython else {}
    ),
)
