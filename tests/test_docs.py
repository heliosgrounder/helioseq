"""Every example in every docstring actually runs.

Discovered by walking the package rather than from a hand-written list, so a
new domain is covered the moment it exists. Docstring examples are the first
thing people copy; an example that silently rots is worse than no example.
"""

from __future__ import annotations

import doctest
import importlib
import importlib.util
import pkgutil

import pytest

import helioseq


def _all_modules():
    names = ["helioseq"]
    for domain in sorted(helioseq._SUBPACKAGES):
        package = importlib.import_module("helioseq." + domain)
        names.append(package.__name__)
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            names.append(info.name)
    return names


def _missing_optional_dependency(module):
    """Name of the optional dependency gating this module's examples, if any.

    Modules that need an optional package expose it via a ``_require_<name>``
    helper (see ``ml._require_numpy``, ``ml_torch._require_torch``). Doctests
    that exercise such a module raise a plain ``ImportError`` when the
    dependency isn't installed -- e.g. torch has no wheel for every Python
    version, so installs without a compiler can't build it from source. That
    ``ImportError`` surfaces as a doctest failure rather than a test error, so
    it has to be checked for explicitly before running the doctests.
    """
    for name in vars(module):
        if name.startswith("_require_"):
            dependency = name[len("_require_") :]
            if importlib.util.find_spec(dependency) is None:
                return dependency
    return None


@pytest.mark.parametrize("module_name", _all_modules())
def test_docstring_examples(module_name):
    module = importlib.import_module(module_name)
    missing = _missing_optional_dependency(module)
    if missing is not None:
        pytest.skip("%s doctests need optional dependency %r" % (module_name, missing))
    result = doctest.testmod(
        module,
        verbose=False,
        optionflags=doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL,
    )
    assert result.failed == 0, "%s: %d failing doctest(s)" % (
        module_name,
        result.failed,
    )


def test_architecture_doc_exists_and_lists_the_domains():
    """The doc is the contract for adding a domain; keep it in step."""
    from pathlib import Path

    candidates = [
        Path(helioseq.__file__).parents[2] / "docs" / "architecture.md",
        Path(helioseq.__file__).parents[3] / "docs" / "architecture.md",
    ]
    doc = next((path for path in candidates if path.exists()), None)
    if doc is None:
        pytest.skip("running from an installed package without the docs directory")

    text = doc.read_text(encoding="utf-8")
    for domain in helioseq._SUBPACKAGES:
        assert domain in text, "docs/architecture.md does not mention %r" % domain
