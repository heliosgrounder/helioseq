"""Tests for the package structure itself.

These are the rules from ``docs/architecture.md``, made executable. They are
cheap, and they are the ones most likely to be broken by accident when a new
domain is added -- which is the whole point of writing them down.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import helioseq

SOURCE_ROOT = Path(helioseq.__file__).parent

# The layer each domain sits in. A module may import from a strictly lower
# layer, never from a higher one, and never from a sibling in the same layer.
LAYERS = {
    "seq": 0,
    "seqio": 1,
    "stats": 2,
    "motifs": 2,
    "shuffle": 2,
    "compat": 3,
    "cli": 4,
}

# Documented exception: a null distribution needs a shuffler. The import is
# inside the function, so the import graph stays acyclic; see the comment in
# helioseq/stats/nulls.py and the note in docs/architecture.md.
ALLOWED_SIBLING_IMPORTS = {("stats", "shuffle")}


def _subpackages():
    return sorted(helioseq._SUBPACKAGES)


def _modules_of(domain: str):
    package = importlib.import_module("helioseq." + domain)
    found = [package.__name__]
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        found.append(info.name)
    return found


class TestTopLevel:
    def test_every_subpackage_imports(self):
        for name in _subpackages():
            assert importlib.import_module("helioseq." + name) is not None

    def test_subpackages_are_reachable_as_attributes(self):
        for name in _subpackages():
            assert getattr(helioseq, name) is importlib.import_module(
                "helioseq." + name
            )

    def test_nothing_shadows_a_subpackage(self):
        """Binding `helioseq.shuffle` to a function would break
        `import helioseq.shuffle.klets`. Nothing at top level may share a name
        with a subpackage."""
        exported = set(helioseq.__all__) - {"__version__"}
        assert exported == helioseq._SUBPACKAGES

    def test_listed_subpackages_all_exist_on_disk(self):
        for name in _subpackages():
            assert (SOURCE_ROOT / name).is_dir(), name

    def test_every_subpackage_on_disk_is_listed(self):
        """A new domain that nobody registered is invisible to users."""
        on_disk = {
            path.name
            for path in SOURCE_ROOT.iterdir()
            if path.is_dir() and (path / "__init__.py").exists()
        }
        assert on_disk == helioseq._SUBPACKAGES

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="no attribute"):
            helioseq.nonexistent_domain

    def test_version_is_a_string(self):
        assert isinstance(helioseq.__version__, str)
        assert helioseq.__version__.count(".") >= 1


class TestLayering:
    """Enforce the dependency direction by reading the import statements."""

    @staticmethod
    def _top_level_imports(path: Path):
        """Imports of helioseq domains at module level (not inside a function)."""
        tree = ast.parse(path.read_text())
        found = set()
        for node in tree.body:  # module level only -- deliberate
            if isinstance(node, ast.ImportFrom) and node.module:
                found.add((node.module, node.level))
            elif isinstance(node, ast.ImportFrom):
                found.add(("", node.level))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("helioseq"):
                        found.add((alias.name, 0))
        return found

    @staticmethod
    def _resolve(module: str, level: int, own_domain: str):
        """Which helioseq domain does this import refer to, if any?"""
        if level >= 2:  # `from ..x import y` -- leaves the domain
            return (module or "").split(".")[0] or None
        if level == 1:  # same domain
            return own_domain
        if module.startswith("helioseq."):
            return module.split(".")[1]
        return None

    @pytest.mark.parametrize("domain", sorted(LAYERS))
    def test_no_module_imports_a_higher_layer(self, domain):
        for path in (SOURCE_ROOT / domain).rglob("*.py"):
            for module, level in self._top_level_imports(path):
                target = self._resolve(module, level, domain)
                if target is None or target == domain or target not in LAYERS:
                    continue
                pair = (domain, target)
                if pair in ALLOWED_SIBLING_IMPORTS:
                    pytest.fail(
                        "%s imports %s at module level; the documented "
                        "exception requires it to be inside the function"
                        % (path.name, target)
                    )
                assert LAYERS[target] < LAYERS[domain], (
                    "%s/%s imports helioseq.%s, which is not a lower layer "
                    "(see docs/architecture.md)" % (domain, path.name, target)
                )


class TestConventions:
    @pytest.mark.parametrize("domain", sorted(LAYERS))
    def test_every_module_declares_all(self, domain):
        for name in _modules_of(domain):
            module = importlib.import_module(name)
            if name.rsplit(".", 1)[-1].startswith("_") and not name.endswith(
                "__init__"
            ):
                continue  # private modules may skip it
            assert hasattr(module, "__all__"), "%s has no __all__" % name

    @pytest.mark.parametrize("domain", sorted(LAYERS))
    def test_everything_in_all_actually_exists(self, domain):
        for name in _modules_of(domain):
            module = importlib.import_module(name)
            for exported in getattr(module, "__all__", ()):
                assert hasattr(module, exported), "%s.%s is missing" % (name, exported)

    @pytest.mark.parametrize("domain", sorted(LAYERS))
    def test_every_module_has_a_docstring(self, domain):
        for name in _modules_of(domain):
            module = importlib.import_module(name)
            assert module.__doc__, "%s has no docstring" % name


def test_numpy_is_not_imported_by_the_core_domains():
    """NumPy belongs to helioseq.shuffle.ml alone. Importing the rest of the
    toolkit must not drag it in -- it is an optional dependency."""
    import subprocess
    import sys

    script = (
        "import sys\n"
        "import helioseq.seq, helioseq.seqio, helioseq.stats, helioseq.motifs\n"
        "import helioseq.shuffle, helioseq.cli\n"
        "sys.exit(1 if 'numpy' in sys.modules else 0)\n"
    )
    assert subprocess.run([sys.executable, "-c", script], check=False).returncode == 0
