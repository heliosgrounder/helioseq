"""Alphabets and input validation.

The shuffling core treats a sequence as opaque bytes, which is what makes it
work for DNA, RNA and protein alike -- but it also means a typo, an HTML
fragment or a stray quality line shuffles just as happily as a real sequence.
These helpers let callers say what they expect and get a clear error instead of
plausible-looking nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .types import SeqLike, as_bytes

__all__ = [
    "Alphabet",
    "DNA",
    "RNA",
    "PROTEIN",
    "ALPHABETS",
    "get",
    "detect",
    "validate",
]


@dataclass(frozen=True)
class Alphabet:
    """A named set of residues, split by how certain each one is."""

    name: str
    core: str
    """Unambiguous residues."""
    ambiguous: str = ""
    """IUPAC ambiguity codes."""
    extra: str = ""
    """Other accepted symbols (stop codons, rare residues)."""
    gap: str = "-."
    wildcard: str = "N"
    """The 'unknown residue' symbol -- what ``--preserve-n`` acts on."""

    @property
    def letters(self) -> str:
        return self.core + self.ambiguous + self.extra

    def members(self, *, ambiguous: bool = True, gaps: bool = False) -> frozenset:
        """The accepted characters, upper and lower case, as a set of ints."""
        chars = self.core + self.extra
        if ambiguous:
            chars += self.ambiguous
        if gaps:
            chars += self.gap
        return frozenset(bytearray(chars.upper() + chars.lower(), "ascii"))


DNA = Alphabet(
    name="dna",
    core="ACGT",
    ambiguous="RYSWKMBDHVN",
    gap="-.",
    wildcard="N",
)

RNA = Alphabet(
    name="rna",
    core="ACGU",
    ambiguous="RYSWKMBDHVN",
    gap="-.",
    wildcard="N",
)

PROTEIN = Alphabet(
    name="protein",
    core="ACDEFGHIKLMNPQRSTVWY",
    ambiguous="BZJX",
    extra="UO*",  # selenocysteine, pyrrolysine, stop
    gap="-.",
    wildcard="X",
)

ALPHABETS: Dict[str, Alphabet] = {a.name: a for a in (DNA, RNA, PROTEIN)}


def get(name: str) -> Alphabet:
    """Look an alphabet up by name (``"dna"``, ``"rna"``, ``"protein"``)."""
    try:
        return ALPHABETS[name.strip().lower()]
    except KeyError:
        raise ValueError(
            "unknown alphabet %r; choose one of %s"
            % (name, ", ".join(sorted(ALPHABETS)))
        ) from None


def detect(sequence: SeqLike, *, threshold: float = 0.9) -> Optional[Alphabet]:
    """Guess the alphabet, or return ``None`` when nothing fits.

    A sequence matches when at least ``threshold`` of its residues are core
    letters of that alphabet. DNA is preferred over RNA on a tie, and an
    ambiguous ACGT-only sequence is reported as DNA rather than protein even
    though those four letters are also amino acids.
    """
    raw = as_bytes(sequence).upper()
    if not raw:
        return None

    counts = {}
    for alphabet in (DNA, RNA, PROTEIN):
        core = frozenset(bytearray(alphabet.core, "ascii"))
        counts[alphabet.name] = sum(1 for b in raw if b in core) / len(raw)

    if counts["dna"] >= threshold:
        return DNA
    if counts["rna"] >= threshold:
        return RNA
    if counts["protein"] >= threshold:
        return PROTEIN
    return None


def validate(
    sequence: SeqLike,
    alphabet: Optional[Alphabet] = None,
    *,
    ambiguous: bool = True,
    gaps: bool = False,
    max_report: int = 5,
) -> Alphabet:
    """Check that every residue belongs to ``alphabet``; return the alphabet used.

    With ``alphabet=None`` the alphabet is detected first. Raises ``ValueError``
    naming the offending characters and where they are, because "unexpected
    character 'Q' at position 4312" saves far more time than a silently odd
    background set.
    """
    raw = as_bytes(sequence)
    if alphabet is None:
        alphabet = detect(raw)
        if alphabet is None:
            raise ValueError(
                "could not detect an alphabet for this sequence; pass "
                "alphabet=helioseq.seq.DNA (or RNA/PROTEIN) explicitly"
            )

    allowed = alphabet.members(ambiguous=ambiguous, gaps=gaps)
    offenders = []
    for position, byte in enumerate(raw):
        if byte not in allowed:
            offenders.append((position, chr(byte)))
            if len(offenders) >= max_report:
                break

    if offenders:
        detail = ", ".join("%r at %d" % (char, pos) for pos, char in offenders)
        more = "" if len(offenders) < max_report else " (and possibly more)"
        raise ValueError(
            "sequence contains characters outside the %s alphabet: %s%s"
            % (alphabet.name, detail, more)
        )
    return alphabet
