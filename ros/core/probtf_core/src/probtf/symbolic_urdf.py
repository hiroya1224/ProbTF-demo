"""ROS-free parsing and materialization for legacy symbolic URDF files."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import numbers
import re
import xml.etree.ElementTree as ET


_PLACEHOLDER_PATTERN = re.compile(r"#\|(?P<name>[A-Za-z0-9_.$:{}-]+)\|#")
_XML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_UNSAFE_XML_PATTERN = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_LEGACY_NOTE_PATTERNS = (
    re.compile(r"^[ \t]*<!--\s*=+\s*SYMBOLIC URDF\s*=+\s*-->[ \t]*(?:\r?\n)?", re.MULTILINE),
    re.compile(
        r"^[ \t]*<!--\s*indefinite variables are written in "
        r"#\|this_form\|#\s*-->[ \t]*(?:\r?\n)?",
        re.MULTILINE,
    ),
)


def _validate_document(document):
    if not isinstance(document, str):
        raise TypeError("symbolic URDF must be a string")
    if not document.strip():
        raise ValueError("symbolic URDF must not be empty")
    if "\x00" in document:
        raise ValueError("symbolic URDF must not contain NUL characters")
    if _UNSAFE_XML_PATTERN.search(document):
        raise ValueError("DOCTYPE and ENTITY declarations are not supported")
    try:
        ET.fromstring(document)
    except ET.ParseError as error:
        raise ValueError("symbolic URDF is not well-formed XML: {}".format(error)) from error


def _content_outside_comments(document):
    return _XML_COMMENT_PATTERN.sub(lambda match: " " * len(match.group(0)), document)


def _parse_placeholder_names(document):
    searchable = _content_outside_comments(document)
    matches = list(_PLACEHOLDER_PATTERN.finditer(searchable))
    without_valid_markers = _PLACEHOLDER_PATTERN.sub("", searchable)
    if "#|" in without_valid_markers or "|#" in without_valid_markers:
        raise ValueError("symbolic URDF contains a malformed placeholder")

    names = []
    seen = set()
    for match in matches:
        name = match.group("name")
        if name not in seen:
            names.append(name)
            seen.add(name)
    return tuple(names)


def _format_number(value, placeholder_name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("substitution '{}' must contain only real numbers".format(placeholder_name))
    if not math.isfinite(float(value)):
        raise ValueError("substitution '{}' must contain only finite numbers".format(placeholder_name))
    return str(value)


def _format_substitution(value, placeholder_name):
    if isinstance(value, str):
        tokens = value.split()
        if not tokens:
            raise ValueError("substitution '{}' must not be empty".format(placeholder_name))
        for token in tokens:
            try:
                parsed = float(token)
            except ValueError as error:
                raise TypeError(
                    "substitution '{}' must be a whitespace-separated numeric value".format(placeholder_name)
                ) from error
            if not math.isfinite(parsed):
                raise ValueError("substitution '{}' must contain only finite numbers".format(placeholder_name))
        return " ".join(tokens)

    if isinstance(value, numbers.Real):
        return _format_number(value, placeholder_name)
    if isinstance(value, Mapping) or isinstance(value, (bytes, bytearray)):
        raise TypeError("substitution '{}' must be a numeric scalar or iterable".format(placeholder_name))
    if not isinstance(value, Iterable):
        raise TypeError("substitution '{}' must be a numeric scalar or iterable".format(placeholder_name))

    values = list(value)
    if not values:
        raise ValueError("substitution '{}' must not be empty".format(placeholder_name))
    return " ".join(_format_number(item, placeholder_name) for item in values)


def _remove_legacy_notes(document):
    for pattern in _LEGACY_NOTE_PATTERNS:
        document = pattern.sub("", document)
    return document


@dataclass(frozen=True)
class SymbolicUrdfTemplate:
    """A validated symbolic URDF and its ordered, unique placeholder names."""

    source: str

    def __post_init__(self):
        _validate_document(self.source)
        _parse_placeholder_names(self.source)

    @property
    def placeholder_names(self):
        return _parse_placeholder_names(self.source)

    def materialize(self, substitutions, *, keep_symbolic_notes=False):
        """Replace every placeholder with a finite numeric scalar or vector."""
        if not isinstance(substitutions, Mapping):
            raise TypeError("substitutions must be a mapping")
        if any(not isinstance(name, str) for name in substitutions):
            raise TypeError("substitution names must be strings")

        expected = set(self.placeholder_names)
        supplied = set(substitutions)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if missing:
            raise ValueError("missing substitutions: {}".format(", ".join(missing)))
        if unexpected:
            raise ValueError("unexpected substitutions: {}".format(", ".join(unexpected)))

        formatted = {
            name: _format_substitution(substitutions[name], name)
            for name in self.placeholder_names
        }
        searchable = _content_outside_comments(self.source)
        cursor = 0
        pieces = []
        for match in _PLACEHOLDER_PATTERN.finditer(searchable):
            pieces.append(self.source[cursor : match.start()])
            pieces.append(formatted[match.group("name")])
            cursor = match.end()
        pieces.append(self.source[cursor:])
        materialized = "".join(pieces)

        if not keep_symbolic_notes:
            materialized = _remove_legacy_notes(materialized)
        _validate_document(materialized)
        if _parse_placeholder_names(materialized):
            raise ValueError("materialized URDF still contains placeholders")
        return materialized


def parse_symbolic_urdf(document):
    """Validate a symbolic URDF and return a reusable parsed template."""
    return SymbolicUrdfTemplate(document)


def find_symbolic_urdf_placeholders(document):
    """Return ordered, unique placeholder names from a symbolic URDF."""
    return parse_symbolic_urdf(document).placeholder_names


def materialize_symbolic_urdf(document, substitutions, *, keep_symbolic_notes=False):
    """Parse and materialize a symbolic URDF in one call."""
    return parse_symbolic_urdf(document).materialize(
        substitutions,
        keep_symbolic_notes=keep_symbolic_notes,
    )
