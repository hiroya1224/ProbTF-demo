"""Small provenance helpers shared by the redesigned data contracts."""

import re
from typing import Any

from ..episode import sha256_file, stable_hash


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validated_sha256(value: Any, field_name: str = "sha256") -> str:
    """Return a normalized SHA-256 digest or reject incomplete provenance."""

    digest = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(field_name))
    return digest


__all__ = ["sha256_file", "stable_hash", "validated_sha256"]
