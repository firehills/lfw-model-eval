"""Small shared persistence helpers.

These helpers keep the on-disk artifacts consistent and prevent path traversal or
checksum mismatches from silently corrupting model, dataset, and report outputs.
"""

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Compute a SHA-256 digest for a file.

    Parameters
    ----------
    path:
        File whose bytes should be hashed. This is used to validate dataset archives,
        prepared tensors, and model files before they are trusted.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write a JSON document in a deterministic, newline-terminated format."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def contained_path(root: Path, relative: str) -> Path:
    """Resolve a path relative to a trusted root and reject escaping attempts.

    This prevents directory traversal such as ../.. and ensures all manifest-referenced
    files remain inside the dataset or report directory tree.
    """
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Path escapes its root: {relative}")
    return path
