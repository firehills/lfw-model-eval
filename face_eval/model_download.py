"""Safe HTTPS download helper for recognition model files."""

from pathlib import Path
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen
import warnings

from face_eval.io import sha256, write_json


def download_model(
    destination: Path, url: str, checksum: str | None = None, max_bytes: int = 2 * 1024**3,
) -> Path:
    """Download a model file without overwriting existing outputs."""
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination.suffix.lower() != ".onnx":
        raise ValueError("Model destination must end in .onnx")
    if urlparse(url).scheme != "https":
        raise ValueError("Model downloads require an HTTPS URL")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if checksum is not None:
        if len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum):
            raise ValueError("SHA-256 must be 64 hexadecimal characters")
    else:
        warnings.warn("No expected SHA-256 provided; authenticity is not verified", stacklevel=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = destination.parent / f"{destination.name}.download.json"
    with tempfile.TemporaryDirectory(prefix="face-model-", dir=destination.parent) as temp:
        staging = Path(temp) / destination.name
        with urlopen(url, timeout=60) as response, staging.open("wb") as stream:
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Download exceeds size limit")
                stream.write(chunk)
        actual = sha256(staging)
        if checksum and actual.lower() != checksum.lower():
            raise ValueError(f"Model checksum mismatch: expected {checksum}, got {actual}")
        staging.rename(destination)
    write_json(metadata, {"url": url, "sha256": actual, "file": destination.name})
    return destination
