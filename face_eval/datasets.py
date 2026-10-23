"""Dataset acquisition and deterministic identity-folder selection.

The data flow in this module is: download or extract a dataset archive -> validate it ->
select the first N enrolment images and next M test images per identity -> return a
stable, reproducible ordered list that downstream preprocessing can consume.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen
import warnings
import zipfile

from face_eval.io import contained_path, sha256, write_json


DATASETS = {
    "lfw": (
        "https://ndownloader.figshare.com/files/5976018",
        "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0",
    ),
    "lfw-funneled": (
        "https://ndownloader.figshare.com/files/5976015",
        "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",
    ),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".bmp"}


def extract_archive(archive: Path, destination: Path, max_bytes: int = 4 * 1024**3) -> None:
    """Validate and extract an archive without allowing traversal or symlink escapes."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            total = 0
            seen = set()
            for member in members:
                target = contained_path(destination, member.filename)
                mode = member.external_attr >> 16
                if mode & 0o170000 not in (0, 0o040000, 0o100000):
                    raise ValueError(f"Archive contains a link or special file: {member.filename}")
                if target in seen:
                    raise ValueError(f"Duplicate archive path: {member.filename}")
                seen.add(target)
                total += member.file_size
                if total > max_bytes:
                    raise ValueError("Archive exceeds extraction size limit")
            for member in members:
                target = contained_path(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive, "r:*") as source:
            members = source.getmembers()
            total = 0
            seen = set()
            for member in members:
                if member.name.rstrip("/") in {".", ""} and member.isdir():
                    continue
                target = contained_path(destination, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Archive contains a link or special file: {member.name}")
                if target in seen:
                    raise ValueError(f"Duplicate archive path: {member.name}")
                seen.add(target)
                total += member.size
                if total > max_bytes:
                    raise ValueError("Archive exceeds extraction size limit")
            for member in members:
                if member.name.rstrip("/") in {".", ""} and member.isdir():
                    continue
                target = contained_path(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = source.extractfile(member)
                    if src is None:
                        raise ValueError(f"Cannot read archive member: {member.name}")
                    with src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)


def download_dataset(
    destination: Path, dataset: str = "lfw", url: str | None = None,
    checksum: str | None = None, max_bytes: int = 4 * 1024**3,
) -> Path:
    """Download a dataset archive to a fresh destination and verify its checksum.

    Parameters
    ----------
    destination:
        Final directory where the extracted dataset should be placed.
    dataset:
        Dataset key name; supported values are defined in DATASETS.
    url:
        Optional override for the remote dataset URL.
    checksum:
        Optional SHA-256 expected for the remote dataset archive.
    max_bytes:
        Maximum payload size accepted for the archive to prevent oversized downloads.
    """
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if url is None:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        url, default_checksum = DATASETS[dataset]
        checksum = checksum or default_checksum
    if urlparse(url).scheme != "https":
        raise ValueError("Dataset downloads require an HTTPS URL")
    if checksum is not None:
        if len(checksum) != 64 or any(c not in "0123456789abcdefABCDEF" for c in checksum):
            raise ValueError("SHA-256 must be 64 hexadecimal characters")
    else:
        warnings.warn("Custom dataset has no expected SHA-256; authenticity is not verified", stacklevel=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="face-dataset-", dir=destination.parent) as temp:
        staging = Path(temp)
        archive = staging / "download"
        with urlopen(url, timeout=60) as response, archive.open("wb") as stream:
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Download exceeds size limit")
                stream.write(chunk)
        actual = sha256(archive)
        if checksum and actual.lower() != checksum.lower():
            raise ValueError(f"Dataset checksum mismatch: expected {checksum}, got {actual}")
        extracted = staging / "extracted"
        extracted.mkdir()
        extract_archive(archive, extracted, max_bytes)
        write_json(extracted / "download.json", {"dataset": dataset, "url": url, "sha256": actual})
        extracted.rename(destination)
    return destination


@dataclass(frozen=True)
class SelectedImage:
    """Concrete image choice used by dataset preparation, including identity and split."""

    identity: str
    path: Path
    split: str


def select_images(
    root: Path, enrolment: int, testing: int, identities: list[str] | None = None,
    max_identities: int | None = None,
) -> tuple[list[SelectedImage], list[dict]]:
    """Select a deterministic first-N-per-identity split for evaluation.

    The function walks identities alphabetically, keeps only those with enough files, and
    returns the first `enrolment` files for enrolment and the next `testing` files for test.
    This ordering is stable and reproducible across runs.
    """
    if enrolment < 1 or testing < 1:
        raise ValueError("enrolment and testing image counts must be positive")
    if max_identities is not None and max_identities < 2:
        raise ValueError("max_identities must be at least 2")
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    if identities is not None and len(set(identities)) != len(identities):
        raise ValueError("Identity selection contains duplicates")
    folders = {p.name: p for p in root.iterdir() if p.is_dir() and not p.is_symlink()}
    names = sorted(folders) if identities is None else sorted(identities)
    selected, excluded = [], []
    kept = 0
    for name in names:
        if max_identities is not None and kept >= max_identities:
            break
        if name not in folders:
            raise ValueError(f"Identity not found: {name}")
        images = sorted(
            (p for p in folders[name].iterdir()
             if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: p.name,
        )
        if len(images) < enrolment + testing:
            if identities is not None:
                raise ValueError(f"{name} has {len(images)} images; needs {enrolment + testing}")
            excluded.append({"identity": name, "available": len(images)})
            continue
        kept += 1
        for i, image in enumerate(images[:enrolment + testing]):
            selected.append(SelectedImage(name, image, "enrolment" if i < enrolment else "test"))
    if not selected:
        raise ValueError("No identities have enough images")
    return selected, excluded
