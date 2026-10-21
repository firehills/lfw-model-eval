import io
import tarfile
import zipfile

import pytest

from face_eval.datasets import download_dataset, extract_archive, select_images
from face_eval.io import sha256


def test_selection_is_sorted_disjoint_and_exact(image_dataset):
    selected, excluded = select_images(image_dataset, 2, 2)
    assert excluded == []
    assert len(selected) == 8
    for identity in ("Alice", "Bob"):
        images = [s for s in selected if s.identity == identity]
        assert [s.split for s in images] == ["enrolment", "enrolment", "test", "test"]
        assert [s.path.name for s in images] == [f"{identity}_{i:04d}.bmp" for i in range(4)]
        assert not {s.path for s in images[:2]} & {s.path for s in images[2:]}


def test_insufficient_identities_are_recorded_or_error(image_dataset):
    (image_dataset / "Few").mkdir()
    _, excluded = select_images(image_dataset, 1, 1)
    assert excluded == [{"identity": "Few", "available": 0}]
    with pytest.raises(ValueError, match="needs"):
        select_images(image_dataset, 1, 1, ["Few"])
    with pytest.raises(ValueError, match="positive"):
        select_images(image_dataset, 0, 1)
    with pytest.raises(ValueError, match="not found"):
        select_images(image_dataset, 1, 1, ["Missing"])
    with pytest.raises(ValueError, match="duplicates"):
        select_images(image_dataset, 1, 1, ["Alice", "Alice"])


def test_max_identities_skips_ineligible_and_limits(image_dataset):
    (image_dataset / "Aaron").mkdir()
    (image_dataset / "Carol").mkdir()
    (image_dataset / "Bob").rename(image_dataset / "Ben")
    for i in range(4):
        (image_dataset / "Alice" / f"Alice_{i:04d}.bmp").rename(image_dataset / "Carol" / f"Carol_{i:04d}.bmp")
    (image_dataset / "Alice").rmdir()
    (image_dataset / "Dan").mkdir()
    for i in range(2):
        (image_dataset / "Dan" / f"Dan_{i:04d}.jpg").write_bytes(b"")
    selected, excluded = select_images(image_dataset, 1, 1, max_identities=2)
    assert sorted({s.identity for s in selected}) == ["Ben", "Carol"]
    assert excluded == [{"identity": "Aaron", "available": 0}]
    with pytest.raises(ValueError, match="at least 2"):
        select_images(image_dataset, 1, 1, max_identities=1)


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_valid_archive_and_size_limit(tmp_path, kind):
    archive = tmp_path / f"data.{kind}"
    if kind == "tar":
        with tarfile.open(archive, "w") as output:
            info = tarfile.TarInfo("identity/image.jpg")
            info.size = 4
            output.addfile(info, io.BytesIO(b"test"))
    else:
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("identity/image.jpg", b"test")
    extract_archive(archive, tmp_path / "valid")
    assert (tmp_path / "valid/identity/image.jpg").read_bytes() == b"test"
    with pytest.raises(ValueError, match="size limit"):
        extract_archive(archive, tmp_path / "small", max_bytes=3)
    assert not (tmp_path / "small").exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
@pytest.mark.parametrize("name", ["../outside", "/absolute"])
def test_rejects_archive_traversal(tmp_path, kind, name):
    archive = tmp_path / f"bad.{kind}"
    if kind == "tar":
        with tarfile.open(archive, "w") as output:
            info = tarfile.TarInfo(name)
            output.addfile(info, io.BytesIO())
    else:
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(name, b"")
    with pytest.raises(ValueError, match="escapes"):
        extract_archive(archive, tmp_path / "output")


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_rejects_archive_links(tmp_path, kind):
    archive = tmp_path / f"link.{kind}"
    if kind == "tar":
        with tarfile.open(archive, "w") as output:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            output.addfile(info)
    else:
        with zipfile.ZipFile(archive, "w") as output:
            info = zipfile.ZipInfo("link")
            info.external_attr = 0o120777 << 16
            output.writestr(info, b"target")
    with pytest.raises(ValueError, match="link"):
        extract_archive(archive, tmp_path / "output")


def test_download_checks_checksum_and_preserves_archive_structure(tmp_path, monkeypatch):
    import face_eval.datasets as datasets
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("people/Alice/photo.jpg", b"image")
    monkeypatch.setattr(datasets, "urlopen", lambda *args, **kwargs: io.BytesIO(archive.read_bytes()))
    destination = tmp_path / "download"
    download_dataset(destination, url="https://example.test/data", checksum=sha256(archive))
    assert (destination / "people/Alice/photo.jpg").read_bytes() == b"image"
    assert (destination / "download.json").is_file()
    with pytest.raises(FileExistsError):
        download_dataset(destination)
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_dataset(tmp_path / "mismatch", url="https://example.test/data", checksum="0" * 64)
    assert not (tmp_path / "mismatch").exists()
    with pytest.raises(ValueError, match="HTTPS"):
        download_dataset(tmp_path / "http", url="http://example.test/data")
    with pytest.raises(ValueError, match="size limit"):
        download_dataset(tmp_path / "large", max_bytes=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["download", "source.zip"]
