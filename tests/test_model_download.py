import io

import pytest

from face_eval.io import sha256
from face_eval.model_download import download_model


def test_download_model_checks_checksum_and_output_rules(tmp_path, monkeypatch):
    import face_eval.model_download as model_download

    source = tmp_path / "source.onnx"
    source.write_bytes(b"onnx-binary")
    monkeypatch.setattr(model_download, "urlopen", lambda *args, **kwargs: io.BytesIO(source.read_bytes()))

    destination = tmp_path / "models" / "arcface.onnx"
    download_model(destination, "https://example.test/model.onnx", sha256(source))
    assert destination.read_bytes() == b"onnx-binary"
    assert (tmp_path / "models" / "arcface.onnx.download.json").is_file()

    with pytest.raises(FileExistsError):
        download_model(destination, "https://example.test/model.onnx")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_model(tmp_path / "models" / "mismatch.onnx", "https://example.test/model.onnx", "0" * 64)
    with pytest.raises(ValueError, match="HTTPS"):
        download_model(tmp_path / "models" / "http.onnx", "http://example.test/model.onnx")
    with pytest.raises(ValueError, match=r"\.onnx"):
        download_model(tmp_path / "models" / "model.bin", "https://example.test/model.onnx")
    with pytest.raises(ValueError, match="size limit"):
        download_model(tmp_path / "models" / "large.onnx", "https://example.test/model.onnx", max_bytes=1)
