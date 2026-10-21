from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest

from conftest import FixedDetector
from face_eval.config import ModelConfig
from face_eval.preprocessing import Face, Preprocessor, RetinaFaceDetector, prepare_dataset, similarity_transform


@pytest.mark.parametrize("size", [112, 240, 320, 57])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
@pytest.mark.parametrize("suffix", [".bmp", ".jpg"])
def test_tensor_shape_layout_color_normalization(tmp_path, size, layout, suffix):
    config = ModelConfig(
        name="test", size=size, layout=layout, color="BGR",
        mean=(1, 2, 3), std=(2, 4, 5),
    )
    path = tmp_path / f"image{suffix}"
    Image.new("RGB", (size, size), color=(120, 60, 30)).save(path)
    tensor, _ = Preprocessor(config, FixedDetector(config)).process(path)
    with Image.open(path) as image:
        rgb_pixel = np.asarray(image)[size // 2, size // 2].astype(float)
    expected = (rgb_pixel[::-1] - [1, 2, 3]) / [2, 4, 5]
    pixel = tensor[0, :, size // 2, size // 2] if layout == "NCHW" else tensor[0, size // 2, size // 2]
    assert tensor.shape == config.shape
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    np.testing.assert_allclose(pixel, expected, rtol=1e-6)


def test_similarity_recovers_rotation_scale_translation(config):
    source = config.target_landmarks()
    expected = np.array([[1.8, -0.4, 4], [0.4, 1.8, -3]])
    target = np.column_stack([source, np.ones(5)]) @ expected.T
    np.testing.assert_allclose(similarity_transform(source, target), expected, atol=1e-12)
    with pytest.raises(ValueError, match="Degenerate"):
        similarity_transform(np.zeros((5, 2)), target)


def test_quantized_pixels_and_saturation(tmp_path):
    config = ModelConfig(
        name="quantized", size=16, layout="NHWC", mean=(0, 0, 0), std=(1, 1, 1),
        dtype="int8", quantization_scale=2.0, quantization_zero_point=-10,
    )
    path = tmp_path / "image.bmp"
    Image.new("RGB", (16, 16), (20, 40, 255)).save(path)
    tensor, _ = Preprocessor(config, FixedDetector(config)).process(path)
    np.testing.assert_array_equal(tensor[0, 8, 8], [0, 10, 118])
    saturated = replace(config, quantization_scale=0.5)
    tensor, _ = Preprocessor(saturated, FixedDetector(saturated)).process(path)
    np.testing.assert_array_equal(tensor[0, 8, 8], [30, 70, 127])


def test_detection_failure_and_multiple_faces(image_dataset, config):
    class Empty:
        def detect(self, image):
            return []

    class Multiple:
        def detect(self, image):
            return [
                Face((0, 0, 8, 8), config.target_landmarks(), 0.99),
                Face((0, 0, 16, 16), config.target_landmarks(), 0.98),
            ]

    path = image_dataset / "Alice/Alice_0000.bmp"
    with pytest.raises(ValueError, match="No face"):
        Preprocessor(config, Empty()).process(path)
    with pytest.raises(ValueError, match="Expected one face"):
        Preprocessor(replace(config, face_selection="single"), Multiple()).process(path)
    _, face = Preprocessor(config, Multiple()).process(path)
    assert face.bbox == (0, 0, 16, 16)


def test_retinaface_adapter_orders_keypoints_in_image_coordinates():
    class Backend:
        @staticmethod
        def detect_faces(image, threshold, model):
            assert image.shape == (16, 16, 3)
            assert threshold == 0.9
            return {"face_1": {
                "facial_area": [0, 0, 16, 16], "score": 0.99,
                "landmarks": {
                    "left_eye": [12, 4], "right_eye": [4, 4], "nose": [8, 8],
                    "mouth_left": [12, 12], "mouth_right": [4, 12],
                },
            }}

    detector = RetinaFaceDetector.__new__(RetinaFaceDetector)
    detector.retinaface, detector.model, detector.threshold = Backend(), None, 0.9
    faces = detector.detect(np.zeros((16, 16, 3), dtype=np.uint8))
    assert len(faces) == 1
    np.testing.assert_array_equal(
        faces[0].landmarks, [[4, 4], [12, 4], [8, 8], [4, 12], [12, 12]],
    )


def test_manifest_and_binary_roundtrip(image_dataset, tmp_path, config):
    path = prepare_dataset(image_dataset, tmp_path / "tensors", config, FixedDetector(config), 2, 2)
    manifest = json.loads(path.read_text())
    assert manifest["shape"] == [1, 3, 16, 16]
    assert manifest["dtype"] == "float32"
    for entry in manifest["images"]:
        tensor_path = path.parent / entry["tensor"]
        assert tensor_path.name.endswith(".bmp.bin")
        assert entry["identity"] in str(tensor_path)
        assert tensor_path.stat().st_size == 1 * 3 * 16 * 16 * 4
        actual = np.fromfile(tensor_path, dtype="<f4").reshape(config.shape)
        expected, _ = Preprocessor(config, FixedDetector(config)).process(image_dataset / entry["source"])
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(FileExistsError):
        prepare_dataset(image_dataset, path.parent, config, FixedDetector(config), 2, 2)


def test_manifest_accepts_numpy_detector_values(image_dataset, tmp_path, config):
    # RetinaFace returns numpy integer boxes and numpy float scores.
    class NumpyDetector:
        def detect(self, image):
            return [Face(
                tuple(np.array([0, 0, 16, 16], dtype=np.int64)),
                config.target_landmarks(), np.float32(0.99),
            )]

    path = prepare_dataset(image_dataset, tmp_path / "tensors", config, NumpyDetector(), 2, 2)
    entry = json.loads(path.read_text())["images"][0]
    assert entry["bbox"] == [0.0, 0.0, 16.0, 16.0]
    assert entry["confidence"] == pytest.approx(0.99)


@pytest.mark.parametrize("kwargs", [
    {"size": 0}, {"size": 1.5}, {"layout": "CHW"}, {"color": "XYZ"},
    {"std": [0, 1, 1]}, {"mean": [float("nan"), 1, 1]},
    {"alignment": "custom"}, {"dtype": "int8"},
    {"quantization_scale": 1.0}, {"face_selection": "random"},
    {"detection_threshold": 2},
])
def test_invalid_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(name="bad", **kwargs)
