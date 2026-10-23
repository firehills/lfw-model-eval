"""Common image -> aligned, normalized, model-ready tensor entry point.

The preprocessing pipeline is the handoff between raw dataset images and the serialized
binary tensors consumed by the runtime. It takes a detected face, aligns it to a fixed
landmark template, normalizes it to the model contract, and emits exactly the tensor
shape/dtype expected by the runtime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageOps

from face_eval.config import ModelConfig
from face_eval.datasets import select_images
from face_eval.io import sha256, write_json


@dataclass(frozen=True)
class Face:
    """Face detection output with location, aligned landmark points, and confidence."""

    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    confidence: float


class Detector(Protocol):
    """Minimal protocol for any detector that can return face boxes and landmarks."""

    def detect(self, bgr_image: np.ndarray) -> list[Face]: ...


class RetinaFaceDetector:
    """Thin adapter around RetinaFace detection used for live preprocessing runs."""

    def __init__(self, threshold: float = 0.9):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        from retinaface import RetinaFace
        self.retinaface = RetinaFace
        self.model = RetinaFace.build_model()
        self.threshold = threshold

    def detect(self, bgr_image: np.ndarray) -> list[Face]:
        result = self.retinaface.detect_faces(
            bgr_image, threshold=self.threshold, model=self.model,
        )
        if not isinstance(result, dict):
            return []
        faces = []
        for detected in result.values():
            points = detected["landmarks"]
            # Canonical templates use image-left to image-right ordering, not anatomical
            # labels, so the points must be reordered before alignment.
            eyes = sorted([points["left_eye"], points["right_eye"]], key=lambda p: p[0])
            mouth = sorted([points["mouth_left"], points["mouth_right"]], key=lambda p: p[0])
            faces.append(Face(
                tuple(float(v) for v in detected["facial_area"]),
                np.asarray([*eyes, points["nose"], *mouth], dtype=np.float64),
                float(detected["score"]),
            ))
        return faces


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Estimate a least-squares similarity transform for five landmark correspondences."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != (5, 2) or target.shape != (5, 2):
        raise ValueError("Alignment requires five [x, y] keypoints")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Alignment keypoints must be finite")
    centered = source - source.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("Degenerate face keypoints")
    x, y = source.T
    design = np.zeros((10, 4))
    design[0::2] = np.column_stack([x, -y, np.ones(5), np.zeros(5)])
    design[1::2] = np.column_stack([y, x, np.zeros(5), np.ones(5)])
    a, b, tx, ty = np.linalg.lstsq(design, target.reshape(-1), rcond=None)[0]
    if a * a + b * b < 1e-12:
        raise ValueError("Degenerate alignment transform")
    return np.array([[a, -b, tx], [b, a, ty]])


class Preprocessor:
    """Reusable face preprocessing utility for model-ready tensor generation."""

    def __init__(self, config: ModelConfig, detector: Detector):
        self.config, self.detector = config, detector

    def process(self, image_path: str | Path) -> tuple[np.ndarray, Face]:
        """Read one image, detect the face, align it, normalize it, and emit a tensor."""
        with Image.open(image_path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        # The detector operates on BGR because RetinaFace exports its result in that color
        # space; the conversion keeps the bound between raw image data and model input exact.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        faces = [
            f for f in self.detector.detect(bgr)
            if np.isfinite(f.confidence) and f.confidence >= self.config.detection_threshold
        ]
        if not faces:
            raise ValueError(f"No face detected: {image_path}")
        if self.config.face_selection == "single" and len(faces) != 1:
            raise ValueError(f"Expected one face, found {len(faces)}: {image_path}")
        for face in faces:
            if len(face.bbox) != 4 or not np.isfinite(face.bbox).all():
                raise ValueError(f"Invalid face bounding box: {image_path}")
            if face.bbox[2] <= face.bbox[0] or face.bbox[3] <= face.bbox[1]:
                raise ValueError(f"Empty face bounding box: {image_path}")
        # The largest box is selected so a single source photo with multiple faces does not
        # silently generate the wrong identity template.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        transform = similarity_transform(face.landmarks, self.config.target_landmarks())
        aligned = cv2.warpAffine(
            rgb, transform, (self.config.size, self.config.size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        if self.config.color == "BGR":
            aligned = aligned[:, :, ::-1]
        values = (aligned.astype(np.float32) - np.asarray(self.config.mean, dtype=np.float32))
        values /= np.asarray(self.config.std, dtype=np.float32)
        if self.config.dtype != "float32":
            limits = np.iinfo(self.config.dtype)
            values = np.clip(
                np.rint(values / self.config.quantization_scale + self.config.quantization_zero_point),
                limits.min, limits.max,
            )
        if self.config.layout == "NCHW":
            values = values.transpose(2, 0, 1)
        tensor = np.ascontiguousarray(values[np.newaxis], dtype=self.config.dtype)
        if not np.isfinite(tensor).all():
            raise ValueError(f"Preprocessing produced a nonfinite tensor: {image_path}")
        return tensor, face


def prepare_dataset(
    root: Path, output: Path, config: ModelConfig, detector: Detector,
    enrolment: int, testing: int, identities: list[str] | None = None,
    max_identities: int | None = None, progress: bool = False,
) -> Path:
    """Prepare a manifest and binary tensors for a dataset split.

    Parameters
    ----------
    root:
        Root directory containing identity folders with raw image files.
    output:
        Destination for the prepared dataset export, including the manifest and binary bins.
    config:
        Preprocessing contract determining shape, dtype, normalization, and alignment.
    detector:
        Face detector producing face crops and landmarks.
    enrolment/testing:
        Number of source images assigned to each identity's enrolment and test lists.
    identities:
        Optional subset of identity names to include; the function filters to those names.
    max_identities:
        Hard cap on the number of identities retained in alphabetical order.
    progress:
        Print one line per processed image when enabled.
    """
    selected, excluded = select_images(root, enrolment, testing, identities, max_identities)
    output.mkdir(parents=True, exist_ok=False)
    processor = Preprocessor(config, detector)
    entries = []
    for number, selected_image in enumerate(selected, 1):
        source = selected_image.path.relative_to(root).as_posix()
        if progress:
            print(f"[{number}/{len(selected)}] {source}", flush=True)
        tensor, face = processor.process(selected_image.path)
        relative = Path(selected_image.split) / selected_image.identity / f"{selected_image.path.name}.bin"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Writing the tensor as little-endian bytes makes the binary on disk match the
        # manifest metadata, and the checksum is recorded separately for validation.
        tensor.astype(np.dtype(config.dtype).newbyteorder("<"), copy=False).tofile(destination)
        entries.append({
            "identity": selected_image.identity, "split": selected_image.split,
            "source": source, "source_sha256": sha256(selected_image.path),
            "tensor": relative.as_posix(), "tensor_sha256": sha256(destination),
            "bbox": [float(v) for v in face.bbox], "landmarks": face.landmarks.tolist(),
            "confidence": float(face.confidence),
        })
    manifest = output / "manifest.json"
    write_json(manifest, {
        "version": 1, "dataset_root": str(root.resolve()),
        "config": config.to_dict(), "shape": list(config.shape), "dtype": config.dtype,
        "byte_order": "little", "enrolment_per_identity": enrolment,
        "test_per_identity": testing, "excluded_identities": excluded, "images": entries,
    })
    return manifest
