"""Explicit, model-specific preprocessing contracts."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    name: str
    size: int = 112
    layout: str = "NCHW"
    color: str = "RGB"
    mean: tuple[float, ...] = (127.5, 127.5, 127.5)
    std: tuple[float, ...] = (127.5, 127.5, 127.5)
    dtype: str = "float32"
    quantization_scale: float | None = None
    quantization_zero_point: int | None = None
    alignment: str = "arcface"
    landmarks: tuple[tuple[float, float], ...] | None = None
    detection_threshold: float = 0.9
    face_selection: str = "largest"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or type(self.size) is not int or self.size < 1:
            raise ValueError("A name and positive integer input size are required")
        if self.layout not in {"NCHW", "NHWC"} or self.color not in {"RGB", "BGR"}:
            raise ValueError("layout must be NCHW/NHWC and color must be RGB/BGR")
        if self.dtype not in {"float32", "uint8", "int8"}:
            raise ValueError("dtype must be float32, uint8 or int8")
        for name, values in (("mean", self.mean), ("std", self.std)):
            if np.shape(values) != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite channel values")
        if np.any(np.asarray(self.std) <= 0):
            raise ValueError("std must be positive")
        if self.alignment not in {"arcface", "custom"}:
            raise ValueError("alignment must be arcface or custom")
        if self.alignment == "custom":
            if np.shape(self.landmarks) != (5, 2) or not np.isfinite(self.landmarks).all():
                raise ValueError("custom alignment requires five finite [x, y] landmarks")
            if np.linalg.matrix_rank(np.asarray(self.landmarks) - np.mean(self.landmarks, axis=0)) < 2:
                raise ValueError("Custom landmarks must span two dimensions")
        elif self.landmarks is not None:
            raise ValueError("Use alignment=custom when supplying landmarks")
        if not np.isfinite(self.detection_threshold) or not 0 < self.detection_threshold <= 1:
            raise ValueError("detection_threshold must be in (0, 1]")
        if self.face_selection not in {"largest", "single"}:
            raise ValueError("face_selection must be largest or single")
        if self.dtype == "float32":
            if self.quantization_scale is not None or self.quantization_zero_point is not None:
                raise ValueError("float32 inputs must not specify quantization")
        else:
            scale, zero = self.quantization_scale, self.quantization_zero_point
            limits = np.iinfo(self.dtype)
            if scale is None or not np.isfinite(scale) or scale <= 0:
                raise ValueError("Integer tensors require a positive quantization_scale")
            if type(zero) is not int or not limits.min <= zero <= limits.max:
                raise ValueError("Integer tensors require a representable quantization_zero_point")

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, 3, self.size, self.size) if self.layout == "NCHW" else (1, self.size, self.size, 3)

    def target_landmarks(self) -> np.ndarray:
        if self.alignment == "custom":
            return np.asarray(self.landmarks, dtype=np.float64)
        return np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041],
        ]) * (self.size / 112.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open() as stream:
            return cls(**json.load(stream))
