from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from face_eval.config import ModelConfig
from face_eval.preprocessing import Face


class FixedDetector:
    def __init__(self, config: ModelConfig):
        self.config = config

    def detect(self, image):
        height, width = image.shape[:2]
        return [Face((0, 0, width, height), self.config.target_landmarks(), 0.99)]


@pytest.fixture
def image_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    for index, identity in enumerate(("Alice", "Bob")):
        folder = root / identity
        folder.mkdir(parents=True)
        for i in range(4):
            pixels = np.full((16, 16, 3), 5 + i, dtype=np.uint8)
            pixels[:, :, index] = 200 + i
            Image.fromarray(pixels).save(folder / f"{identity}_{i:04d}.bmp")
    return root


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(name="synthetic", size=16, mean=(0, 0, 0), std=(255, 255, 255))
