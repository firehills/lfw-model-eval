"""Strict CPU/GPU inference adapters for ONNX and TensorFlow Lite.

GPU runs never fall back to CPU silently: if the requested accelerator cannot be used,
loading fails with an explicit error. This keeps the evaluation pipeline honest and makes
it clear when a user actually requested a GPU configuration.
"""

import os

from pathlib import Path
from typing import Protocol

import numpy as np

from face_eval.config import ModelConfig


class Runtime(Protocol):
    """Protocol describing the inference boundary used by the evaluation pipeline."""

    metadata: dict

    def embed(self, tensor: np.ndarray) -> np.ndarray: ...


def embedding_vector(output: np.ndarray) -> np.ndarray:
    """Validate and flatten model outputs into a single nonzero embedding vector."""
    output = np.asarray(output)
    if np.count_nonzero(np.asarray(output.shape) > 1) != 1 or output.size < 2:
        raise ValueError(f"Expected a single embedding vector, got shape {output.shape}")
    vector = output.astype(np.float64).reshape(-1)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Model produced a zero or nonfinite embedding")
    return vector


def check_input(shape, dtype, config: ModelConfig) -> None:
    """Verify that a runtime input matches the model preprocessing contract."""
    if len(shape) != 4 or any(
        isinstance(actual, (int, np.integer)) and actual > 0 and actual != expected
        for actual, expected in zip(shape, config.shape)
    ):
        raise ValueError(f"Model shape {shape} does not match configured shape {config.shape}")
    if np.dtype(dtype) != np.dtype(config.dtype):
        raise ValueError(f"Model dtype {dtype} does not match configured dtype {config.dtype}")


class OnnxRuntime:
    """ONNX Runtime adapter for float/uint8/int8 embedding models."""

    def __init__(
        self, path: Path, config: ModelConfig, threads: int, output_name: str | None,
        device: str = "cpu", gpu_id: int = 0,
    ):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        if device == "gpu":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable; install onnxruntime-gpu "
                    "(use the GPU devcontainer) or run with --device cpu"
                )
            # ORT still needs the CPU provider for host-side ops; the CUDA check below
            # guarantees the session really runs on the GPU.
            providers = [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if device == "gpu" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"CUDA provider failed to initialise (active: {self.session.get_providers()}); "
                "check CUDA/cuDNN libraries"
            )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError("Only single-input embedding models are supported")
        self.input = inputs[0]
        dtypes = {"tensor(float)": "float32", "tensor(uint8)": "uint8", "tensor(int8)": "int8"}
        if self.input.type not in dtypes:
            raise ValueError(f"Unsupported ONNX input type: {self.input.type}")
        check_input(self.input.shape, dtypes[self.input.type], config)
        if output_name is None:
            if len(outputs) != 1:
                raise ValueError("Multiple outputs: select the embedding with --output-name")
            output_name = outputs[0].name
        if output_name not in {o.name for o in outputs}:
            raise ValueError(f"Output not found: {output_name}")
        output = next(o for o in outputs if o.name == output_name)
        if output.type not in {"tensor(float)", "tensor(double)", "tensor(float16)"}:
            raise ValueError("ONNX output must be floating point; export dequantized embeddings")
        self.output_name = output_name
        self.metadata = {
            "runtime": "onnxruntime", "version": ort.__version__, "device": device,
            "gpu_id": gpu_id if device == "gpu" else None,
            "providers": self.session.get_providers(), "threads": threads,
            "input_name": self.input.name, "output_name": output_name,
        }

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        """Run one model inference and convert the output into a feature vector."""
        output = self.session.run([self.output_name], {self.input.name: tensor})[0]
        return embedding_vector(output)


class TFLiteRuntime:
    """TensorFlow Lite adapter with optional GPU delegate support."""

    def __init__(
        self, path: Path, config: ModelConfig, threads: int, output_name: str | None,
        device: str = "cpu", gpu_delegate: Path | None = None,
    ):
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        delegates = []
        if device == "gpu":
            if gpu_delegate is None:
                raise ValueError(
                    "TFLite GPU inference needs a GPU delegate shared library: pass "
                    "--gpu-delegate or set TFLITE_GPU_DELEGATE (see README)"
                )
            if not Path(gpu_delegate).is_file():
                raise FileNotFoundError(f"TFLite GPU delegate not found: {gpu_delegate}")
            delegates = [tf.lite.experimental.load_delegate(str(gpu_delegate))]
        self.interpreter = tf.lite.Interpreter(
            model_path=str(path), num_threads=threads, experimental_delegates=delegates or None,
        )
        inputs = self.interpreter.get_input_details()
        if len(inputs) != 1:
            raise ValueError("Only single-input embedding models are supported")
        self.input = inputs[0]
        check_input(self.input["shape_signature"], self.input["dtype"], config)
        if tuple(self.input["shape"]) != config.shape:
            self.interpreter.resize_tensor_input(self.input["index"], config.shape, strict=True)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        outputs = self.interpreter.get_output_details()
        if output_name is None:
            if len(outputs) != 1:
                raise ValueError("Multiple outputs: select the embedding with --output-name")
            self.output = outputs[0]
        else:
            matches = [o for o in outputs if o["name"] == output_name]
            if not matches:
                raise ValueError(f"Output not found: {output_name}")
            self.output = matches[0]
        if config.dtype != "float32":
            parameters = self.input["quantization_parameters"]
            scales, zeros = parameters["scales"], parameters["zero_points"]
            if len(scales) != 1 or len(zeros) != 1:
                raise ValueError("Only per-tensor input quantization is supported")
            if not np.isclose(scales[0], config.quantization_scale) or zeros[0] != config.quantization_zero_point:
                raise ValueError("Model input quantization does not match preprocessing configuration")
        self.output_scale, self.output_zero = 1.0, 0
        if np.issubdtype(self.output["dtype"], np.integer):
            params = self.output["quantization_parameters"]
            if len(params["scales"]) != 1 or len(params["zero_points"]) != 1 or params["scales"][0] <= 0:
                raise ValueError("Only per-tensor quantized embedding outputs are supported")
            self.output_scale = float(params["scales"][0])
            self.output_zero = int(params["zero_points"][0])
        self.metadata = {
            "runtime": "tensorflow-lite", "version": tf.__version__, "device": device,
            "gpu_delegate": str(gpu_delegate) if delegates else None,
            "threads": threads, "input_name": self.input["name"],
            "output_name": self.output["name"],
        }

    def embed(self, tensor: np.ndarray) -> np.ndarray:
        """Set the input tensor, invoke the interpreter, and dequantize the output."""
        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output["index"])
        output = (output.astype(np.float64) - self.output_zero) * self.output_scale
        return embedding_vector(output)


def load_runtime(
    path: Path, config: ModelConfig, threads: int = 1,
    device: str = "cpu", output_name: str | None = None,
    gpu_id: int = 0, gpu_delegate: Path | None = None,
) -> Runtime:
    """Select the correct backend and instantiate the runtime for a given model file."""
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    if gpu_id < 0:
        raise ValueError("gpu_id must be nonnegative")
    if threads < 1:
        raise ValueError("threads must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".onnx":
        return OnnxRuntime(path, config, threads, output_name, device, gpu_id)
    if path.suffix.lower() == ".tflite":
        if gpu_delegate is None and os.environ.get("TFLITE_GPU_DELEGATE"):
            gpu_delegate = Path(os.environ["TFLITE_GPU_DELEGATE"])
        return TFLiteRuntime(path, config, threads, output_name, device, gpu_delegate)
    raise ValueError("Model must have an .onnx or .tflite extension")
