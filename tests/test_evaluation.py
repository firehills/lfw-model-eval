from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from conftest import FixedDetector
from face_eval.evaluation import evaluate, load_manifest
from face_eval.preprocessing import prepare_dataset
from face_eval.runtimes import embedding_vector, load_runtime


def make_onnx(path: Path, config, dimension: int):
    rng = np.random.default_rng(42)
    weights = rng.uniform(0.01, 1, (3, dimension)).astype(np.float32)
    axes = [2, 3] if config.layout == "NCHW" else [1, 2]
    graph = helper.make_graph([
        helper.make_node("ReduceMean", ["input"], ["mean"], axes=axes, keepdims=0),
        helper.make_node("MatMul", ["mean", "weights"], ["embedding"]),
    ], "synthetic-embedding", [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, config.shape),
    ], [
        helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, dimension]),
    ], [numpy_helper.from_array(weights, "weights")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.mark.parametrize("dimension", [128, 192, 512])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_real_onnx_evaluation_and_reports(image_dataset, config, tmp_path, dimension, layout):
    config = replace(config, layout=layout)
    manifest = prepare_dataset(image_dataset, tmp_path / "tensors", config, FixedDetector(config), 2, 2)
    model = tmp_path / "model.onnx"
    make_onnx(model, config, dimension)
    output = tmp_path / "report"
    report = evaluate(manifest, model, output, "cosine", 0.95, warmup=1)
    assert report["model"]["embedding_dimension"] == dimension
    assert report["verification"]["true_accepted"] == 4
    assert report["verification"]["true_rejected"] == 4
    assert report["identification"]["rank1_accuracy"] == 1
    assert report["timing"]["count"] == 8
    assert report["timing"]["mean_ms"] > 0
    assert report["timing"]["by_split"]["test"]["count"] == 4
    persisted = json.loads((output / "report.json").read_text())
    assert persisted == report
    assert len((output / "scores.csv").read_text().splitlines()) == 9
    arrays = np.load(output / "embeddings.npz", allow_pickle=False)
    assert arrays["embeddings"].shape == (8, dimension)
    assert arrays["templates"].shape == (2, dimension)
    with pytest.raises(FileExistsError):
        evaluate(manifest, model, output, "cosine", 0.95)


def test_evaluate_cli(image_dataset, config, tmp_path):
    manifest = prepare_dataset(image_dataset, tmp_path / "tensors", config, FixedDetector(config), 1, 1)
    model = tmp_path / "model.onnx"
    make_onnx(model, config, 128)
    result = subprocess.run([
        sys.executable, "evaluate.py", "--manifest", str(manifest), "--model", str(model),
        "--output", str(tmp_path / "report"), "--threshold", "0.95",
        "--metric", "euclidean", "--warmup", "0",
    ], check=True, capture_output=True, text=True)
    assert "Full report:" in result.stdout
    assert (tmp_path / "report/report.json").is_file()


def test_manifest_rejects_corruption_leakage_and_traversal(image_dataset, config, tmp_path):
    manifest = prepare_dataset(image_dataset, tmp_path / "tensors", config, FixedDetector(config), 1, 1)
    original = json.loads(manifest.read_text())
    damaged = json.loads(manifest.read_text())
    damaged["images"][1]["source_sha256"] = damaged["images"][0]["source_sha256"]
    manifest.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="Identical source"):
        load_manifest(manifest)
    damaged["images"][0]["tensor"] = "../outside.bin"
    manifest.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="escapes"):
        load_manifest(manifest)
    manifest.write_text(json.dumps(original))
    tensor = manifest.parent / original["images"][0]["tensor"]
    data = tensor.read_bytes()
    tensor.write_bytes(bytes([data[0] ^ 255]) + data[1:])
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(manifest)
    tensor.write_bytes(b"short")
    with pytest.raises(ValueError, match="byte count"):
        load_manifest(manifest)


def test_model_contract_failures(tmp_path, config):
    model = tmp_path / "model.onnx"
    make_onnx(model, config, 128)
    with pytest.raises(ValueError, match="device"):
        load_runtime(model, config, device="tpu")
    with pytest.raises(ValueError, match="shape"):
        load_runtime(model, replace(config, size=32))
    with pytest.raises(ValueError, match="Output not found"):
        load_runtime(model, config, output_name="absent")
    for bad in (np.zeros((1, 128)), np.full((1, 128), np.nan), np.ones((2, 128))):
        with pytest.raises(ValueError):
            embedding_vector(bad)


@pytest.mark.parametrize("normalize", [False, True])
def test_template_averaging_and_optional_normalization(image_dataset, config, tmp_path, normalize):
    class KnownRuntime:
        metadata = {"runtime": "test", "device": "cpu"}

        def __init__(self):
            self.vectors = iter([
                [3, 0], [0, 4], [1, 1], [1, 1],
                [-3, 0], [0, -4], [-1, -1], [-1, -1],
            ])

        def embed(self, tensor):
            return np.asarray(next(self.vectors), dtype=float)

    manifest = prepare_dataset(image_dataset, tmp_path / "tensors", config, FixedDetector(config), 2, 2)
    model = tmp_path / "synthetic.onnx"
    model.write_bytes(b"test double; not loaded by a real runtime")
    output = tmp_path / "report"
    report = evaluate(
        manifest, model, output, "dot", 0.5 if normalize else 3.5,
        normalize=normalize, warmup=0, runtime=KnownRuntime(),
    )
    expected = np.array([1, 1]) / np.sqrt(2) if normalize else np.array([1.5, 2])
    arrays = np.load(output / "embeddings.npz", allow_pickle=False)
    np.testing.assert_allclose(arrays["templates"][0], expected)
    np.testing.assert_allclose(arrays["templates"][1], -expected)
    assert report["verification"]["accuracy"] == 1


@pytest.mark.parametrize("quantized", [False, True])
def test_real_tflite_evaluation(image_dataset, config, tmp_path, quantized):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    config = replace(config, layout="NHWC")
    weights = tf.constant(np.random.default_rng(42).uniform(0.01, 1, (3, 192)), dtype=tf.float32)

    class Embedder(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(config.shape, tf.float32, name="image")])
        def embed(self, image):
            return tf.matmul(tf.reduce_mean(image, axis=[1, 2]), weights)

    module = Embedder()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([module.embed.get_concrete_function()], module)
    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative():
            for value in np.linspace(0, 1, 10):
                yield [np.full(config.shape, value, dtype=np.float32)]

        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    model = tmp_path / "model.tflite"
    model.write_bytes(converter.convert())
    if quantized:
        interpreter = tf.lite.Interpreter(model_path=str(model))
        details = interpreter.get_input_details()[0]
        scale, zero = details["quantization"]
        config = replace(config, dtype="int8", quantization_scale=scale, quantization_zero_point=zero)
    manifest = prepare_dataset(image_dataset, tmp_path / "tensors", config, FixedDetector(config), 2, 2)
    report = evaluate(manifest, model, tmp_path / "report", "cosine", 0.95, warmup=1)
    assert report["runtime"]["runtime"] == "tensorflow-lite"
    assert report["model"]["embedding_dimension"] == 192
    assert report["verification"]["accuracy"] == 1
    assert report["identification"]["rank1_accuracy"] == 1
    if quantized:
        wrong_scale = replace(config, quantization_scale=config.quantization_scale * 2)
        with pytest.raises(ValueError, match="quantization does not match"):
            load_runtime(model, wrong_scale)


def test_gpu_never_falls_back_to_cpu(tmp_path, config, monkeypatch):
    import onnxruntime as ort
    model = tmp_path / "model.onnx"
    make_onnx(model, config, 128)
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            runtime = load_runtime(model, config, device="gpu")
        except RuntimeError as error:
            assert "CUDA" in str(error)
        else:
            assert runtime.metadata["device"] == "gpu"
            assert runtime.metadata["providers"][0] == "CUDAExecutionProvider"
            assert runtime.embed(np.ones(config.shape, dtype=np.float32)).shape == (128,)
    else:
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
            load_runtime(model, config, device="gpu")

    monkeypatch.delenv("TFLITE_GPU_DELEGATE", raising=False)
    tflite = tmp_path / "model.tflite"
    tflite.write_bytes(b"unused")
    with pytest.raises(ValueError, match="GPU delegate"):
        load_runtime(tflite, config, device="gpu")
    monkeypatch.setenv("TFLITE_GPU_DELEGATE", str(tmp_path / "missing.so"))
    with pytest.raises(FileNotFoundError, match="delegate"):
        load_runtime(tflite, config, device="gpu")
