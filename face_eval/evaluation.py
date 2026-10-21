"""Manifest validation, embedding timing, enrolment templates and reports."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

import numpy as np

from face_eval.config import ModelConfig
from face_eval.io import contained_path, sha256, write_json
from face_eval.metrics import higher_is_better, score, unit, verification_metrics
from face_eval.runtimes import Runtime, load_runtime


def load_manifest(path: Path) -> tuple[dict, ModelConfig]:
    with path.open() as stream:
        manifest = json.load(stream)
    if manifest["version"] != 1 or manifest["byte_order"] != "little":
        raise ValueError("Unsupported manifest version or byte order")
    config = ModelConfig(**manifest["config"])
    if manifest["shape"] != list(config.shape) or manifest["dtype"] != config.dtype:
        raise ValueError("Manifest shape/dtype disagrees with preprocessing configuration")
    images = manifest["images"]
    if not images:
        raise ValueError("Manifest contains no images")
    tensor_paths, sources, content_hashes = set(), set(), {}
    enrolment_ids, test_ids = set(), set()
    for entry in images:
        identity = entry["identity"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("Each image requires a nonempty identity")
        split = entry["split"]
        if split not in {"enrolment", "test"}:
            raise ValueError(f"Unknown split: {split}")
        (enrolment_ids if split == "enrolment" else test_ids).add(identity)
        tensor_path = contained_path(path.parent, entry["tensor"])
        if tensor_path in tensor_paths or entry["source"] in sources:
            raise ValueError("Duplicate tensor or source image in manifest")
        tensor_paths.add(tensor_path)
        sources.add(entry["source"])
        source_hash = entry["source_sha256"]
        if source_hash in content_hashes and content_hashes[source_hash] != split:
            raise ValueError("Identical source image content occurs in both enrolment and test")
        content_hashes[source_hash] = split
        expected_bytes = int(np.prod(config.shape)) * np.dtype(config.dtype).itemsize
        if tensor_path.stat().st_size != expected_bytes:
            raise ValueError(f"Incorrect tensor byte count: {tensor_path}")
        if sha256(tensor_path) != entry["tensor_sha256"]:
            raise ValueError(f"Tensor checksum mismatch: {tensor_path}")
    if len(enrolment_ids) < 2 or test_ids != enrolment_ids:
        raise ValueError("Need at least two identities, each with enrolment and test images")
    return manifest, config


def read_tensor(root: Path, entry: dict, config: ModelConfig) -> np.ndarray:
    tensor = np.fromfile(
        contained_path(root, entry["tensor"]), dtype=np.dtype(config.dtype).newbyteorder("<"),
    ).reshape(config.shape)
    if not np.isfinite(tensor).all():
        raise ValueError(f"Nonfinite tensor: {entry['tensor']}")
    return tensor


def evaluate(
    manifest_path: Path, model: Path, output: Path, metric: str, threshold: float,
    normalize: bool = True, warmup: int = 3, threads: int = 1, device: str = "cpu",
    output_name: str | None = None, runtime: Runtime | None = None,
    gpu_id: int = 0, gpu_delegate: Path | None = None,
) -> dict:
    if not np.isfinite(threshold) or warmup < 0:
        raise ValueError("threshold must be finite and warmup must be nonnegative")
    higher = higher_is_better(metric)
    manifest, config = load_manifest(manifest_path)
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    model_hash = sha256(model)
    runtime = runtime or load_runtime(model, config, threads, device, output_name, gpu_id, gpu_delegate)
    images = manifest["images"]
    first = read_tensor(manifest_path.parent, images[0], config)
    for _ in range(warmup):
        runtime.embed(first)
    embeddings, durations = [], []
    dimension = None
    for entry in images:
        tensor = read_tensor(manifest_path.parent, entry, config)
        started = perf_counter_ns()
        vector = runtime.embed(tensor)
        durations.append((perf_counter_ns() - started) / 1_000_000)
        if dimension is not None and len(vector) != dimension:
            raise ValueError("Embedding dimension changed between images")
        dimension = len(vector)
        embeddings.append(unit(vector) if normalize else vector)
    embeddings = np.stack(embeddings)
    identities = sorted({e["identity"] for e in images})
    templates = []
    for identity in identities:
        members = [
            i for i, e in enumerate(images)
            if e["split"] == "enrolment" and e["identity"] == identity
        ]
        template = embeddings[members].mean(axis=0)
        templates.append(unit(template) if normalize else template)
    templates = np.stack(templates)
    rows, labels, scores = [], [], []
    correct, correct_accepted, probes = 0, 0, 0
    for index, entry in enumerate(images):
        if entry["split"] != "test":
            continue
        values = score(embeddings[index], templates, metric)
        if not np.isfinite(values).all():
            raise ValueError("Distance metric produced nonfinite scores")
        best = int(np.argmax(values) if higher else np.argmin(values))
        accepted = values >= threshold if higher else values <= threshold
        matched = identities[best] == entry["identity"]
        correct += int(matched)
        correct_accepted += int(matched and accepted[best])
        probes += 1
        for j, identity in enumerate(identities):
            genuine = identity == entry["identity"]
            labels.append(genuine)
            scores.append(float(values[j]))
            rows.append({
                "source": entry["source"], "probe_identity": entry["identity"],
                "claimed_identity": identity, "genuine": genuine,
                "score": float(values[j]), "accepted": bool(accepted[j]),
                "rank1_identity": identities[best],
            })
    metrics = verification_metrics(np.array(labels), np.array(scores), threshold, metric)
    times = np.asarray(durations)
    by_split = {}
    for split in ("enrolment", "test"):
        split_times = times[[i for i, e in enumerate(images) if e["split"] == split]]
        by_split[split] = {"count": len(split_times), "mean_ms": float(split_times.mean())}
    report = {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "enrolment-template-verification",
        "model": {"path": str(model.resolve()), "sha256": model_hash, "embedding_dimension": dimension},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "preprocessing": config.to_dict(), "runtime": runtime.metadata,
        "environment": {
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
        },
        "settings": {
            "metric": metric, "threshold": threshold,
            "acceptance": "score >= threshold" if higher else "score <= threshold",
            "normalize_embeddings": normalize, "template": "arithmetic mean, then L2 if enabled",
            "warmup_runs": warmup,
        },
        "counts": {"identities": len(identities), "images": len(images), "probes": probes},
        "verification": metrics,
        "identification": {
            "rank1_accuracy": correct / probes,
            "rank1_correct_and_accepted_rate": correct_accepted / probes,
            "tie_break": "alphabetically first identity",
        },
        "timing": {
            "scope": "synchronous embed call including input/output transfer and output validation; excludes image preprocessing, file I/O, model loading, warmup and template construction",
            "count": len(times), "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "min_ms": float(times.min()), "max_ms": float(times.max()),
            "by_split": by_split,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        output / "embeddings.npz", embeddings=embeddings, templates=templates,
        template_identities=np.asarray(identities),
        sources=np.asarray([e["source"] for e in images]),
        identities=np.asarray([e["identity"] for e in images]),
        splits=np.asarray([e["split"] for e in images]), times_ms=times,
    )
    write_json(output / "report.json", report)
    return report
