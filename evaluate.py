"""Evaluate ONNX/TFLite embeddings against enrolment identity templates."""

import argparse
import json
from pathlib import Path

from face_eval.evaluation import evaluate
from face_eval.metrics import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--metric", choices=METRICS, default="cosine")
    parser.add_argument("--threshold", type=float, required=True, help="Explicit acceptance threshold")
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="gpu: ONNX uses CUDAExecutionProvider, TFLite uses --gpu-delegate; no CPU fallback")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index for ONNX GPU runs")
    parser.add_argument("--gpu-delegate", type=Path,
                        help="TFLite GPU delegate .so (default: $TFLITE_GPU_DELEGATE)")
    parser.add_argument("--output-name", help="Embedding output name for models with multiple outputs")
    args = parser.parse_args()
    report = evaluate(
        args.manifest, args.model, args.output, args.metric, args.threshold,
        not args.no_normalize_embeddings, args.warmup, args.threads, args.device, args.output_name,
        gpu_id=args.gpu_id, gpu_delegate=args.gpu_delegate,
    )
    print(json.dumps({
        "verification": {k: v for k, v in report["verification"].items() if k != "roc"},
        "identification": report["identification"], "timing": report["timing"],
    }, indent=2, allow_nan=False))
    print(f"Full report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
