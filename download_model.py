"""Download an HTTPS-hosted ONNX model file with optional SHA-256 verification."""

import argparse
from pathlib import Path

from face_eval.model_download import download_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="HTTPS URL to the ONNX model file")
    parser.add_argument("--sha256", help="Expected model SHA-256 (64 hex chars)")
    parser.add_argument("--output", type=Path, required=True, help="New .onnx destination path")
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    print(download_model(args.output, args.url, args.sha256, args.max_bytes))


if __name__ == "__main__":
    main()
