"""Detect, align and serialize disjoint enrolment/test image tensors."""

import argparse
from pathlib import Path

from face_eval.config import ModelConfig
from face_eval.preprocessing import RetinaFaceDetector, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing identity folders")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New tensor directory")
    parser.add_argument("--enrolment", type=int, required=True)
    parser.add_argument("--testing", type=int, required=True)
    parser.add_argument("--identities", nargs="+", help="Exact identity folder names; default: all eligible")
    parser.add_argument("--max-identities", type=int,
                        help="Use only the first N eligible identities (alphabetical)")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-image progress")
    args = parser.parse_args()
    config = ModelConfig.load(args.config)
    detector = RetinaFaceDetector(config.detection_threshold)
    manifest = prepare_dataset(
        args.dataset_root, args.output, config, detector,
        args.enrolment, args.testing, args.identities, args.max_identities, not args.quiet,
    )
    print(manifest)


if __name__ == "__main__":
    main()
