"""Download and safely extract LFW or an HTTPS-hosted identity-folder archive."""

import argparse
from pathlib import Path

from face_eval.datasets import DATASETS, download_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="lfw")
    parser.add_argument("--url", help="Custom HTTPS tar/zip archive; overrides dataset URL")
    parser.add_argument("--sha256", help="Expected archive SHA-256")
    parser.add_argument("--output", type=Path, required=True, help="New extraction directory")
    parser.add_argument("--max-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args()
    print(download_dataset(args.output, args.dataset, args.url, args.sha256, args.max_bytes))


if __name__ == "__main__":
    main()
