"""Plot threshold sweep results from report.json files in reports/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(reports_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for report_path in sorted(reports_dir.glob("w600k_r50_at*/report.json")):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        verification = data.get("verification", {})
        settings = data.get("settings", {})
        rows.append(
            {
                "threshold": float(settings.get("threshold", 0.0)),
                "tar": float(verification.get("true_accept_rate", 0.0)),
                "far": float(verification.get("false_accept_rate", 0.0)),
                "frr": float(verification.get("false_reject_rate", 0.0)),
                "precision": float(verification.get("precision", 0.0)),
                "accuracy": float(verification.get("accuracy", 0.0)),
                "f1": float(verification.get("f1", 0.0)),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No report.json files found under {reports_dir / 'w600k_r50_at*'}")

    rows.sort(key=lambda item: item["threshold"])
    return rows


def plot_threshold_sweep(rows: list[dict], output_path: Path) -> None:
    thresholds = [row["threshold"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, [row["tar"] for row in rows], marker="o", label="TAR (Recall)")
    ax.plot(thresholds, [row["far"] for row in rows], marker="o", label="FAR")
    ax.plot(thresholds, [row["frr"] for row in rows], marker="o", label="FRR")
    ax.plot(thresholds, [row["precision"] for row in rows], marker="o", label="Precision")
    ax.plot(thresholds, [row["f1"] for row in rows], marker="o", label="F1")
    ax.plot(thresholds, [row["accuracy"] for row in rows], marker="o", label="Accuracy")

    ax.set_title(f"w600k_r50 threshold sweep ({len(rows)} points)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score / Rate")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory containing runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/w600k_r50_threshold_sweep.png"),
        help="Where to write the plot image",
    )
    args = parser.parse_args()

    rows = load_rows(args.reports_dir)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_threshold_sweep(rows, output_path)
    print(f"Saved plot to {output_path}")
    print("Thresholds used:", ", ".join(str(row["threshold"]) for row in rows))


if __name__ == "__main__":
    main()
