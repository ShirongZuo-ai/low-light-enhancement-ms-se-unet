from __future__ import annotations

import csv
from pathlib import Path


FIELDNAMES = [
    "model",
    "multiscale",
    "se_attention",
    "loss",
    "average_psnr",
    "average_ssim",
    "average_inference_time",
]

RUNS = [
    {
        "model": "unet_baseline",
        "multiscale": "False",
        "se_attention": "False",
        "loss": "L1",
        "metrics_path": Path("results/metrics_unet.csv"),
    },
    {
        "model": "ms_se_unet_l1",
        "multiscale": "True",
        "se_attention": "True",
        "loss": "L1",
        "metrics_path": Path("results/metrics_ms_se_unet_l1.csv"),
    },
    {
        "model": "ms_se_unet_combined",
        "multiscale": "True",
        "se_attention": "True",
        "loss": "Combined",
        "metrics_path": Path("results/metrics_ms_se_unet.csv"),
    },
]


def read_average(metrics_path: Path) -> tuple[str, str, str]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_path}")

    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("filename") == "AVERAGE":
                return row["psnr"], row["ssim"], row["inference_time"]

    raise ValueError(f"Missing AVERAGE row in metrics file: {metrics_path}")


def main() -> None:
    rows: list[dict[str, str]] = []
    for run in RUNS:
        average_psnr, average_ssim, average_time = read_average(run["metrics_path"])
        rows.append(
            {
                "model": run["model"],
                "multiscale": run["multiscale"],
                "se_attention": run["se_attention"],
                "loss": run["loss"],
                "average_psnr": average_psnr,
                "average_ssim": average_ssim,
                "average_inference_time": average_time,
            }
        )

    output_path = Path("results/metrics_ablation.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved ablation metrics: {output_path}")


if __name__ == "__main__":
    main()
