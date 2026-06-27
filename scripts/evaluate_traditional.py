from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods import clahe_enhance, gamma_enhance, retinex_enhance
from utils.metrics import calculate_psnr, calculate_ssim


EnhanceFunction = Callable[[np.ndarray], np.ndarray]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
METHODS: dict[str, EnhanceFunction] = {
    "gamma": gamma_enhance,
    "clahe": clahe_enhance,
    "retinex": retinex_enhance,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate traditional methods on the LOL test set.")
    parser.add_argument("--data-root", type=Path, default=Path("data/LOL"), help="Root directory of the LOL dataset.")
    parser.add_argument("--image-size", type=int, default=256, help="Square size used for metric calculation.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/traditional_eval"),
        help="Root directory for enhanced traditional method outputs.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for metrics CSV files.")
    parser.add_argument(
        "--unet-metrics",
        type=Path,
        default=Path("results/metrics_unet.csv"),
        help="Existing U-Net metrics CSV to include in the comparison table.",
    )
    parser.add_argument(
        "--ms-se-unet-metrics",
        type=Path,
        default=Path("results/metrics_ms_se_unet.csv"),
        help="Existing MS-SE-U-Net metrics CSV to include in the comparison table.",
    )
    return parser.parse_args()


def collect_pairs(data_root: Path) -> list[tuple[str, Path, Path]]:
    low_dir = data_root / "test" / "low"
    high_dir = data_root / "test" / "high"
    if not low_dir.is_dir():
        raise FileNotFoundError(f"Missing low image directory: {low_dir}")
    if not high_dir.is_dir():
        raise FileNotFoundError(f"Missing high image directory: {high_dir}")

    low_paths = {
        path.name: path
        for path in sorted(low_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    high_paths = {
        path.name: path
        for path in sorted(high_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    missing_high = sorted(set(low_paths) - set(high_paths))
    missing_low = sorted(set(high_paths) - set(low_paths))
    if missing_high or missing_low:
        message = ["LOL test low/high filenames do not match."]
        if missing_high:
            message.append(f"Missing in high/: {missing_high[:20]}")
        if missing_low:
            message.append(f"Missing in low/: {missing_low[:20]}")
        raise ValueError(" ".join(message))

    return [(name, low_paths[name], high_paths[name]) for name in sorted(low_paths)]


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image with OpenCV: {path}")
    return image


def save_bgr(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


def bgr_to_metric_tensor(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image_float = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(image_float, (2, 0, 1))).float()


def write_method_metrics(results_dir: Path, method: str, rows: list[dict[str, str]]) -> tuple[float, float, float]:
    average_psnr = sum(float(row["psnr"]) for row in rows) / len(rows)
    average_ssim = sum(float(row["ssim"]) for row in rows) / len(rows)
    average_time = sum(float(row["inference_time"]) for row in rows) / len(rows)

    metrics_path = results_dir / f"metrics_{method}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "inference_time"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "filename": "AVERAGE",
                "psnr": f"{average_psnr:.6f}",
                "ssim": f"{average_ssim:.6f}",
                "inference_time": f"{average_time:.6f}",
            }
        )

    return average_psnr, average_ssim, average_time


def read_model_average(metrics_path: Path, label: str) -> tuple[float, float, float] | None:
    if not metrics_path.exists():
        print(f"{label} metrics not found, skipping comparison row: {metrics_path}")
        return None

    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row.get("filename") == "AVERAGE":
            return float(row["psnr"]), float(row["ssim"]), float(row["inference_time"])

    if rows:
        return (
            sum(float(row["psnr"]) for row in rows) / len(rows),
            sum(float(row["ssim"]) for row in rows) / len(rows),
            sum(float(row["inference_time"]) for row in rows) / len(rows),
        )
    return None


def write_comparison(
    path: Path,
    traditional_summary: list[dict[str, str]],
    model_summaries: list[tuple[str, tuple[float, float, float] | None]],
) -> None:
    rows = list(traditional_summary)
    for method, average in model_summaries:
        if average is None:
            continue
        rows.append(
            {
                "method": method,
                "average_psnr": f"{average[0]:.6f}",
                "average_ssim": f"{average[1]:.6f}",
                "average_inference_time": f"{average[2]:.6f}",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "average_psnr", "average_ssim", "average_inference_time"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.data_root)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    high_tensors = {filename: bgr_to_metric_tensor(read_bgr(high_path), args.image_size) for filename, _, high_path in pairs}
    comparison_rows: list[dict[str, str]] = []

    for method, enhance in METHODS.items():
        method_output_dir = args.output_root / method
        rows: list[dict[str, str]] = []

        for filename, low_path, _ in pairs:
            low_bgr = read_bgr(low_path)
            start_time = time.perf_counter()
            enhanced_bgr = enhance(low_bgr)
            inference_time = time.perf_counter() - start_time

            save_bgr(enhanced_bgr, method_output_dir / filename)

            enhanced_tensor = bgr_to_metric_tensor(enhanced_bgr, args.image_size)
            target_tensor = high_tensors[filename]
            psnr = calculate_psnr(enhanced_tensor, target_tensor)
            ssim = calculate_ssim(enhanced_tensor, target_tensor)
            rows.append(
                {
                    "filename": filename,
                    "psnr": f"{psnr:.6f}",
                    "ssim": f"{ssim:.6f}",
                    "inference_time": f"{inference_time:.6f}",
                }
            )

        average_psnr, average_ssim, average_time = write_method_metrics(args.results_dir, method, rows)
        comparison_rows.append(
            {
                "method": method,
                "average_psnr": f"{average_psnr:.6f}",
                "average_ssim": f"{average_ssim:.6f}",
                "average_inference_time": f"{average_time:.6f}",
            }
        )
        print(
            f"{method}: Average PSNR={average_psnr:.4f}, "
            f"SSIM={average_ssim:.4f}, time={average_time:.6f}s"
        )

    comparison_path = args.results_dir / "metrics_comparison.csv"
    write_comparison(
        comparison_path,
        comparison_rows,
        [
            ("unet", read_model_average(args.unet_metrics, "U-Net")),
            ("ms_se_unet", read_model_average(args.ms_se_unet_metrics, "MS-SE-U-Net")),
        ],
    )
    print(f"Saved comparison metrics: {comparison_path}")


if __name__ == "__main__":
    main()
