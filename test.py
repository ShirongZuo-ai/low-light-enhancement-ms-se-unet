from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import MultiScaleSEUNet, UNet
from utils.dataset import LOLDataset
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test enhancement models on LOL.")
    parser.add_argument("--data-root", type=str, default="data/LOL", help="Root directory of the LOL dataset.")
    parser.add_argument("--model", choices=("unet", "ms_se_unet"), default="unet", help="Model architecture to test.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize size used by LOLDataset.")
    parser.add_argument("--batch-size", type=int, default=1, help="Testing batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--run-name", type=str, default=None, help="Name used for default output and metrics paths.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for enhanced images.")
    parser.add_argument(
        "--original-size-output-dir",
        type=str,
        default=None,
        help="Directory for enhanced images resized back to original low image size.",
    )
    parser.add_argument("--metrics-path", type=str, default=None, help="CSV metrics path.")
    parser.add_argument(
        "--save-original-size",
        action="store_true",
        help="Also save enhanced images resized back to each original low image size.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    requested = requested_device.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested device 'cuda' is not available. Falling back to CPU.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(requested if requested.startswith("cuda") else "cuda" if torch.cuda.is_available() else "cpu")


def build_model(name: str) -> torch.nn.Module:
    if name == "unet":
        return UNet()
    if name == "ms_se_unet":
        return MultiScaleSEUNet()
    raise ValueError(f"Unsupported model: {name}")


def load_model(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    model = build_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_rgb_tensor(image: torch.Tensor, output_path: Path, size: tuple[int, int] | None = None) -> None:
    image_np = image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    image_uint8 = (image_np * 255.0).round().astype(np.uint8)
    if size is not None:
        image_uint8 = cv2.resize(image_uint8, size, interpolation=cv2.INTER_CUBIC)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise IOError(f"Failed to write image: {output_path}")


def get_original_low_size(data_root: Path, filename: str) -> tuple[int, int]:
    image_path = data_root / "test" / "low" / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read original low image size with OpenCV: {image_path}")
    height, width = image.shape[:2]
    return width, height


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    run_name = args.run_name or args.model

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path("checkpoints") / f"{run_name}_best.pth"
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    original_size_output_dir = (
        Path(args.original_size_output_dir)
        if args.original_size_output_dir
        else Path("results") / f"{run_name}_original_size"
    )
    if args.save_original_size:
        original_size_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else Path("results") / f"metrics_{run_name}.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    test_dataset = LOLDataset(
        root=data_root,
        split="test",
        image_size=(args.image_size, args.image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(args.model, checkpoint_path, device)
    rows: list[dict[str, str]] = []
    warmup_done = False

    with torch.no_grad():
        for low_images, high_images, filenames in test_loader:
            low_images = low_images.to(device, non_blocking=True)
            if not warmup_done:
                _ = model(low_images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                warmup_done = True

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            outputs = model(low_images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_inference_time = time.perf_counter() - start_time
            per_image_inference_time = batch_inference_time / low_images.size(0)
            outputs = outputs.cpu()

            for output, target, filename in zip(outputs, high_images, filenames):
                save_rgb_tensor(output, output_dir / filename)
                if args.save_original_size:
                    original_size = get_original_low_size(data_root, filename)
                    save_rgb_tensor(output, original_size_output_dir / filename, size=original_size)
                psnr = calculate_psnr(output, target)
                ssim = calculate_ssim(output, target)
                rows.append(
                    {
                        "filename": filename,
                        "psnr": f"{psnr:.6f}",
                        "ssim": f"{ssim:.6f}",
                        "inference_time": f"{per_image_inference_time:.6f}",
                    }
                )

    mean_psnr = sum(float(row["psnr"]) for row in rows) / len(rows)
    mean_ssim = sum(float(row["ssim"]) for row in rows) / len(rows)
    mean_inference_time = sum(float(row["inference_time"]) for row in rows) / len(rows)

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "inference_time"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "filename": "AVERAGE",
                "psnr": f"{mean_psnr:.6f}",
                "ssim": f"{mean_ssim:.6f}",
                "inference_time": f"{mean_inference_time:.6f}",
            }
        )

    print(f"Saved enhanced images: {output_dir}")
    if args.save_original_size:
        print(f"Saved original-size enhanced images: {original_size_output_dir}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Average PSNR={mean_psnr:.4f}, SSIM={mean_ssim:.4f}, time={mean_inference_time:.6f}s")


if __name__ == "__main__":
    main()
