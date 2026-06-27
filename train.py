from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import MultiScaleSEUNet, UNet
from utils.dataset import LOLDataset
from utils.losses import build_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train enhancement models on LOL.")
    parser.add_argument("--data-root", type=str, default="data/LOL", help="Root directory of the LOL dataset.")
    parser.add_argument("--model", choices=("unet", "ms_se_unet"), default="unet", help="Model architecture to train.")
    parser.add_argument("--loss", choices=("l1", "combined"), default="l1", help="Training loss to use.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu.")
    parser.add_argument("--image-size", type=int, default=256, help="Square resize size used by LOLDataset.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--log-path", type=str, default=None, help="CSV training log path.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training from.")
    parser.add_argument("--run-name", type=str, default=None, help="Name used for checkpoint and log filenames.")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    requested = requested_device.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Requested device 'cuda' is not available. Falling back to CPU.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(requested if requested.startswith("cuda") else "cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    best_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "best_loss": best_loss,
        },
        path,
    )


def load_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float | None]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = checkpoint.get("best_loss")
        return start_epoch, float(best_loss) if best_loss is not None else None

    model.load_state_dict(checkpoint)
    return 1, None


def read_checkpoint_loss(path: Path, device: torch.device) -> float | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        return None
    if "best_loss" in checkpoint:
        return float(checkpoint["best_loss"])
    if "train_loss" in checkpoint:
        return float(checkpoint["train_loss"])
    return None


def build_model(name: str) -> torch.nn.Module:
    if name == "unet":
        return UNet()
    if name == "ms_se_unet":
        return MultiScaleSEUNet()
    raise ValueError(f"Unsupported model: {name}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    run_name = args.run_name or args.model

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else Path("results") / f"train_log_{run_name}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = LOLDataset(
        root=args.data_root,
        split="train",
        image_size=(args.image_size, args.image_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model).to(device)
    criterion = build_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    latest_path = checkpoint_dir / f"{run_name}_latest.pth"
    best_path = checkpoint_dir / f"{run_name}_best.pth"
    start_epoch = 1

    if args.resume:
        start_epoch, resumed_best_loss = load_resume_checkpoint(Path(args.resume), model, optimizer, device)
        best_loss = resumed_best_loss if resumed_best_loss is not None else best_loss
        if resumed_best_loss is None:
            saved_best_loss = read_checkpoint_loss(best_path, device)
            if saved_best_loss is not None:
                best_loss = saved_best_loss
        print(f"Resumed training from {args.resume} at epoch {start_epoch}")

    log_mode = "a" if args.resume and log_path.exists() else "w"
    write_header = log_mode == "w" or log_path.stat().st_size == 0

    with log_path.open(log_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "model", "loss", "train_loss", "lr"])
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for low_images, high_images, _ in train_loader:
                low_images = low_images.to(device, non_blocking=True)
                high_images = high_images.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(low_images)
                loss = criterion(outputs, high_images)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * low_images.size(0)

            train_loss = running_loss / len(train_dataset)
            writer.writerow(
                {
                    "epoch": epoch,
                    "model": args.model,
                    "loss": args.loss,
                    "train_loss": f"{train_loss:.6f}",
                    "lr": args.lr,
                }
            )
            f.flush()
            os.fsync(f.fileno())

            if train_loss < best_loss:
                best_loss = train_loss
                save_checkpoint(best_path, model, optimizer, epoch, train_loss, best_loss)

            save_checkpoint(latest_path, model, optimizer, epoch, train_loss, best_loss)

            print(f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.6f} best_loss={best_loss:.6f}")

    print(f"Saved latest checkpoint: {latest_path}")
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
