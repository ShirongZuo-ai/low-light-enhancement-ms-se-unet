from __future__ import annotations

import argparse
import base64
import inspect
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import gradio as gr
except ModuleNotFoundError as exc:
    gr = None
    GRADIO_IMPORT_ERROR = exc
else:
    GRADIO_IMPORT_ERROR = None

try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None

from methods import clahe_enhance, gamma_enhance, retinex_enhance
from models import MultiScaleSEUNet, UNet


IMAGE_SIZE = 256
OUTPUT_DIR = Path("results/app_outputs")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_TIME_BADGE = '<span class="soft-badge">推理时间：等待照亮</span>'
INITIAL_STATUS = "选择一张昏暗的照片，让模型帮你找回被光线隐藏的细节。"

METHOD_DESCRIPTIONS = {
    "Gamma": "Gamma：快速调整整体亮度，适合简单低光照场景。",
    "CLAHE": "CLAHE：增强局部对比度，但可能带来局部过强或噪声问题。",
    "Retinex": "Retinex：基于光照与反射分解思想的传统增强方法。",
    "U-Net": "U-Net：基于编码器-解码器结构的深度学习基线模型。",
    "MS-SE-U-Net": (
        "MS-SE-U-Net：本文改进模型，融合多尺度特征提取、SE 通道注意力机制和组合损失函数，"
        "更关注图像结构、细节和整体观感。"
    ),
}

CHECKPOINTS = {
    "U-Net": Path("checkpoints/unet_best.pth"),
    "MS-SE-U-Net": Path("checkpoints/ms_se_unet_best.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the low-light enhancement Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Create a temporary public gradio.live link.")
    parser.add_argument(
        "--auth",
        nargs=2,
        metavar=("USERNAME", "PASSWORD"),
        default=None,
        help="Optional username and password for the Gradio demo.",
    )
    return parser.parse_args()


def load_state_dict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def load_models() -> tuple[dict[str, torch.nn.Module], dict[str, str]]:
    models: dict[str, torch.nn.Module] = {}
    errors: dict[str, str] = {}

    model_specs = {
        "U-Net": (UNet, CHECKPOINTS["U-Net"]),
        "MS-SE-U-Net": (MultiScaleSEUNet, CHECKPOINTS["MS-SE-U-Net"]),
    }
    for name, (model_cls, checkpoint_path) in model_specs.items():
        if not checkpoint_path.exists():
            errors[name] = f"暂时找不到 {checkpoint_path}。请确认 checkpoint 已放在项目对应目录后再试。"
            continue
        try:
            model = model_cls().to(DEVICE)
            load_state_dict(model, checkpoint_path)
            model.eval()
            models[name] = model
        except Exception as exc:
            errors[name] = f"{name} 模型加载失败：{exc}"

    return models, errors


MODELS, MODEL_ERRORS = load_models()


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("请先上传一张低光照图片。")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def enhance_with_model(image_rgb: np.ndarray, method: str) -> np.ndarray:
    if method not in MODELS:
        raise RuntimeError(MODEL_ERRORS.get(method, f"{method} 模型尚未成功加载。"))

    original_height, original_width = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        output = MODELS[method](tensor)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)

    output_rgb = output.squeeze(0).detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    output_uint8 = (output_rgb * 255.0).round().astype(np.uint8)
    return cv2.resize(output_uint8, (original_width, original_height), interpolation=cv2.INTER_CUBIC)


def enhance_image(image_rgb: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    image_rgb = ensure_uint8_rgb(image_rgb)
    start_time = time.perf_counter()

    if method == "Gamma":
        enhanced_rgb = bgr_to_rgb(gamma_enhance(rgb_to_bgr(image_rgb)))
    elif method == "CLAHE":
        enhanced_rgb = bgr_to_rgb(clahe_enhance(rgb_to_bgr(image_rgb)))
    elif method == "Retinex":
        enhanced_rgb = bgr_to_rgb(retinex_enhance(rgb_to_bgr(image_rgb)))
    elif method in {"U-Net", "MS-SE-U-Net"}:
        enhanced_rgb = enhance_with_model(image_rgb, method)
    else:
        raise ValueError(f"不支持的增强方法：{method}")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    inference_time = time.perf_counter() - start_time
    return enhanced_rgb, inference_time


def make_placeholder(width: int, height: int, title: str, subtitle: str = "") -> np.ndarray:
    image = np.full((height, width, 3), (250, 247, 242), dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (width - 18, height - 18), (232, 220, 207), 2, cv2.LINE_AA)
    cv2.rectangle(image, (32, 32), (width - 32, height - 32), (255, 253, 249), -1, cv2.LINE_AA)
    cv2.putText(image, title, (54, height // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (92, 74, 57), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(
            image,
            subtitle,
            (54, height // 2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (123, 106, 91),
            1,
            cv2.LINE_AA,
        )
    return image


def make_histogram(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    if plt is None:
        raise RuntimeError("当前环境缺少 matplotlib，暂时无法生成亮度直方图。")

    before_gray = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY)

    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=130)
    fig.patch.set_facecolor("#FFFDF9")
    ax.set_facecolor("#FFFDF9")
    ax.hist(
        before_gray.ravel(),
        bins=72,
        range=(0, 255),
        histtype="stepfilled",
        color="#8E877F",
        alpha=0.28,
        label="Original",
    )
    ax.hist(
        after_gray.ravel(),
        bins=72,
        range=(0, 255),
        histtype="stepfilled",
        color="#B98B62",
        alpha=0.56,
        label="Enhanced",
    )
    ax.set_title("Brightness Distribution", color="#3E342C", fontsize=13, pad=10)
    ax.set_xlabel("Brightness", color="#7B6A5B", labelpad=7)
    ax.set_ylabel("Pixel Count", color="#7B6A5B", labelpad=7)
    ax.tick_params(colors="#7B6A5B", labelsize=9)
    ax.grid(axis="y", color="#E8DCCF", linewidth=0.75, alpha=0.65)
    ax.legend(
        frameon=True,
        facecolor="#FFFDF9",
        edgecolor="#E8DCCF",
        labelcolor="#3E342C",
        fontsize=9,
    )
    for spine in ax.spines.values():
        spine.set_color("#E8DCCF")
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    histogram = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return histogram


def save_output(image_rgb: np.ndarray, method: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_method = method.lower().replace("-", "_").replace(" ", "_")
    output_path = OUTPUT_DIR / f"{safe_method}_{timestamp}.png"
    image_bgr = rgb_to_bgr(image_rgb)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise IOError(f"增强结果保存失败：{output_path}")
    return output_path


def image_to_data_uri(path: Path) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    target_width = 280
    target_height = max(1, round(height * target_width / width))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_memory_album_html() -> str:
    album_dir, album_label = resolve_album_source()
    if album_dir is None:
        return ""

    paths = [
        path
        for path in sorted(album_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ][:5]
    image_uris = [(path.name, image_to_data_uri(path)) for path in paths]
    image_uris = [(name, uri) for name, uri in image_uris if uri is not None]
    if not image_uris:
        return ""

    rotations = [-7, 4, -2, 7, -5]
    photos = "\n".join(
        (
            f'<figure class="album-photo album-photo-{index}" '
            f'style="--rotate:{rotations[index % len(rotations)]}deg; --delay:{index * 2.8}s">'
            f'<img src="{uri}" alt="示例低光照照片 {name}">'
            f'<figcaption>{album_label}</figcaption>'
            "</figure>"
        )
        for index, (name, uri) in enumerate(image_uris)
    )
    return f"""
    <details class="memory-album" open>
        <summary class="album-toggle"></summary>
        <div class="album-copy">
            <p class="album-eyebrow">被重新照亮的旧时光</p>
            <h2>回忆图册</h2>
            <p>这些图像来自模型增强后的结果，展示了低光照图像被重新照亮后的细节与质感。</p>
        </div>
        <div class="album-stage" aria-label="回忆图册示例照片">
            {photos}
        </div>
    </details>
    """


def resolve_album_source() -> tuple[Path | None, str]:
    candidates = [
        (Path("results/ms_se_unet_original_size"), "MS-SE-U-Net"),
        (Path("results/ms_se_unet"), "MS-SE-U-Net"),
        (Path("results/unet_original_size"), "U-Net Enhanced"),
        (Path("results/unet"), "U-Net Enhanced"),
        (Path("data/LOL/test/low"), "Low-light Input"),
    ]
    for directory, label in candidates:
        if not directory.is_dir():
            continue
        has_images = any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in directory.iterdir()
        )
        if has_images:
            return directory, label
    return None, "Enhanced"


def run_enhancement(image: np.ndarray | None, method: str) -> tuple[object, object, str, str, str, object]:
    try:
        image_rgb = ensure_uint8_rgb(image)
        enhanced_rgb, inference_time = enhance_image(image_rgb, method)
        output_path = save_output(enhanced_rgb, method)
        histogram = make_histogram(image_rgb, enhanced_rgb)
        return (
            image_rgb,
            enhanced_rgb,
            f'<span class="soft-badge">推理时间：{inference_time:.4f} 秒</span>',
            METHOD_DESCRIPTIONS[method],
            f'图像已完成增强，暗部细节和整体亮度得到改善。<br><span class="path-chip">保存位置：{output_path}</span>',
            histogram,
        )
    except Exception as exc:
        description = METHOD_DESCRIPTIONS.get(method, "请选择一种增强方法。")
        message = f"暂时没有完成增强：{exc}"
        return (
            image,
            make_placeholder(720, 430, "Enhanced result", "The illuminated memory will appear here."),
            '<span class="soft-badge">推理时间：未完成</span>',
            description,
            message,
            make_placeholder(900, 360, "Brightness histogram", "Upload and enhance a photo to view the change."),
        )


def clear_outputs() -> tuple[None, object, object, str, str, str, str, object]:
    return (
        None,
        make_placeholder(720, 430, "Original photo", "Your low-light image will appear here."),
        make_placeholder(720, 430, "Enhanced result", "The illuminated memory will appear here."),
        "MS-SE-U-Net",
        METHOD_DESCRIPTIONS["MS-SE-U-Net"],
        INITIAL_TIME_BADGE,
        INITIAL_STATUS,
        make_placeholder(900, 360, "Brightness histogram", "Enhance a photo to compare luminance."),
    )


def collect_examples() -> list[list[str]]:
    low_dir = Path("data/LOL/test/low")
    if not low_dir.is_dir():
        return []
    paths = [
        path
        for path in sorted(low_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return [[str(path), "MS-SE-U-Net"] for path in paths[:3]]


RESULTS_MARKDOWN = """
| 方法 | PSNR | SSIM |
| --- | ---: | ---: |
| Gamma | 12.3661 | 0.6373 |
| CLAHE | 9.1733 | 0.3795 |
| Retinex | 14.4669 | 0.5342 |
| U-Net | 20.3917 | 0.8021 |
| **MS-SE-U-Net** | **20.5354** | **0.8142** |

MS-SE-U-Net 在 PSNR 和 SSIM 上均取得最佳结果，说明多尺度注意力结构和组合损失对增强质量具有积极作用。
"""

ABLATION_MARKDOWN = """
| 实验设置 | PSNR | SSIM |
| --- | ---: | ---: |
| U-Net baseline | 20.3917 | 0.8021 |
| MS-SE-U-Net + L1 | 19.8512 | 0.7801 |
| **MS-SE-U-Net + Combined Loss** | **20.5354** | **0.8142** |
"""

CUSTOM_CSS = """
:root {
    --warm-bg: #FAF7F2;
    --paper: #FFFDF9;
    --paper-strong: #FFFFFF;
    --ink: #3E342C;
    --muted: #7B6A5B;
    --accent: #A67C52;
    --accent-soft: #D8BFA5;
    --border: #E8DCCF;
}

.gradio-container {
    background:
        radial-gradient(circle at 12% 0%, rgba(216, 191, 165, 0.28), transparent 32%),
        radial-gradient(circle at 88% 14%, rgba(185, 139, 98, 0.12), transparent 28%),
        linear-gradient(180deg, #F8F1E8 0%, #FAF7F2 52%, #FFFDF9 100%);
    color: var(--ink);
    font-family: system-ui, "Microsoft YaHei", "PingFang SC", sans-serif;
}

.gradio-container .contain,
.gradio-container main,
.gradio-container .main {
    max-width: 1280px !important;
    margin: 0 auto !important;
}

.gradio-container * {
    letter-spacing: 0 !important;
}

.hero {
    max-width: 1180px;
    margin: 0 auto 14px auto;
    padding: 30px 28px 18px;
    border-bottom: 1px solid rgba(232, 220, 207, 0.75);
}

.hero h1 {
    color: var(--ink);
    font-size: 44px;
    line-height: 1.18;
    font-weight: 750;
    margin: 0 0 12px;
    letter-spacing: 0;
}

.hero .subtitle {
    color: #5F4A39;
    font-size: 18px;
    line-height: 1.7;
    margin: 0;
    font-weight: 520;
}

.hero .en-subtitle {
    color: #856B55;
    font-size: 13px;
    margin-top: 4px;
}

.hero .warm-line {
    color: #6F5845;
    font-size: 15px;
    margin-top: 13px;
}

.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.tag {
    border: 1px solid var(--border);
    border-radius: 999px;
    color: #6F5845;
    background: rgba(255, 253, 249, 0.86);
    padding: 7px 13px;
    font-size: 13px;
    box-shadow: 0 4px 12px rgba(90, 65, 43, 0.06);
}

.soft-card {
    background: rgba(255, 253, 249, 0.94);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 16px 38px rgba(90, 65, 43, 0.09);
    padding: 18px !important;
    overflow: hidden;
}

.soft-card h3,
.section-title {
    color: var(--ink);
    font-size: 19px;
    margin: 0 0 10px;
}

.hint-text {
    color: #5F4A39;
    font-size: 15px;
    line-height: 1.85;
}

.result-note {
    color: #5F4A39;
    line-height: 1.7;
    background: #FBF4EC;
    border: 1px solid #E8DCCF;
    border-radius: 14px;
    padding: 13px 14px;
}

.status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 10px 0 14px;
}

.soft-badge {
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    border-radius: 999px;
    border: 1px solid #E3CFB9;
    background: #F7EBDD;
    color: #5F4A39;
    padding: 7px 13px;
    font-size: 14px;
    font-weight: 650;
}

.method-copy {
    background: #FFF8F0;
    border: 1px solid #E8DCCF;
    border-radius: 14px;
    padding: 14px 15px;
    color: #4E4035;
    line-height: 1.8;
}

.path-chip {
    display: inline-flex;
    margin-top: 10px;
    border: 1px solid #E3CFB9;
    background: #FFFDF9;
    color: #5F4A39;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 13px;
    word-break: break-all;
}

.memory-album {
    display: block;
    background:
        linear-gradient(135deg, rgba(255, 253, 249, 0.96), rgba(248, 241, 232, 0.92));
    border: 1px solid #E8DCCF;
    border-radius: 18px;
    box-shadow: 0 12px 26px rgba(90, 65, 43, 0.08);
    padding: 14px;
    margin-top: 14px;
    overflow: hidden;
}

.memory-album:not([open]) {
    display: block;
}

.memory-album:not([open]) .album-copy,
.memory-album:not([open]) .album-stage {
    display: none;
}

.memory-album:not([open]) .album-toggle::after {
    content: "展开图册";
}

.memory-album[open] .album-toggle::after {
    content: "收起图册";
}

.album-eyebrow {
    margin: 0 0 8px;
    color: #9A704E;
    font-size: 13px;
    font-weight: 700;
}

.album-copy h2 {
    margin: 0 0 10px;
    color: #3E342C;
    font-size: 20px;
}

.album-copy p {
    color: #5F4A39;
    line-height: 1.72;
    margin: 0 0 10px;
    font-size: 13px;
}

.album-toggle {
    width: fit-content;
    list-style: none;
    border: 1px solid #D8BFA5;
    background: #FFFDF9;
    color: #3E342C;
    border-radius: 999px;
    padding: 9px 16px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 8px 18px rgba(90, 65, 43, 0.08);
    margin-bottom: 8px;
}

.album-toggle::-webkit-details-marker {
    display: none;
}

.album-stage {
    position: relative;
    min-height: 210px;
    transition: min-height 0.28s ease;
}

.album-photo {
    position: absolute;
    width: 148px;
    margin: 0;
    padding: 7px 7px 22px;
    border-radius: 12px;
    border: 1px solid #E2D2C1;
    background: #FFFDF9;
    box-shadow: 0 16px 26px rgba(90, 65, 43, 0.16);
    transform: translate(var(--x, 0px), var(--y, 0px)) rotate(var(--rotate));
    transition: transform 0.45s ease, filter 0.45s ease, opacity 0.45s ease;
    animation: albumCycle 14s infinite ease-in-out;
    animation-delay: var(--delay);
}

.album-photo img {
    width: 100%;
    height: 96px;
    object-fit: cover;
    border-radius: 9px;
    border: 1px solid #E8DCCF;
    display: block;
}

.album-photo figcaption {
    position: absolute;
    left: 10px;
    bottom: 5px;
    color: #7B6A5B;
    font-size: 11px;
}

.album-photo-0 { --x: 6px; --y: 22px; }
.album-photo-1 { --x: 126px; --y: 8px; }
.album-photo-2 { --x: 70px; --y: 82px; }
.album-photo-3 { --x: 202px; --y: 62px; }
.album-photo-4 { --x: 154px; --y: 124px; }

@keyframes albumCycle {
    0%, 18% {
        filter: saturate(1.1) brightness(1.05);
        transform: translate(var(--x, 0px), calc(var(--y, 0px) - 10px)) rotate(0deg) scale(1.04);
    }
    24%, 100% {
        filter: saturate(0.96) brightness(0.98);
        transform: translate(var(--x, 0px), var(--y, 0px)) rotate(var(--rotate)) scale(1);
    }
}

button.primary {
    background: linear-gradient(180deg, #B98B62 0%, #A67C52 100%) !important;
    border: 1px solid #9B704A !important;
    color: white !important;
    border-radius: 999px !important;
    box-shadow: 0 10px 22px rgba(166, 124, 82, 0.24) !important;
    min-height: 44px !important;
    font-weight: 720 !important;
}

button.secondary {
    background: #FFFDF9 !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    border-radius: 999px !important;
    min-height: 44px !important;
}

.gradio-container button:hover {
    filter: brightness(0.99);
    transform: translateY(-1px);
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .block-title {
    color: #5F4A39 !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #FFFDF9 !important;
    color: var(--ink) !important;
    border-color: var(--border) !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    background: transparent !important;
    border-color: var(--border) !important;
}

.gradio-container .wrap,
.gradio-container .image-container,
.gradio-container .upload-container,
.gradio-container .preview,
.gradio-container .empty,
.gradio-container .dropzone,
.gradio-container [data-testid="image"],
.gradio-container [data-testid="image"] > div {
    background: #FFF8F0 !important;
    border-color: #E4D4C3 !important;
    color: #6F5845 !important;
}

.upload-panel .image-container,
.upload-panel .upload-container,
.upload-panel [data-testid="image"] {
    border: 1.5px dashed #D8BFA5 !important;
    border-radius: 18px !important;
    background:
        linear-gradient(135deg, rgba(255, 253, 249, 0.88), rgba(248, 241, 232, 0.88)) !important;
}

.gradio-container .image-container svg,
.gradio-container .upload-container svg {
    color: #A67C52 !important;
    opacity: 0.7 !important;
}

.gradio-container .image-container img,
.gradio-container .image-container canvas,
.gradio-container .preview img {
    object-fit: contain !important;
    background: #FFFDF9 !important;
}

.image-card img {
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 20px rgba(90, 65, 43, 0.08);
}

.image-card .image-container,
.histogram-card .image-container {
    border-radius: 16px !important;
    background: #FFF8F0 !important;
    border: 1px solid #E8DCCF !important;
}

#method_selector {
    background: #FBF4EC !important;
    border: 1px solid #E8DCCF !important;
    border-radius: 18px !important;
    padding: 12px !important;
}

#method_selector .wrap,
#method_selector .radio,
#method_selector fieldset,
#method_selector [role="radiogroup"] {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    background: transparent !important;
    border: 0 !important;
}

#method_selector label,
#method_selector .radio label,
#method_selector [role="radio"] {
    position: relative !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 42px !important;
    padding: 9px 16px 9px 16px !important;
    border-radius: 999px !important;
    border: 1.5px solid #D8BFA5 !important;
    background: #FFFDF9 !important;
    color: #3E342C !important;
    box-shadow: 0 6px 14px rgba(90, 65, 43, 0.06) !important;
    font-weight: 720 !important;
    cursor: pointer !important;
    opacity: 1 !important;
}

#method_selector input[type="radio"],
#method_selector label > input[type="radio"],
#method_selector .radio input[type="radio"] {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
    appearance: none !important;
    -webkit-appearance: none !important;
}

#method_selector label::after,
#method_selector [role="radio"]::after,
#method_selector .radio label::after {
    display: none !important;
    content: none !important;
}

#method_selector svg,
#method_selector .icon,
#method_selector .check,
#method_selector .dot,
#method_selector [class*="radio"] svg {
    display: none !important;
}

#method_selector label span,
#method_selector [role="radio"] span,
#method_selector label * {
    color: #3E342C !important;
    font-weight: 720 !important;
}

#method_selector label:has(input:checked),
#method_selector [role="radio"][aria-checked="true"] {
    background: linear-gradient(180deg, #B98B62 0%, #A67C52 100%) !important;
    border-color: #9B704A !important;
    color: #FFFDF7 !important;
    box-shadow: 0 12px 22px rgba(166, 124, 82, 0.26) !important;
}

#method_selector label:has(input:checked)::before,
#method_selector [role="radio"][aria-checked="true"]::before {
    content: "✓";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 17px;
    height: 17px;
    margin-right: 7px;
    border-radius: 999px;
    background: rgba(255, 253, 247, 0.22);
    color: #FFFDF7;
    font-size: 12px;
}

#method_selector label:has(input:checked) span,
#method_selector label:has(input:checked) *,
#method_selector [role="radio"][aria-checked="true"] span,
#method_selector [role="radio"][aria-checked="true"] * {
    color: #FFFDF7 !important;
}

.examples-soft {
    margin-top: 12px;
}

.examples-soft table,
.examples-soft .table-wrap,
.examples-soft .dataset {
    background: transparent !important;
    border: 0 !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: none !important;
}

.examples-soft th {
    background: #F6ECE1 !important;
    color: #5F4A39 !important;
    border: 0 !important;
}

.examples-soft td {
    background: #FFFDF9 !important;
    color: #5F4A39 !important;
    border: 0 !important;
    padding: 8px !important;
}

.examples-soft img {
    border-radius: 12px !important;
    border: 1px solid #E8DCCF !important;
    box-shadow: 0 8px 16px rgba(90, 65, 43, 0.10) !important;
}

.table-wrap table {
    width: 100%;
    border-collapse: collapse;
    background: #FFFDF9;
}

.table-wrap th,
.table-wrap td {
    border-bottom: 1px solid var(--border);
    padding: 10px 12px;
    color: var(--ink);
}

.table-wrap th {
    background: #F6ECE1;
    color: #5F4A39;
}

.table-wrap strong {
    color: #8A5E38;
}

footer,
.footer,
.gradio-container > footer,
a[href*="gradio.app"],
button[aria-label="Use via API"] {
    opacity: 0.08 !important;
    max-height: 14px !important;
    overflow: hidden !important;
}

@media (max-width: 780px) {
    .hero {
        padding: 22px 14px 12px;
    }
    .hero h1 {
        font-size: 32px;
    }
    .soft-card {
        padding: 14px !important;
    }
    .memory-album {
        grid-template-columns: 1fr;
    }
    .album-stage {
        min-height: 260px;
    }
    .album-photo {
        width: 180px;
    }
    .album-photo-3,
    .album-photo-4 {
        display: none;
    }
}
"""


def build_demo() -> gr.Blocks:
    if gr is None:
        raise RuntimeError("当前环境缺少 gradio，暂时无法启动网页演示。")

    block_kwargs = {"title": "把暗处的记忆重新照亮"}
    launch_accepts_css = "css" in inspect.signature(gr.Blocks.launch).parameters
    if "css" in inspect.signature(gr.Blocks).parameters and not launch_accepts_css:
        block_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**block_kwargs) as demo:
        gr.HTML(
            """
            <section class="hero">
                <h1>把暗处的记忆重新照亮</h1>
                <p class="subtitle">基于深度学习的低光照图像增强与老照片修复雏形系统</p>
                <p class="en-subtitle">Low-Light Image Enhancement as a Prototype for Warm Photo Restoration</p>
                <p class="warm-line">选择一张昏暗的照片，让模型帮你找回被光线隐藏的细节。</p>
                <div class="tags">
                    <span class="tag">低光照增强</span>
                    <span class="tag">记忆修复雏形</span>
                    <span class="tag">MS-SE-U-Net</span>
                </div>
            </section>
            """
        )

        gr.Markdown(
            """
            本系统面向低光照图像增强任务，通过传统图像处理方法、U-Net 基线模型以及本文改进的
            MS-SE-U-Net 模型，对昏暗图像进行亮度、结构和细节恢复。它不仅是一个图像增强实验系统，
            也可以作为未来老照片修复、家庭影像修复和视觉记忆重建工具的早期雏形。
            """,
            elem_classes=["soft-card", "hint-text"],
        )
        with gr.Row():
            with gr.Column(scale=5, elem_classes=["soft-card", "upload-panel"]):
                gr.Markdown("### 上传与选择")
                input_image = gr.Image(label="上传低光照图片", type="numpy", height=280)
                method = gr.Radio(
                    choices=["Gamma", "CLAHE", "Retinex", "U-Net", "MS-SE-U-Net"],
                    value="MS-SE-U-Net",
                    label="增强方法",
                    elem_classes=["radio-group"],
                    elem_id="method_selector",
                )
                with gr.Row():
                    enhance_button = gr.Button("一键照亮", variant="primary", elem_classes=["primary"])
                    clear_button = gr.Button("清空重来", variant="secondary", elem_classes=["secondary"])

                examples = collect_examples()
                if examples:
                    with gr.Column(elem_classes=["examples-soft"]):
                        gr.Markdown("#### 示例回忆照片")
                        gr.Examples(examples=examples, inputs=[input_image, method], label="")

            with gr.Column(scale=4, elem_classes=["soft-card"]):
                gr.Markdown("### 当前方法说明")
                method_description = gr.Markdown(METHOD_DESCRIPTIONS["MS-SE-U-Net"], elem_classes=["method-copy"])
                gr.HTML(
                    f"""
                    <div class="status-row">
                        <span class="soft-badge">运行设备：{DEVICE}</span>
                    </div>
                    """
                )
                inference_time = gr.Markdown(INITIAL_TIME_BADGE)
                status_message = gr.Markdown(INITIAL_STATUS, elem_classes=["result-note"])
                album_html = build_memory_album_html()
                if album_html:
                    gr.HTML(album_html)

        with gr.Row():
            with gr.Column(elem_classes=["soft-card", "image-card"]):
                gr.Markdown("### 原始低光照图像")
                original_output = gr.Image(
                    label="原始照片将在这里显示",
                    type="numpy",
                    height=340,
                    value=make_placeholder(720, 430, "Original photo", "Your low-light image will appear here."),
                )
            with gr.Column(elem_classes=["soft-card", "image-card"]):
                gr.Markdown("### 增强结果图像")
                enhanced_output = gr.Image(
                    label="增强结果将在这里显示",
                    type="numpy",
                    height=340,
                    value=make_placeholder(720, 430, "Enhanced result", "The illuminated memory will appear here."),
                )

        with gr.Column(elem_classes=["soft-card", "histogram-card"]):
            gr.Markdown("### 亮度分布变化")
            gr.Markdown("增强后，图像整体亮度分布向更清晰、可见的区域移动。")
            histogram_output = gr.Image(
                label="亮度直方图",
                type="numpy",
                height=310,
                value=make_placeholder(900, 360, "Brightness histogram", "Enhance a photo to compare luminance."),
            )

        with gr.Row():
            with gr.Column(elem_classes=["soft-card", "table-wrap"]):
                gr.Markdown("### 实验结果摘要")
                gr.Markdown(RESULTS_MARKDOWN)
            with gr.Column(elem_classes=["soft-card", "table-wrap"]):
                gr.Markdown("### 消融实验摘要")
                gr.Markdown(ABLATION_MARKDOWN)

        method.change(fn=lambda selected: METHOD_DESCRIPTIONS[selected], inputs=method, outputs=method_description)
        enhance_button.click(
            fn=run_enhancement,
            inputs=[input_image, method],
            outputs=[
                original_output,
                enhanced_output,
                inference_time,
                method_description,
                status_message,
                histogram_output,
            ],
        )
        clear_button.click(
            fn=clear_outputs,
            inputs=None,
            outputs=[
                input_image,
                original_output,
                enhanced_output,
                method,
                method_description,
                inference_time,
                status_message,
                histogram_output,
            ],
        )

    return demo


def launch_demo(
    demo: gr.Blocks,
    share: bool = False,
    auth: tuple[str, str] | None = None,
) -> None:
    launch_kwargs = {"server_name": "0.0.0.0", "share": share}
    if auth is not None:
        launch_kwargs["auth"] = auth
    if "css" in inspect.signature(demo.launch).parameters:
        launch_kwargs["css"] = CUSTOM_CSS

    last_error: OSError | None = None
    for port in range(7860, 7870):
        try:
            demo.launch(server_port=port, **launch_kwargs)
            return
        except OSError as exc:
            last_error = exc
            if "Cannot find empty port" not in str(exc) and "port" not in str(exc).lower():
                raise
            print(f"Port {port} is not available, trying {port + 1}...")

    raise RuntimeError("7860-7869 端口都不可用，暂时无法启动 Gradio 服务。") from last_error


def main() -> None:
    args = parse_args()
    missing_dependencies = []
    if GRADIO_IMPORT_ERROR is not None:
        missing_dependencies.append("gradio")
    if MATPLOTLIB_IMPORT_ERROR is not None:
        missing_dependencies.append("matplotlib")
    if missing_dependencies:
        missing = ", ".join(missing_dependencies)
        raise SystemExit(f"缺少依赖：{missing}。请先运行 `pip install -r requirements.txt` 后再启动 app.py。")

    print(f"Using device: {DEVICE}")
    if MODEL_ERRORS:
        for name, message in MODEL_ERRORS.items():
            print(f"{name}: {message}")
    else:
        print("Loaded checkpoints: checkpoints/unet_best.pth, checkpoints/ms_se_unet_best.pth")

    demo = build_demo()
    auth = tuple(args.auth) if args.auth is not None else None
    launch_demo(demo, share=args.share, auth=auth)


if __name__ == "__main__":
    main()
