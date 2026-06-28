"""
app_v3.py — Low-Light Image Enhancement · Premium Product Landing Page
=======================================================================
Design: Apple editorial-minimalist, warm ivory + soft gray.
Structure: Hero → Overview → AI Restoration Workspace → Metrics → Footer.

Usage:
  Local test:           python app_v3.py
  LAN access:           python app_v3.py --server-name 0.0.0.0
  Public share + auth:  python app_v3.py --share --auth demo 123456
  Custom port:          python app_v3.py --server-port 7861
"""

from __future__ import annotations

import argparse
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

_PROJECT_ROOT = Path(__file__).resolve().parent
_ASSETS_DIR = _PROJECT_ROOT / "assets"
try:
    if gr is not None and hasattr(gr, "set_static_paths"):
        gr.set_static_paths(paths=[str(_ASSETS_DIR)])
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 256
OUTPUT_DIR = Path("results/app_outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERO_VIDEO = _ASSETS_DIR / "hero_memory_light.mp4"
HERO_VIDEO_EXISTS = HERO_VIDEO.is_file()

METHOD_DESCRIPTIONS = {
    "Gamma": (
        "**Gamma Correction** applies a global nonlinear brightness transform. "
        "Fast and straightforward for mild underexposure. Limited structural recovery."
    ),
    "CLAHE": (
        "**CLAHE** stretches local contrast in tiles. Effective for revealing "
        "dark-region details but may amplify noise or create blocking artifacts."
    ),
    "Retinex": (
        "**Retinex** decomposes an image into illumination and reflectance layers, "
        "compensating for the lighting component. Good color fidelity for natural scenes."
    ),
    "U-Net": (
        "**U-Net** is the encoder-decoder baseline with skip connections preserving "
        "spatial detail. Already delivers meaningful brightness recovery."
    ),
    "MS-SE-U-Net": (
        "**MS-SE-U-Net** combines multi-scale feature extraction, SE channel attention, "
        "and a combined restoration loss to improve brightness, structure, "
        "color consistency, and visual naturalness."
    ),
}

CHECKPOINTS = {
    "U-Net": Path("checkpoints/unet_best.pth"),
    "MS-SE-U-Net": Path("checkpoints/ms_se_unet_best.pth"),
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = r"""
/* ==================================================================
   TOKENS
   ================================================================== */
:root {
    --c-ink: #2F2924;
    --c-body: #3A3028;
    --c-secondary: #6F5E4F;
    --c-label: #8B6F4E;
    --c-accent: #B8733B;
    --c-accent-soft: #D9BFA2;
    --c-surface: rgba(255,255,255,0.74);
    --c-border: rgba(130,105,80,0.14);
    --c-border-soft: rgba(130,105,80,0.08);
    --c-badge-bg: #FBF4EC;
    --c-shadow: 0 1px 3px rgba(47,41,36,0.04), 0 4px 14px rgba(47,41,36,0.03);
    --c-shadow-lg: 0 2px 8px rgba(47,41,36,0.05), 0 8px 32px rgba(47,41,36,0.04);
    --c-shadow-app: 0 24px 80px rgba(70,55,38,0.10);
    --c-radius: 18px;
    --c-radius-sm: 10px;
    --c-radius-pill: 999px;
    --c-font: "Inter","SF Pro Display","Helvetica Neue",system-ui,"PingFang SC","Microsoft YaHei",sans-serif;
    --c-font-display: "Georgia","Times New Roman","STSong","Noto Serif CJK SC","Songti SC","SimSun","PingFang SC",serif;
}
html { scroll-behavior: smooth; }

/* ==================================================================
   GLOBAL RESET
   ================================================================== */
.gradio-container {
    font-family: var(--c-font) !important;
    background:
        radial-gradient(circle at 22% 18%, rgba(185,145,95,0.08), transparent 36%),
        linear-gradient(180deg, #f7f1e9 0%, #fbf8f3 45%, #f4eee6 100%) !important;
    color: var(--c-ink) !important;
    -webkit-font-smoothing: antialiased;
    margin: 0 !important; padding: 0 !important; max-width: 100% !important;
}
.gradio-container .contain,.gradio-container main,.gradio-container .main,.gradio-container .app {
    max-width: 100% !important; margin: 0 !important; padding: 0 !important;
}
.gradio-container * { box-sizing: border-box; }
.gradio-container span,.gradio-container p,.gradio-container label,
.gradio-container h1,.gradio-container h2,.gradio-container h3 { letter-spacing: 0 !important; }
.gradio-container .block,.gradio-container .form,.gradio-container .panel,.gradio-container .wrap {
    background: transparent !important; border: none !important; box-shadow: none !important; margin: 0 !important;
}
.gradio-container .gap,.gradio-container .gr-box { gap: 0 !important; }

/* ==================================================================
   HERO
   ================================================================== */
#hero-section-v3 {
    position: relative; width: 100vw; min-height: 100vh;
    margin: 0 !important; padding: 0 !important; overflow: hidden;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 900'%3E%3Cdefs%3E%3CradialGradient id='g' cx='28%25' cy='38%25' r='72%25'%3E%3Cstop offset='0%25' stop-color='%23F5EEE5'/%3E%3Cstop offset='35%25' stop-color='%23F9F6F1'/%3E%3Cstop offset='72%25' stop-color='%23FAF7F3'/%3E%3Cstop offset='100%25' stop-color='%23FBF8F4'/%3E%3C/radialGradient%3E%3C/defs%3E%3Crect width='1440' height='900' fill='url(%23g)'/%3E%3C/svg%3E");
    background-size: cover; background-position: center;
}
.hero-video-bg {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    object-fit: cover; z-index: 0;
}
#hero-section-v3::after {
    content: ""; position: absolute; bottom: 0; left: 0;
    width: 100%; height: 240px; z-index: 2; pointer-events: none;
    background: linear-gradient(to bottom, transparent, rgba(247,241,233,0.78) 42%, #f7f1e9 100%);
}
.hero-overlay {
    position: absolute; inset: 0; z-index: 1; pointer-events: none;
    background:
        linear-gradient(90deg, rgba(35,28,22,0.58) 0%, rgba(35,28,22,0.28) 40%, rgba(35,28,22,0.05) 68%, rgba(35,28,22,0.02) 100%),
        linear-gradient(0deg, rgba(247,241,233,0) 0%, rgba(247,241,233,0) 74%, rgba(247,241,233,0.20) 100%);
}
.hero-content {
    position: relative; z-index: 3; display: flex; align-items: center; justify-content: flex-start;
    max-width: 1340px; margin: 0 auto; min-height: 100vh; padding: 80px 64px 120px;
}
.hero-text { flex: 0 1 56%; max-width: 700px; }
.hero-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 0.18em !important; color: rgba(254,253,251,0.72); text-transform: uppercase; margin: 0 0 28px; }
.hero-headline { font-family: var(--c-font-display); font-size: clamp(46px,6.6vw,92px); font-weight: 400; line-height: 1.06; color: #FEFDFB; margin: 0 0 18px; letter-spacing: -0.015em !important; }
.hero-headline em { font-style: italic; color: #E8D3BC; }
.hero-sub { font-size: 18px; font-weight: 500; line-height: 1.6; color: rgba(254,253,251,0.78); margin: 0 0 10px; }
.hero-desc { font-size: 14px; line-height: 1.7; color: rgba(254,253,251,0.58); margin: 0 0 32px; max-width: 50ch; }
.hero-tags-row { display: flex; flex-wrap: wrap; gap: 9px; margin: 0 0 36px; }
.hero-tag { display: inline-flex; align-items: center; height: 30px; padding: 0 14px; border: 1px solid rgba(254,253,251,0.14); border-radius: var(--c-radius-pill); background: rgba(254,253,251,0.06); color: rgba(254,253,251,0.55); font-size: 12px; font-weight: 600; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); transition: transform 0.2s ease,background 0.2s ease,color 0.2s ease; }
.hero-tag:hover { transform: translateY(-1px); background: rgba(254,253,251,0.14); color: rgba(254,253,251,0.76); }
.hero-ctas { display: flex; gap: 14px; flex-wrap: wrap; }
.hero-cta { display: inline-flex; align-items: center; justify-content: center; min-height: 46px; padding: 0 28px; border-radius: var(--c-radius-pill); font-size: 15px; font-weight: 680; text-decoration: none !important; cursor: pointer; transition: transform 0.18s ease,box-shadow 0.18s ease; }
.hero-cta:hover { transform: translateY(-1px); }
.hero-cta.primary { background: #FEFDFB; color: #2F2924; border: 1px solid rgba(254,253,251,0.3); box-shadow: 0 4px 18px rgba(35,28,22,0.14); }
.hero-cta.primary:hover { box-shadow: 0 6px 24px rgba(35,28,22,0.20); }
.hero-cta.secondary { background: rgba(254,253,251,0.10); color: rgba(254,253,251,0.82); border: 1px solid rgba(254,253,251,0.18); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
.hero-cta.secondary:hover { background: rgba(254,253,251,0.16); color: #FEFDFB; }

/* ==================================================================
   OVERVIEW
   ================================================================== */
#overview-section-v3 {
    position: relative; z-index: 5; max-width: 1340px;
    margin: -100px auto 0; padding: 0 64px;
}
.overview-quote { font-family: var(--c-font-display); font-size: 22px; font-style: italic; color: var(--c-secondary); text-align: center; margin: 0 0 36px; line-height: 1.5; }
.overview-cards { display: grid; grid-template-columns: repeat(3,1fr); gap: 20px; margin-bottom: 56px; }
.overview-card { background: rgba(255,255,255,0.74); border: 1px solid var(--c-border-soft); border-radius: var(--c-radius); padding: 28px 24px; box-shadow: var(--c-shadow); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); }
.overview-icon { width: 40px; height: 40px; border-radius: 50%; margin-bottom: 14px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; }
.ov-icon-t { background: rgba(184,115,59,0.10); color: var(--c-accent); }
.ov-icon-d { background: rgba(184,115,59,0.14); color: var(--c-accent); }
.ov-icon-e { background: rgba(90,80,68,0.08); color: var(--c-secondary); }
.overview-card h3 { font-size: 15px; font-weight: 700; color: var(--c-ink); margin: 0 0 8px; }
.overview-card p { font-size: 13px; color: var(--c-secondary); line-height: 1.6; margin: 0; }

/* ==================================================================
   SEPARATORS & SECTION HEADERS
   ================================================================== */
.ws-sep { width: 100%; max-width: 1340px; margin: 0 auto; height: 48px; background: linear-gradient(to bottom, rgba(247,241,233,0) 0%, rgba(247,241,233,0.18) 40%, rgba(247,241,233,0) 100%); pointer-events: none; }
.ws-header { text-align: center; max-width: 640px; margin: 0 auto 14px; }
.ws-eyebrow { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.16em !important; color: var(--c-label); margin: 0 0 8px; }
.ws-title { font-family: var(--c-font-display); font-size: 34px; font-weight: 400; color: var(--c-ink); margin: 0 0 8px; line-height: 1.2; letter-spacing: -0.01em !important; }
.ws-desc { font-size: 14px; color: var(--c-secondary); line-height: 1.68; margin: 0; }

/* ==================================================================
   UNIFIED AI RESTORATION WORKSPACE
   ================================================================== */
#studio-section { max-width: 1340px; margin: 0 auto; padding: 40px 64px 0; }
.workspace-shell {
    background: var(--c-surface); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--c-border); border-radius: var(--c-radius);
    box-shadow: var(--c-shadow-app); padding: 0; overflow: hidden;
}
.workspace-grid {
    display: grid; grid-template-columns: 340px 1fr; gap: 0;
    align-items: start;
}

/* ── Left: Control Panel ── */
.control-panel {
    background: rgba(255,253,251,0.60); border-right: 1px solid var(--c-border-soft);
    padding: 28px 24px; display: flex; flex-direction: column; gap: 20px;
}
.cp-title { font-size: 14px; font-weight: 700; color: var(--c-ink); margin: 0; }

/* Upload zone */
.upload-zone {
    border: 1.5px dashed rgba(143,110,76,0.28); border-radius: var(--c-radius);
    background: rgba(255,255,255,0.64); overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    min-height: 240px; max-height: 300px;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.upload-zone:hover { border-color: rgba(184,115,59,0.26); background: rgba(255,255,255,0.78); }

/* Override Gradio upload inside control panel */
#upload-panel-cp .image-container,
#upload-panel-cp [data-testid="image"] {
    border: none !important; border-radius: 0 !important;
    background: transparent !important; min-height: 240px !important; max-height: 300px !important;
    box-shadow: none !important;
}
#upload-panel-cp .image-container:hover,
#upload-panel-cp [data-testid="image"]:hover { border: none !important; }
#upload-panel-cp .image-container img,
#upload-panel-cp [data-testid="image"] img { object-fit: contain !important; padding: 8px !important; }

/* Method segmented control */
.method-segment {
    display: flex; flex-wrap: wrap; gap: 5px;
    background: rgba(255,253,251,0.60); border: 1px solid var(--c-border-soft);
    border-radius: var(--c-radius-pill); padding: 4px;
}

/* Kill all Gradio radio defaults strongly */
#method-selector {
    background: transparent !important; border: 0 !important; padding: 0 !important;
}
#method-selector .wrap,#method-selector fieldset,#method-selector [role="radiogroup"] {
    display: flex !important; flex-wrap: wrap !important; gap: 4px !important;
    background: transparent !important; border: 0 !important;
}
#method-selector label,#method-selector [role="radio"] {
    position: relative !important; display: inline-flex !important;
    align-items: center !important; justify-content: center !important;
    min-height: 34px !important; padding: 6px 13px !important;
    border-radius: var(--c-radius-pill) !important; border: 1.5px solid transparent !important;
    background: transparent !important; color: var(--c-secondary) !important;
    font-weight: 600 !important; font-size: 12px !important;
    cursor: pointer !important; opacity: 1 !important; box-shadow: none !important;
    transition: all 0.16s ease !important;
}
#method-selector label:hover { background: rgba(255,255,255,0.70) !important; color: var(--c-ink) !important; }
#method-selector input[type="radio"] {
    position: absolute !important; width: 1px !important; height: 1px !important;
    margin: 0 !important; padding: 0 !important; opacity: 0 !important;
    pointer-events: none !important; appearance: none !important; -webkit-appearance: none !important;
}
#method-selector label::after,#method-selector [role="radio"]::after { display: none !important; content: none !important; }
#method-selector svg,#method-selector .icon,#method-selector .check,#method-selector .dot { display: none !important; }
#method-selector label span,#method-selector label * { color: inherit !important; font-weight: 600 !important; }
#method-selector label:has(input:checked),
#method-selector [role="radio"][aria-checked="true"] {
    background: linear-gradient(175deg, #C48657 0%, #B8733B 100%) !important;
    border-color: #A56535 !important; color: #FFFDF8 !important;
    box-shadow: 0 2px 8px rgba(184,115,59,0.16) !important;
}
#method-selector label:has(input:checked) span,
#method-selector label:has(input:checked) *,
#method-selector [role="radio"][aria-checked="true"] * { color: #FFFDF8 !important; font-weight: 700 !important; }

/* Hide "单选框" label text from Gradio */
#method-selector + label,
label:has(+ #method-selector),
#method-selector ~ label,
[for*="method"]:not([for*="method-selector"]),
.gradio-container label:has(~ #method-selector) {
    display: none !important; opacity: 0 !important; height: 0 !important; overflow: hidden !important;
}
/* Broad hiding of radio labels */
label:has(+ [id*="method"]) { display: none !important; height: 0 !important; overflow: hidden !important; }

/* Action buttons inside control panel */
.action-row { display: flex; flex-direction: column; gap: 10px; }
.v3-btn-primary {
    width: 100% !important; background: #2F2924 !important; border: 1px solid #3D352D !important;
    color: #FEFDFB !important; border-radius: var(--c-radius-pill) !important;
    min-height: 46px !important; font-weight: 650 !important; font-size: 14px !important;
    letter-spacing: 0.01em !important; transition: all 0.18s ease !important;
    box-shadow: 0 4px 14px rgba(47,41,36,0.10) !important;
}
.v3-btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(47,41,36,0.16) !important; background: #3A332D !important; }
.v3-btn-primary:active { transform: translateY(0) scale(0.985); }
.v3-btn-secondary {
    width: 100% !important; background: rgba(255,255,255,0.78) !important;
    border: 1px solid var(--c-border) !important; color: var(--c-ink) !important;
    border-radius: var(--c-radius-pill) !important; min-height: 42px !important;
    font-weight: 600 !important; font-size: 13px !important;
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    transition: all 0.18s ease !important;
}
.v3-btn-secondary:hover { background: rgba(255,255,255,0.92) !important; transform: translateY(-1px); box-shadow: var(--c-shadow) !important; }

/* ── Right: Preview + Method Panel ── */
.main-panel { padding: 28px 28px; display: flex; flex-direction: column; gap: 24px; overflow-y: auto; }

/* Live Preview */
.preview-area { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.preview-card {
    background: rgba(255,253,251,0.55); border: 1px solid var(--c-border-soft);
    border-radius: var(--c-radius); overflow: hidden;
    box-shadow: var(--c-shadow);
}
.preview-card.after-card {
    box-shadow: 0 0 0 1px rgba(184,115,59,0.10), var(--c-shadow-lg);
}
.pv-label { font-size: 12px; font-weight: 700; color: var(--c-secondary); padding: 14px 16px 0; }
.pv-label-after { font-size: 12px; font-weight: 700; color: var(--c-accent); padding: 14px 16px 0; }
.preview-title { font-size: 13px; font-weight: 700; color: var(--c-ink); margin: 0 0 12px; }

/* Override Gradio image inside preview cards */
.preview-card .image-container,
.preview-card [data-testid="image"] {
    border: none !important; background: transparent !important;
    box-shadow: none !important; border-radius: 0 !important; overflow: hidden !important;
}
.preview-card .image-container img,
.preview-card [data-testid="image"] img { object-fit: contain !important; padding: 8px 16px 16px !important; }

/* Method info inside main panel */
.mp-method-title { font-size: 13px; font-weight: 700; color: var(--c-label); margin: 0 0 8px; text-transform: uppercase; letter-spacing: 0.08em; }
.mp-method-copy {
    background: rgba(251,244,237,0.55); border: 1px solid rgba(130,105,80,0.10);
    border-radius: var(--c-radius-sm); padding: 16px 18px;
    color: var(--c-body); font-size: 13px; line-height: 1.72;
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
}
.mp-method-copy strong { color: var(--c-ink); }
.mp-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
.mp-chip { display: inline-flex; align-items: center; min-height: 28px; padding: 4px 12px; border-radius: var(--c-radius-pill); border: 1px solid rgba(130,105,80,0.12); background: rgba(255,255,255,0.72); color: var(--c-secondary); font-size: 11px; font-weight: 650; }
.mp-status { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 0; }
.v3-badge { display: inline-flex; align-items: center; min-height: 30px; padding: 4px 11px; border-radius: var(--c-radius-pill); border: 1px solid rgba(130,105,80,0.10); background: rgba(255,255,255,0.70); color: var(--c-secondary); font-size: 11px; font-weight: 600; }
.v3-badge-accent { display: inline-flex; align-items: center; min-height: 30px; padding: 4px 11px; border-radius: var(--c-radius-pill); border: 1px solid rgba(184,115,59,0.18); background: var(--c-badge-bg); color: var(--c-accent); font-size: 11px; font-weight: 680; }
.mp-status-msg { color: var(--c-secondary); font-size: 12px; line-height: 1.6; }
.mp-path { display: inline-flex; margin-top: 4px; padding: 4px 10px; border: 1px solid rgba(130,105,80,0.10); background: rgba(255,255,255,0.78); color: var(--c-secondary); border-radius: var(--c-radius-pill); font-size: 11px; word-break: break-all; }

/* ==================================================================
   METRICS
   ================================================================== */
#metrics-section { max-width: 1340px; margin: 0 auto; padding: 40px 64px 0; }
.v3-glass-card {
    background: var(--c-surface); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--c-border); border-radius: var(--c-radius);
    box-shadow: var(--c-shadow); padding: 28px; overflow: hidden;
}
#histo-card .image-container,#histo-card [data-testid="image"] {
    border: 1px solid rgba(130,105,80,0.08) !important;
    border-radius: var(--c-radius) !important;
    background: rgba(255,255,255,0.55) !important; box-shadow: var(--c-shadow) !important;
}

/* ==================================================================
   FOOTER
   ================================================================== */
.site-footer {
    max-width: 1340px; margin: 64px auto 40px; padding: 32px 64px 0;
    background: linear-gradient(to bottom, rgba(247,241,233,0) 0%, rgba(247,241,233,0.18) 40%, transparent 100%);
    text-align: center;
}
.footer-text { font-size: 12px; color: var(--c-secondary); line-height: 1.7; }

/* ==================================================================
   GLOBAL IMAGE RESET
   ================================================================== */
.gradio-container .image-container,.gradio-container [data-testid="image"] {
    background: rgba(255,255,255,0.50) !important;
    border: 1px solid rgba(130,105,80,0.08) !important;
    border-radius: var(--c-radius) !important; overflow: hidden !important;
}
.gradio-container .image-container img,.gradio-container [data-testid="image"] img {
    object-fit: contain !important; border-radius: var(--c-radius-sm) !important;
}

/* ==================================================================
   HIDE GRADIO DEFAULTS
   ================================================================== */
.gradio-container label,.gradio-container .label-wrap { color: var(--c-secondary) !important; font-weight: 600 !important; font-size: 13px !important; }
.gradio-container input,.gradio-container textarea,.gradio-container select {
    background: rgba(255,255,255,0.78) !important; color: var(--c-ink) !important;
    border-color: var(--c-border) !important; border-radius: var(--c-radius-sm) !important;
}
.gradio-container>footer,footer,a[href*="gradio.app"],a[href*="gradio.live"],button[aria-label="Use via API"] {
    opacity: 0.04 !important; max-height: 8px !important; overflow: hidden !important;
}

/* Hide method-selector's auto-label */
[data-testid="radio-group"] > label:first-child,
label[for^="method-selector"]:not([for="method-selector"]) { display: none !important; }

/* ==================================================================
   RESPONSIVE
   ================================================================== */
@media (max-width: 900px) {
    .hero-content { flex-direction: column; padding: 40px 24px; }
    .hero-headline { font-size: 36px; }
    .hero-text { flex: 1 1 100%; max-width: none; }
    .workspace-grid { grid-template-columns: 1fr; }
    .control-panel { border-right: none; border-bottom: 1px solid var(--c-border-soft); }
    .overview-cards { grid-template-columns: 1fr; }
    #overview-section-v3,#studio-section,#metrics-section,.site-footer { padding-left: 24px; padding-right: 24px; }
    #overview-section-v3 { margin-top: -70px; }
}
@media (prefers-reduced-motion: reduce) {
    html { scroll-behavior: auto; }
    .hero-video-bg { display: none !important; }
}
"""

# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def build_hero_html() -> str:
    video = ""
    if HERO_VIDEO_EXISTS:
        video = (
            '<video autoplay muted loop playsinline preload="auto" class="hero-video-bg">'
            '<source src="/gradio_api/file=assets/hero_memory_light.mp4" type="video/mp4">'
            '<source src="/file=assets/hero_memory_light.mp4" type="video/mp4">'
            '</video>'
        )
    return f"""
<section id="hero-section-v3">
    {video}
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <div class="hero-text">
            <p class="hero-eyebrow">MS-SE-U-NET &middot; PHOTO RESTORATION PROTOTYPE</p>
            <h1 class="hero-headline">Illuminate<br><em>What Was</em><br>Left in Shadow</h1>
            <p class="hero-sub">把暗处的记忆重新照亮</p>
            <p class="hero-desc">A warm low-light image enhancement prototype powered by MS-SE-U-Net, multi-scale attention, and structure-aware restoration.</p>
            <div class="hero-tags-row">
                <span class="hero-tag">Low-Light Enhancement</span>
                <span class="hero-tag">Photo Restoration</span>
                <span class="hero-tag">Multi-scale Attention</span>
                <span class="hero-tag">MS-SE-U-Net</span>
            </div>
            <div class="hero-ctas">
                <button class="hero-cta primary" onclick="document.getElementById('studio-section').scrollIntoView({{behavior:'smooth'}})">Try Now</button>
                <button class="hero-cta secondary" onclick="document.getElementById('metrics-section').scrollIntoView({{behavior:'smooth'}})">View Results</button>
            </div>
        </div>
    </div>
</section>
"""


def build_overview_html() -> str:
    return """
<div id="overview-section-v3">
    <p class="overview-quote">From darkness to detail, from pixels to memory.</p>
    <div class="overview-cards">
        <div class="overview-card"><div class="overview-icon ov-icon-t">T</div><h3>Traditional Methods</h3><p>Gamma correction, CLAHE, and Retinex provide fast training-free baselines for basic illumination adjustment.</p></div>
        <div class="overview-card"><div class="overview-icon ov-icon-d">D</div><h3>Deep Learning</h3><p>U-Net serves as the encoder-decoder baseline; MS-SE-U-Net adds multi-scale attention and a combined restoration loss.</p></div>
        <div class="overview-card"><div class="overview-icon ov-icon-e">E</div><h3>Evaluation</h3><p>PSNR, SSIM, and inference time quantify reconstruction quality, structural similarity, and real-time performance.</p></div>
    </div>
</div>
"""


def build_ws_header(eyebrow: str, title: str, desc: str) -> str:
    return f"""
<div class="ws-header">
    <p class="ws-eyebrow">{eyebrow}</p>
    <h2 class="ws-title">{title}</h2>
    <p class="ws-desc">{desc}</p>
</div>
"""


# ---------------------------------------------------------------------------
# Core logic (unchanged)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-Light Image Enhancement App v3 — Premium landing page with AI restoration studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n  Local test:   python app_v3.py\n  LAN access:   python app_v3.py --server-name 0.0.0.0\n  Share + auth: python app_v3.py --share --auth demo 123456\n  Custom port:  python app_v3.py --server-port 7861",
    )
    parser.add_argument("--share", action="store_true", help="Create a temporary public gradio.live link.")
    parser.add_argument("--auth", nargs=2, metavar=("USER", "PASS"), default=None, help="Enable login protection.")
    parser.add_argument("--server-name", default="0.0.0.0", help="Server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port.")
    return parser.parse_args()


def load_state_dict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def load_models() -> tuple[dict[str, torch.nn.Module], dict[str, str]]:
    models: dict[str, torch.nn.Module] = {}
    errors: dict[str, str] = {}
    specs = {
        "U-Net": (UNet, CHECKPOINTS["U-Net"]),
        "MS-SE-U-Net": (MultiScaleSEUNet, CHECKPOINTS["MS-SE-U-Net"]),
    }
    for name, (model_cls, ckpt) in specs.items():
        if not ckpt.exists():
            errors[name] = f"{ckpt} not found."
            continue
        try:
            model = model_cls().to(DEVICE)
            load_state_dict(model, ckpt)
            model.eval()
            models[name] = model
        except Exception as exc:
            errors[name] = f"{name} failed: {exc}"
    return models, errors


MODELS, MODEL_ERRORS = load_models()


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Please upload a low-light image.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def enhance_with_model(image_rgb: np.ndarray, method: str) -> np.ndarray:
    if method not in MODELS:
        raise RuntimeError(MODEL_ERRORS.get(method, f"{method} not loaded."))
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        output = MODELS[method](tensor)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
    out_rgb = output.squeeze(0).detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return cv2.resize((out_rgb * 255.0).round().astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)


def enhance_image(image_rgb: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    image_rgb = ensure_uint8_rgb(image_rgb)
    start = time.perf_counter()
    if method == "Gamma":
        enhanced = bgr_to_rgb(gamma_enhance(rgb_to_bgr(image_rgb)))
    elif method == "CLAHE":
        enhanced = bgr_to_rgb(clahe_enhance(rgb_to_bgr(image_rgb)))
    elif method == "Retinex":
        enhanced = bgr_to_rgb(retinex_enhance(rgb_to_bgr(image_rgb)))
    elif method in {"U-Net", "MS-SE-U-Net"}:
        enhanced = enhance_with_model(image_rgb, method)
    else:
        raise ValueError(f"Unsupported method: {method}")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    return enhanced, time.perf_counter() - start


def make_placeholder(width: int, height: int, label: str) -> np.ndarray:
    img = np.full((height, width, 3), (249, 246, 242), dtype=np.uint8)
    m = 24
    cv2.rectangle(img, (m, m), (width - m, height - m), (236, 230, 222), 1, cv2.LINE_AA)
    fs = max(0.5, min(0.8, width / 960))
    ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
    tx = max(0, (width - ts[0]) // 2)
    ty = max(0, (height + ts[1]) // 2)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (118, 100, 84), 1, cv2.LINE_AA)
    return img


def make_histogram(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    if plt is None:
        raise RuntimeError("matplotlib not available.")
    before_gray = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY)
    fig, ax = plt.subplots(figsize=(7.4, 2.8), dpi=130)
    fig.patch.set_facecolor("#FDFBF9")
    ax.set_facecolor("#FDFBF9")
    ax.hist(before_gray.ravel(), bins=72, range=(0, 255), histtype="stepfilled", color="#A89888", alpha=0.22, label="Original")
    ax.hist(after_gray.ravel(), bins=72, range=(0, 255), histtype="stepfilled", color="#B8733B", alpha=0.50, label="Enhanced")
    ax.set_title("Brightness Distribution", color="#2F2924", fontsize=13, pad=10, fontweight="bold")
    ax.set_xlabel("Brightness", color="#7A6B5E", labelpad=7, fontsize=10)
    ax.set_ylabel("Pixel Count", color="#7A6B5E", labelpad=7, fontsize=10)
    ax.tick_params(colors="#7A6B5E", labelsize=9)
    ax.grid(axis="y", color="#E6DDD3", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=True, facecolor="#FDFBF9", edgecolor="#E6DDD3", labelcolor="#2F2924", fontsize=9, loc="upper right")
    for spine in ax.spines.values():
        spine.set_color("#E6DDD3")
        spine.set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.1)
    fig.canvas.draw()
    hist = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return hist


def save_output(image_rgb: np.ndarray, method: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe = method.lower().replace("-", "_").replace(" ", "_")
    path = OUTPUT_DIR / f"{safe}_{ts}.png"
    if not cv2.imwrite(str(path), rgb_to_bgr(image_rgb)):
        raise IOError(f"Save failed: {path}")
    return path


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def run_enhancement(image: np.ndarray | None, method: str):
    try:
        image_rgb = ensure_uint8_rgb(image)
        enhanced_rgb, t = enhance_image(image_rgb, method)
        out_path = save_output(enhanced_rgb, method)
        hist = make_histogram(image_rgb, enhanced_rgb)
        time_html = f'<span class="v3-badge-accent">Inference: {t:.4f}s</span>'
        status_html = (
            f"Enhancement complete. Brightness recovered and details restored."
            f'<br><span class="mp-path">Saved: {out_path}</span>'
        )
        return (image_rgb, enhanced_rgb, time_html, METHOD_DESCRIPTIONS[method], status_html, hist)
    except Exception as exc:
        return (image, make_placeholder(720, 440, "Enhancement Failed"),
                '<span class="v3-badge">Inference: failed</span>',
                METHOD_DESCRIPTIONS.get(method, "Select a method."),
                f"Enhancement failed: {exc}",
                make_placeholder(900, 300, "Histogram Unavailable"))


def clear_outputs():
    return (None,
            make_placeholder(720, 440, "Before"),
            make_placeholder(720, 440, "After"),
            "MS-SE-U-Net",
            METHOD_DESCRIPTIONS["MS-SE-U-Net"],
            '<span class="v3-badge">Inference: waiting</span>',
            "Upload a low-light photo and choose a method to begin.",
            make_placeholder(900, 300, "Brightness Histogram"))


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    if gr is None:
        raise RuntimeError("gradio not available.")

    block_kwargs = {"title": "Illuminate What Was Left in Shadow"}
    launch_accepts_css = "css" in inspect.signature(gr.Blocks.launch).parameters
    if "css" in inspect.signature(gr.Blocks).parameters and not launch_accepts_css:
        block_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**block_kwargs) as demo:

        # ================================================================
        # HERO
        # ================================================================
        gr.HTML(build_hero_html())

        # ================================================================
        # OVERVIEW
        # ================================================================
        gr.HTML(build_overview_html())

        # ================================================================
        # AI RESTORATION WORKSPACE
        # ================================================================
        gr.HTML('<div class="ws-sep"></div>')
        gr.HTML(build_ws_header(
            "AI Restoration Studio",
            "图像增强工作台",
            "Upload a low-light image, choose an enhancement method, and preview the restored result in real time.",
        ))

        with gr.HTML('<div id="studio-section"><div class="workspace-shell"><div class="workspace-grid">'):
            pass

        # ── Left: Control Panel ──
        with gr.Column(elem_id="upload-panel-cp", elem_classes=["control-panel"]):
            gr.HTML('<p class="cp-title">Control Panel</p>')

            input_image = gr.Image(label="", type="numpy", height=280, show_label=False)

            method = gr.Radio(
                choices=["Gamma", "CLAHE", "Retinex", "U-Net", "MS-SE-U-Net"],
                value="MS-SE-U-Net",
                label="",
                elem_id="method-selector",
                interactive=True,
            )

            with gr.Column(elem_classes=["action-row"]):
                enhance_btn = gr.Button("Illuminate Image", variant="primary", elem_classes=["v3-btn-primary"])
                clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["v3-btn-secondary"])

        # ── Right: Preview + Method Panel ──
        with gr.Column(elem_classes=["main-panel"]):
            # Live Preview
            gr.HTML('<p class="preview-title">Live Preview</p>')
            with gr.Row(elem_classes=["preview-area"]):
                with gr.Column(elem_classes=["preview-card"]):
                    gr.HTML('<p class="pv-label">Before</p>')
                    original_output = gr.Image(label="", type="numpy", height=240, show_label=False,
                                               value=make_placeholder(480, 320, "Original"))
                with gr.Column(elem_classes=["preview-card", "after-card"]):
                    gr.HTML('<p class="pv-label-after">After</p>')
                    enhanced_output = gr.Image(label="", type="numpy", height=240, show_label=False,
                                               value=make_placeholder(480, 320, "Enhanced"))

            # Method info
            gr.HTML('<p class="mp-method-title">Current Method</p>')
            method_desc = gr.Markdown(METHOD_DESCRIPTIONS["MS-SE-U-Net"], elem_classes=["mp-method-copy"])

            gr.HTML(f"""
            <div class="mp-chips">
                <span class="mp-chip">Multi-scale Feature Extraction</span>
                <span class="mp-chip">SE Channel Attention</span>
                <span class="mp-chip">Combined Restoration Loss</span>
            </div>
            <div class="mp-status">
                <span class="v3-badge">Device: {DEVICE}</span>
                <span class="v3-badge">5 Methods Available</span>
            </div>
            """)

            inference_time = gr.Markdown('<span class="v3-badge-accent">Inference: waiting</span>')
            status_msg = gr.Markdown("Upload a low-light photo and choose a method to begin.", elem_classes=["mp-status-msg"])

        gr.HTML('</div></div></div></div>')  # close workspace-grid, workspace-shell, studio-section

        # ================================================================
        # METRICS & HISTOGRAM
        # ================================================================
        gr.HTML('<div class="ws-sep"></div>')
        gr.HTML(build_ws_header(
            "Metrics & Analysis",
            "指标与分析",
            "Inference time and brightness distribution after enhancement.",
        ))

        with gr.HTML('<div id="metrics-section">'):
            pass

        with gr.Column(elem_id="histo-card", elem_classes=["v3-glass-card"]):
            histogram_output = gr.Image(label="", type="numpy", height=280, show_label=False,
                                        value=make_placeholder(900, 300, "Brightness Histogram"))

        gr.HTML('</div>')

        # ================================================================
        # FOOTER
        # ================================================================
        gr.HTML('<div class="site-footer"><p class="footer-text">Low-Light Image Enhancement &middot; MS-SE-U-Net &middot; Gradio Prototype &middot; PyTorch</p></div>')

        # ================================================================
        # Event bindings
        # ================================================================
        method.change(fn=lambda s: METHOD_DESCRIPTIONS[s], inputs=method, outputs=method_desc)
        enhance_btn.click(fn=run_enhancement, inputs=[input_image, method],
                          outputs=[original_output, enhanced_output, inference_time, method_desc, status_msg, histogram_output])
        clear_btn.click(fn=clear_outputs, inputs=None,
                        outputs=[input_image, original_output, enhanced_output, method, method_desc, inference_time, status_msg, histogram_output])

    return demo


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_demo(demo: gr.Blocks, *, share: bool = False,
                auth: tuple[str, str] | None = None,
                server_name: str = "0.0.0.0", server_port: int = 7860) -> None:
    launch_kwargs = {
        "server_name": server_name, "server_port": server_port,
        "share": share, "allowed_paths": [str(_ASSETS_DIR)],
    }
    if auth is not None:
        launch_kwargs["auth"] = auth
    if "css" in inspect.signature(demo.launch).parameters:
        launch_kwargs["css"] = CUSTOM_CSS
    try:
        demo.launch(**launch_kwargs)
    except OSError as exc:
        if "port" in str(exc).lower():
            print(f"Port {server_port} unavailable. Try --server-port {server_port + 1}")
        raise


def main() -> None:
    args = parse_args()
    missing = []
    if GRADIO_IMPORT_ERROR is not None:
        missing.append("gradio")
    if MATPLOTLIB_IMPORT_ERROR is not None:
        missing.append("matplotlib")
    if missing:
        raise SystemExit(f"Missing: {', '.join(missing)}. Run pip install -r requirements.txt.")

    print("=" * 56)
    print("  Low-Light Image Enhancement App v3")
    print("=" * 56)
    print(f"  Device:      {DEVICE}")
    if MODEL_ERRORS:
        for n, m in MODEL_ERRORS.items():
            print(f"  {n}: {m}")
    else:
        print(f"  Checkpoints: loaded")
    if HERO_VIDEO_EXISTS:
        print(f"  Hero video:  loaded ({HERO_VIDEO.name})")
    else:
        print(f"  Hero video:  not found — using static fallback")

    auth = tuple(args.auth) if args.auth is not None else None

    print()
    print(f"  Local URL:   http://127.0.0.1:{args.server_port}")
    if args.server_name not in ("127.0.0.1", "localhost"):
        print(f"  LAN URL:     http://<your-ip>:{args.server_port}")
    print()
    if args.share:
        print("  Gradio will print a temporary public URL below.")
    if auth:
        print("  Auth enabled. Share username/password only with teammates.")
    print("  Keep this terminal running. Press Ctrl+C to stop.")
    print("=" * 56)
    print()

    demo = build_demo()
    launch_demo(demo, share=args.share, auth=auth, server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
