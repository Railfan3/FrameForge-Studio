"""
Image Resizer Pro — Advanced Edition (PyQt6 + Pillow)
====================================================
A professional dark-themed desktop dashboard to batch resize, convert, crop, watermark, optimize, and manage images with presets, automation, and polished UX.

Highlights
---------
• Advanced Processing: multiple resize modes, smart crop (center/face*), format conversion (JPEG/PNG/WebP/TIFF/BMP), quality/compression, EXIF keep/strip, color adjust (brightness/contrast/saturation/sharpen), filters (grayscale/blur), background fill for non-alpha formats, watermark (text/logo) with opacity/size/position.
• Workflow Automation: add files/folders, drag & drop, optional folder watcher*, presets save/load, background processing with progress & logs.
• Output Handling: custom naming patterns, auto subfolders, duplicate handling, ZIP download, open output folder, summary report.
• Pro UI: dark theme with high contrast, emoji accents 📸🖼️✨, responsive splitter layout, header banner, thumbnail dashboard + live preview, accessible controls.

*Optional: Face crop and folder watcher require extra packages (OpenCV + NumPy, watchdog).

Dependencies
------------
    pip install PyQt6 Pillow
    # Optional features:
    pip install opencv-python numpy watchdog

Run
---
    python image_resizer_pro_advanced.py
"""
from __future__ import annotations
import os, sys, io, zipfile, time, json, math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from PyQt6.QtWidgets import QScrollArea


from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Optional deps
_HAS_CV2 = False
_HAS_WATCHDOG = False
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    _HAS_CV2 = True
except Exception:
    pass
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
    _HAS_WATCHDOG = True
except Exception:
    pass

from PyQt6.QtCore import Qt, QSize, QThreadPool, QRunnable, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QPixmap, QAction, QPainter, QLinearGradient, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QFileDialog, QPushButton, QLabel, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QLineEdit,
    QProgressBar, QTextEdit, QMessageBox, QScrollArea
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
FORMATS_UI = ["original", "jpeg", "png", "webp", "tiff", "bmp"]

@dataclass
class JobConfig:
    # Resize
    resize_mode: str  # fit, exact, percent, maxedge
    width: int
    height: int
    scale_percent: int
    max_edge: int
    # Crop
    crop_mode: str  # none, center, face
    # Conversion/quality/metadata
    out_format: str
    quality: int
    keep_exif: bool
    strip_metadata: bool
    # Color & Filters
    adjust_brightness: float  # 1.0 = no change
    adjust_contrast: float
    adjust_saturation: float
    sharpen_amount: float
    do_grayscale: bool
    blur_radius: float
    # Watermark
    wm_text: str
    wm_logo_path: str
    wm_position: str  # Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center
    wm_opacity: int   # 1-100
    wm_size_percent: int
    wm_margin: int
    # Background color for non-alpha formats
    bg_color: Tuple[int, int, int]
    # Output
    out_dir: str
    name_pattern: str
    make_subfolders_by_format: bool
    overwrite: bool

POSITIONS = {
    "Top-Left": (0.0, 0.0),
    "Top-Right": (1.0, 0.0),
    "Bottom-Left": (0.0, 1.0),
    "Bottom-Right": (1.0, 1.0),
    "Center": (0.5, 0.5),
}

# ---------------- Worker -----------------
class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    file_done = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal(int, int, str)  # ok, total, summary

class ProcessTask(QRunnable):
    def __init__(self, files: List[str], cfg: JobConfig):
        super().__init__()
        self.files = files
        self.cfg = cfg
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        ok = 0
        total = len(self.files)
        total_saved_bytes = 0
        for idx, path in enumerate(self.files, start=1):
            try:
                before = os.path.getsize(path) if os.path.exists(path) else 0
                out_path = self.process_one(path)
                after = os.path.getsize(out_path) if os.path.exists(out_path) else before
                total_saved_bytes += max(0, before - after)
                ok += 1
                self.signals.file_done.emit(out_path)
            except Exception as e:
                self.signals.error.emit(f"{os.path.basename(path)} → {e}")
            self.signals.progress.emit(int(idx / total * 100))
        saved_mb = total_saved_bytes / (1024*1024)
        summary = f"Saved ~{saved_mb:.2f} MB across {ok}/{total} files"
        self.signals.finished.emit(ok, total, summary)

    # Core pipeline
    def process_one(self, path: str) -> str:
        base_name, ext = os.path.splitext(os.path.basename(path))
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            # Crop
            if self.cfg.crop_mode == "center":
                im = self.center_crop_to_aspect(im, self.cfg.width, self.cfg.height)
            elif self.cfg.crop_mode == "face" and _HAS_CV2:
                im = self.face_crop(im, self.cfg.width, self.cfg.height) or im
            # Resize
            im = self.apply_resize(im)
            # Color adjustments and filters
            im = self.apply_adjustments(im)
            # Watermark
            if self.cfg.wm_text or (self.cfg.wm_logo_path and os.path.exists(self.cfg.wm_logo_path)):
                im = self.apply_watermark(im)
            # Background for non-alpha target formats
            target_fmt = self.resolve_format(ext.lower(), self.cfg.out_format)
            im = self.prepare_background(im, target_fmt)
            # Save
            w, h = im.size
            fname = self.cfg.name_pattern.format(
                name=base_name, index="{0:04d}".format(0), width=w, height=h, format=target_fmt.lower()
            )
            out_ext = self.ext_for_format(target_fmt)
            fname = os.path.splitext(fname)[0] + out_ext
            out_dir = self.cfg.out_dir
            if self.cfg.make_subfolders_by_format:
                out_dir = os.path.join(out_dir, target_fmt.lower())
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            if not self.cfg.overwrite:
                out_path = self.unique_path(out_path)
            save_kwargs = self.save_kwargs_for_format(target_fmt)
            if self.cfg.keep_exif and "exif" in im.info:
                save_kwargs["exif"] = im.info["exif"]
            im.save(out_path, format=target_fmt, **save_kwargs)
            return out_path

    # --- helpers ---
    def resolve_format(self, input_ext_lower: str, out_fmt_choice: str) -> str:
        if out_fmt_choice.lower() == "original":
            return {
                ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP",
                ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF", ".gif": "GIF"
            }.get(input_ext_lower, "JPEG")
        return out_fmt_choice.upper()

    def ext_for_format(self, fmt: str) -> str:
        return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp", "GIF": ".gif"}.get(fmt, ".jpg")

    def unique_path(self, path: str) -> str:
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        i = 1
        while True:
            cand = f"{base}_{i}{ext}"
            if not os.path.exists(cand):
                return cand
            i += 1

    def save_kwargs_for_format(self, fmt: str) -> dict:
        kwargs = {}
        if fmt == "JPEG":
            kwargs.update(quality=self.cfg.quality, optimize=True, progressive=True)
        elif fmt == "WEBP":
            kwargs.update(quality=self.cfg.quality)
        elif fmt == "PNG":
            kwargs.update(optimize=True)
        return kwargs

    def prepare_background(self, im: Image.Image, target_fmt: str) -> Image.Image:
        needs_bg = target_fmt in {"JPEG", "BMP", "TIFF"} and ("A" in im.getbands())
        if needs_bg:
            bg = Image.new("RGB", im.size, self.cfg.bg_color)
            bg.paste(im, mask=im.split()[-1])
            return bg
        if target_fmt in {"JPEG", "BMP", "TIFF"} and im.mode not in {"RGB", "L"}:
            return im.convert("RGB")
        return im

    def apply_resize(self, im: Image.Image) -> Image.Image:
        m = self.cfg.resize_mode
        if m == "fit":
            target = (max(1, self.cfg.width), max(1, self.cfg.height))
            return ImageOps.contain(im, target, Image.Resampling.LANCZOS)
        if m == "exact":
            target = (max(1, self.cfg.width), max(1, self.cfg.height))
            return im.resize(target, Image.Resampling.LANCZOS)
        if m == "percent":
            scale = max(1, self.cfg.scale_percent) / 100.0
            w = max(1, int(im.width * scale)); h = max(1, int(im.height * scale))
            return im.resize((w, h), Image.Resampling.LANCZOS)
        if m == "maxedge":
            maxe = max(1, self.cfg.max_edge)
            ratio = max(im.width, im.height) / float(maxe)
            if ratio > 1:
                w = max(1, int(im.width / ratio)); h = max(1, int(im.height / ratio))
                return im.resize((w, h), Image.Resampling.LANCZOS)
            return im
        return im

    def center_crop_to_aspect(self, im: Image.Image, tw: int, th: int) -> Image.Image:
        if tw <= 0 or th <= 0:
            return im
        target_aspect = tw / float(th)
        w, h = im.size
        src_aspect = w / float(h)
        if src_aspect > target_aspect:
            new_w = int(h * target_aspect)
            left = (w - new_w) // 2
            return im.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_aspect)
            top = (h - new_h) // 2
            return im.crop((0, top, w, top + new_h))

    def face_crop(self, im: Image.Image, tw: int, th: int) -> Optional[Image.Image]:
        if not _HAS_CV2 or tw <= 0 or th <= 0:
            return None
        cv = cv2.cvtColor(np.array(im.convert('RGB')), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + w // 2, y + h // 2
        target_aspect = tw / float(th)
        crop_w = min(cv.shape[1], int(max(w * 2.2, th * target_aspect)))
        crop_h = int(crop_w / target_aspect)
        left = max(0, min(cv.shape[1] - crop_w, cx - crop_w // 2))
        top = max(0, min(cv.shape[0] - crop_h, cy - crop_h // 2))
        cv_crop = cv[top:top+crop_h, left:left+crop_w]
        return Image.fromarray(cv2.cvtColor(cv_crop, cv2.COLOR_BGR2RGB))

    def apply_adjustments(self, im: Image.Image) -> Image.Image:
        # Brightness/contrast/saturation
        if abs(self.cfg.adjust_brightness - 1.0) > 1e-3:
            im = ImageEnhance.Brightness(im).enhance(self.cfg.adjust_brightness)
        if abs(self.cfg.adjust_contrast - 1.0) > 1e-3:
            im = ImageEnhance.Contrast(im).enhance(self.cfg.adjust_contrast)
        if abs(self.cfg.adjust_saturation - 1.0) > 1e-3:
            im = ImageEnhance.Color(im).enhance(self.cfg.adjust_saturation)
        if self.cfg.sharpen_amount > 0:
            im = ImageEnhance.Sharpness(im).enhance(1.0 + self.cfg.sharpen_amount)
        if self.cfg.do_grayscale:
            im = ImageOps.grayscale(im).convert("RGB")
        if self.cfg.blur_radius > 0:
            im = im.filter(ImageFilter.GaussianBlur(self.cfg.blur_radius))
        return im

    def apply_watermark(self, im: Image.Image) -> Image.Image:
        im = im.convert("RGBA")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        W, H = im.size
        margin = self.cfg.wm_margin
        px, py = POSITIONS.get(self.cfg.wm_position, (1.0, 1.0))
        # Logo
        if self.cfg.wm_logo_path and os.path.exists(self.cfg.wm_logo_path):
            try:
                logo = Image.open(self.cfg.wm_logo_path).convert("RGBA")
                scale = max(1, self.cfg.wm_size_percent) / 100.0
                lw = max(1, int(W * 0.22 * scale))
                ratio = lw / float(logo.width)
                lh = max(1, int(logo.height * ratio))
                logo = logo.resize((lw, lh), Image.Resampling.LANCZOS)
                x = int(px * (W - lw)); y = int(py * (H - lh))
                x = max(margin, min(W - lw - margin, x))
                y = max(margin, min(H - lh - margin, y))
                if self.cfg.wm_opacity < 100:
                    a = logo.getchannel("A").point(lambda p: int(p * self.cfg.wm_opacity / 100))
                    logo.putalpha(a)
                overlay.paste(logo, (x, y), logo)
            except Exception:
                pass
        # Text
        if self.cfg.wm_text:
            try:
                font_size = max(12, int(min(W, H) * 0.05 * (self.cfg.wm_size_percent / 100.0)))
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
                tw, th = draw.textbbox((0, 0), self.cfg.wm_text, font=font)[2:]
                x = int(px * (W - tw)); y = int(py * (H - th))
                x = max(margin, min(W - tw - margin, x))
                y = max(margin, min(H - th - margin, y))
                # Outline + fill for contrast
                alpha = int(255 * self.cfg.wm_opacity / 100)
                draw.text((x+1, y+1), self.cfg.wm_text, font=font, fill=(0,0,0,alpha))
                draw.text((x, y), self.cfg.wm_text, font=font, fill=(255,255,255,alpha))
            except Exception:
                pass
        return Image.alpha_composite(im, overlay)

# ---------------- UI -----------------
class Banner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(84)

    def paintEvent(self, e):
        p = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QColor("#0d0f14"))
        grad.setColorAt(1.0, QColor("#1f2a44"))
        p.fillRect(self.rect(), grad)
        p.setPen(QColor("#e6f0ff"))
        font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(20, 30, "📸 FrameForge Studio")
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(20, 56, "Batch Resize • Convert • Watermark • Optimize — Dark Edition 🖼️✨")

class ThumbList(QListWidget):
    def __init__(self):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setIconSize(QSize(140, 140))
        self.setGridSize(QSize(160, 180))
        self.setSpacing(8)
        self.setAcceptDrops(True)
        self.setStyleSheet("background:#121418;border:1px solid #253045;border-radius:10px;color:#cfe2ff")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: super().dragEnterEvent(e)
    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
        else: super().dragMoveEvent(e)
    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = []
            for url in e.mimeData().urls():
                p = url.toLocalFile()
                if os.path.isdir(p):
                    for root, _, files in os.walk(p):
                        for f in files:
                            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                                paths.append(os.path.join(root, f))
                else:
                    if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS:
                        paths.append(p)
            self.add_files(paths)
            e.acceptProposedAction()
        else:
            super().dropEvent(e)

    def add_files(self, file_paths: List[str]):
        for path in file_paths:
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            try:
                with Image.open(path) as im:
                    im.thumbnail((140, 140))
                    bio = io.BytesIO(); im.save(bio, format="PNG")
                    px = QPixmap(); px.loadFromData(bio.getvalue())
                    item.setIcon(QIcon(px))
            except Exception:
                pass
            self.addItem(item)

    def selected_paths(self) -> List[str]:
        return [it.data(Qt.ItemDataRole.UserRole) for it in (self.selectedItems() or [])]
    def all_paths(self) -> List[str]:
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FrameForge Studio")
        self.setMinimumSize(1320, 780)
        self.threadpool = QThreadPool.globalInstance()
        self._watcher = None
        self._watch_dir = None
        self._preset_path = os.path.join(os.path.expanduser("~"), ".image_resizer_pro_preset.json")
        self._build_ui()
        self._apply_theme()

    # UI build
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(10,10,10,10); root.setSpacing(10)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(5, 5, 5, 5)


        # Banner
        root.addWidget(Banner())

        # Top actions
        bar = QHBoxLayout()
        self.btn_add = QPushButton("📂 Add Files/Folders…"); self.btn_add.clicked.connect(self.on_add)
        self.btn_clear = QPushButton("🧹 Clear"); self.btn_clear.clicked.connect(lambda: self.thumb.clear())
        self.btn_open_out = QPushButton("📁 Open Output"); self.btn_open_out.clicked.connect(self.on_open_output)
        self.btn_zip = QPushButton("⬇️ Download ZIP"); self.btn_zip.clicked.connect(self.on_zip_download)
        self.btn_save_preset = QPushButton("💾 Save Preset"); self.btn_save_preset.clicked.connect(self.on_save_preset)
        self.btn_load_preset = QPushButton("📤 Load Preset"); self.btn_load_preset.clicked.connect(self.on_load_preset)
        bar.addWidget(self.btn_add); bar.addWidget(self.btn_clear); bar.addStretch(1)
        bar.addWidget(self.btn_save_preset); bar.addWidget(self.btn_load_preset)
        bar.addWidget(self.btn_open_out); bar.addWidget(self.btn_zip)
        root.addLayout(bar)

        # Splitter: left = dashboard, right = controls & preview
        split = QSplitter(Qt.Orientation.Horizontal)
        # Left (thumbs)
        leftwrap = QWidget(); lv = QVBoxLayout(leftwrap); lv.setContentsMargins(0,0,0,0); lv.setSpacing(8)
        

        self.section_lbl = QLabel("Dashboard — Drop images/folders here or click Add 📸"); self.section_lbl.setStyleSheet("color:#cfe2ff;font-weight:600")
        self.thumb = ThumbList()
        lv.addWidget(self.section_lbl); lv.addWidget(self.thumb)

        # Right (controls + preview)
        rightwrap = QWidget(); rv = QVBoxLayout(rightwrap); rv.setContentsMargins(0,0,0,0); rv.setSpacing(8)
        controls = self._build_controls()
        preview = self._build_preview()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(controls)
        rv.addWidget(scroll_area, 1)

        rv.addWidget(preview, 1)

        split.addWidget(leftwrap); split.addWidget(rightwrap)
        split.setSizes([820, 480])
        root.addWidget(split, 1)

        # Bottom: process + progress + logs
        bottom = QHBoxLayout()
        self.btn_process_sel = QPushButton("🚀 Process Selected")
        self.btn_process_all = QPushButton("✨ Process All")
        self.btn_process_sel.clicked.connect(lambda: self.start_process(sel=True))
        self.btn_process_all.clicked.connect(lambda: self.start_process(sel=False))
        bottom.addWidget(self.btn_process_sel); bottom.addWidget(self.btn_process_all)
        bottom.addStretch(1)
        root.addLayout(bottom)

        self.progress = QProgressBar(); self.progress.setValue(0)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(140)
        root.addWidget(self.progress)
        root.addWidget(self.log)

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Controls ⚙️"); f = QFormLayout(box); f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Resize
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["fit","exact","percent","maxedge"])
        self.spin_w = QSpinBox(); self.spin_w.setRange(1, 10000); self.spin_w.setValue(1080)
        self.spin_h = QSpinBox(); self.spin_h.setRange(1, 10000); self.spin_h.setValue(1080)
        self.spin_percent = QSpinBox(); self.spin_percent.setRange(1, 1000); self.spin_percent.setValue(100)
        self.spin_maxedge = QSpinBox(); self.spin_maxedge.setRange(1, 10000); self.spin_maxedge.setValue(1600)
        # Crop
        crop_items = ["none", "center"] + (["face"] if _HAS_CV2 else [])
        self.cmb_crop = QComboBox(); self.cmb_crop.addItems(crop_items)
        # Format / quality / metadata
        self.cmb_fmt = QComboBox(); self.cmb_fmt.addItems(FORMATS_UI)
        self.spin_quality = QSpinBox(); self.spin_quality.setRange(1,100); self.spin_quality.setValue(85)
        self.chk_keep_exif = QCheckBox("Keep EXIF"); self.chk_strip_meta = QCheckBox("Strip metadata")
        # Color/filters
        self.dsp_bri = QDoubleSpinBox(); self.dsp_bri.setRange(0.1, 3.0); self.dsp_bri.setSingleStep(0.1); self.dsp_bri.setValue(1.0)
        self.dsp_con = QDoubleSpinBox(); self.dsp_con.setRange(0.1, 3.0); self.dsp_con.setSingleStep(0.1); self.dsp_con.setValue(1.0)
        self.dsp_sat = QDoubleSpinBox(); self.dsp_sat.setRange(0.1, 3.0); self.dsp_sat.setSingleStep(0.1); self.dsp_sat.setValue(1.0)
        self.dsp_sharp = QDoubleSpinBox(); self.dsp_sharp.setRange(0.0, 3.0); self.dsp_sharp.setSingleStep(0.1); self.dsp_sharp.setValue(0.0)
        self.chk_gray = QCheckBox("Grayscale")
        self.dsp_blur = QDoubleSpinBox(); self.dsp_blur.setRange(0.0, 10.0); self.dsp_blur.setSingleStep(0.1); self.dsp_blur.setValue(0.0)
        # Watermark
        self.edit_wm_text = QLineEdit(); self.edit_wm_logo = QLineEdit()
        self.cmb_wm_pos = QComboBox(); self.cmb_wm_pos.addItems(list(POSITIONS.keys()))
        self.spin_wm_opacity = QSpinBox(); self.spin_wm_opacity.setRange(1,100); self.spin_wm_opacity.setValue(60)
        self.spin_wm_size = QSpinBox(); self.spin_wm_size.setRange(10,300); self.spin_wm_size.setValue(100)
        self.spin_wm_margin = QSpinBox(); self.spin_wm_margin.setRange(0,200); self.spin_wm_margin.setValue(16)
        # Background
        self.edit_bg = QLineEdit("255,255,255")
        # Output
        self.edit_out = QLineEdit(os.path.join(os.path.expanduser("~"), "ImageResizerPro_Output"))
        self.edit_name = QLineEdit("{name}_{width}x{height}")
        self.chk_subfmt = QCheckBox("Subfolder by format")
        self.chk_overwrite = QCheckBox("Overwrite if exists")
        # Watch folder (optional)
        self.edit_watch = QLineEdit("")
        self.chk_watch = QCheckBox("Auto-process new files in watch folder (requires watchdog)")

        # Arrange
        f.addRow("Resize mode:", self.cmb_mode)
        f.addRow("Width:", self.spin_w)
        f.addRow("Height:", self.spin_h)
        f.addRow("Scale %:", self.spin_percent)
        f.addRow("Max edge:", self.spin_maxedge)
        f.addRow("Crop:", self.cmb_crop)
        f.addRow("Output format:", self.cmb_fmt)
        f.addRow("Quality:", self.spin_quality)
        f.addRow(self.chk_keep_exif)
        f.addRow(self.chk_strip_meta)
        f.addRow(QLabel("— Color & Filters —"))
        f.addRow("Brightness:", self.dsp_bri)
        f.addRow("Contrast:", self.dsp_con)
        f.addRow("Saturation:", self.dsp_sat)
        f.addRow("Sharpen:", self.dsp_sharp)
        f.addRow("Blur radius:", self.dsp_blur)
        f.addRow(self.chk_gray)
        f.addRow(QLabel("— Watermark —"))
        f.addRow("Text:", self.edit_wm_text)
        f.addRow("Logo path:", self.edit_wm_logo)
        f.addRow("Position:", self.cmb_wm_pos)
        f.addRow("Opacity %:", self.spin_wm_opacity)
        f.addRow("Size %:", self.spin_wm_size)
        f.addRow("Margin px:", self.spin_wm_margin)
        f.addRow(QLabel("— Background (RGB for non-alpha) —"))
        f.addRow("RGB:", self.edit_bg)
        f.addRow(QLabel("— Output & Naming —"))
        f.addRow("Folder:", self.edit_out)
        f.addRow("Pattern:", self.edit_name)
        f.addRow(self.chk_subfmt)
        f.addRow(self.chk_overwrite)
        f.addRow(QLabel("— Watch Folder —"))
        f.addRow("Folder:", self.edit_watch)
        f.addRow(self.chk_watch)
        return box

    def _build_preview(self) -> QWidget:
        box = QGroupBox("Preview 🖼️"); v = QVBoxLayout(box)
        self.preview = QLabel("Select an image to preview"); self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#0d0f14;border:1px solid #253045;border-radius:10px;color:#7aa2ff")
        v.addWidget(self.preview)
        # Hook selection change
        self.thumb.itemSelectionChanged.connect(self.update_preview)
        return box

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#0b0e13; color:#e6f0ff; font: 13px 'Segoe UI'; }
            QGroupBox { background:#10141b; border:1px solid #253045; border-radius:12px; margin-top:10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 4px 8px; color:#9cc4ff; }
            QPushButton { background:#263247; color:#e6f0ff; border:none; padding:8px 12px; border-radius:10px; }
            QPushButton:hover { background:#31415b; }
            QPushButton:pressed { background:#1f2a44; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background:#0f131a; color:#e6f0ff; border:1px solid #253045; padding:6px; border-radius:8px; }
            QTextEdit { background:#0f131a; color:#cfe2ff; border:1px solid #253045; border-radius:8px; }
            QProgressBar { background:#0f131a; border:1px solid #253045; border-radius:8px; text-align:center; }
            QProgressBar::chunk { background:#7aa2ff; border-radius:8px; }
            QListWidget { background:#0f131a; border:1px solid #253045; border-radius:12px; padding:8px; }
            QLabel { color:#e6f0ff; }
            """
        )

    # -------- Actions --------
    def on_add(self):
        # Files
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif)")
        paths = [str(p) for p in files]
        # Optional folder
        dir_path = QFileDialog.getExistingDirectory(self, "Or pick a folder (optional)")
        if dir_path:
            for root, _, fs in os.walk(dir_path):
                for f in fs:
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                        paths.append(os.path.join(root, f))
        self.thumb.add_files(paths)

    def current_config(self) -> JobConfig:
        bg_tuple = (255,255,255)
        try:
            parts = [int(x.strip()) for x in self.edit_bg.text().split(',')]
            if len(parts) == 3:
                bg_tuple = (max(0,min(255,parts[0])), max(0,min(255,parts[1])), max(0,min(255,parts[2])))
        except Exception:
            pass
        return JobConfig(
            resize_mode=self.cmb_mode.currentText(), width=self.spin_w.value(), height=self.spin_h.value(),
            scale_percent=self.spin_percent.value(), max_edge=self.spin_maxedge.value(),
            crop_mode=self.cmb_crop.currentText(), out_format=self.cmb_fmt.currentText(), quality=self.spin_quality.value(),
            keep_exif=self.chk_keep_exif.isChecked(), strip_metadata=self.chk_strip_meta.isChecked(),
            adjust_brightness=self.dsp_bri.value(), adjust_contrast=self.dsp_con.value(), adjust_saturation=self.dsp_sat.value(),
            sharpen_amount=self.dsp_sharp.value(), do_grayscale=self.chk_gray.isChecked(), blur_radius=self.dsp_blur.value(),
            wm_text=self.edit_wm_text.text().strip(), wm_logo_path=self.edit_wm_logo.text().strip(),
            wm_position=self.cmb_wm_pos.currentText(), wm_opacity=self.spin_wm_opacity.value(), wm_size_percent=self.spin_wm_size.value(),
            wm_margin=self.spin_wm_margin.value(), bg_color=bg_tuple,
            out_dir=self.edit_out.text().strip(), name_pattern=self.edit_name.text().strip() or "{name}_{width}x{height}",
            make_subfolders_by_format=self.chk_subfmt.isChecked(), overwrite=self.chk_overwrite.isChecked()
        )

    def start_process(self, sel: bool):
        files = self.thumb.selected_paths() if sel else self.thumb.all_paths()
        if not files:
            QMessageBox.warning(self, "No images", "Please add/select images to process.")
            return
        cfg = self.current_config()
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.progress.setValue(0)
        self.log.append("Starting…")
        task = ProcessTask(files, cfg)
        task.signals.progress.connect(self.progress.setValue)
        task.signals.file_done.connect(lambda p: self.log.append(f"✔ {os.path.basename(p)}"))
        task.signals.error.connect(lambda m: self.log.append(f"❌ {m}"))
        task.signals.finished.connect(lambda ok,total,summary: QMessageBox.information(self, "Done", f"Processed {ok}/{total} images\n{summary}"))
        self.threadpool.start(task)
        # Folder watch
        if _HAS_WATCHDOG and self.chk_watch.isChecked() and self.edit_watch.text().strip():
            self.start_watching(self.edit_watch.text().strip(), cfg)

    def update_preview(self):
        paths = self.thumb.selected_paths()
        if not paths:
            self.preview.setText("Select an image to preview")
            return
        p = paths[0]
        try:
            with Image.open(p) as im:
                im.thumbnail((800, 480))
                bio = io.BytesIO(); im.save(bio, format="PNG")
                px = QPixmap(); px.loadFromData(bio.getvalue())
                self.preview.setPixmap(px)
        except Exception:
            self.preview.setText("Preview not available")

    def on_open_output(self):
        d = self.edit_out.text().strip()
        if not os.path.isdir(d):
            QMessageBox.warning(self, "Output", "Output folder does not exist yet.")
            return
        if sys.platform.startswith('win'):
            os.startfile(d)
        elif sys.platform == 'darwin':
            os.system(f"open '{d}'")
        else:
            os.system(f"xdg-open '{d}'")

    def on_zip_download(self):
        out_dir = self.edit_out.text().strip()
        if not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Download", "No output folder found.")
            return
        zpath, _ = QFileDialog.getSaveFileName(self, "Save ZIP", os.path.join(out_dir, "images_processed.zip"), "Zip (*.zip)")
        if not zpath:
            return
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(out_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    arc = os.path.relpath(fp, out_dir)
                    z.write(fp, arc)
        QMessageBox.information(self, "Saved", f"ZIP saved at:\n{zpath}")

    # Presets
    def on_save_preset(self):
        cfg = self.current_config()
        try:
            with open(self._preset_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(cfg), f, indent=2)
            QMessageBox.information(self, "Preset", f"Saved preset to {self._preset_path}")
        except Exception as e:
            QMessageBox.warning(self, "Preset", f"Failed to save preset: {e}")

    def on_load_preset(self):
        try:
            with open(self._preset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.apply_preset(JobConfig(**data))
            QMessageBox.information(self, "Preset", "Preset loaded.")
        except Exception as e:
            QMessageBox.warning(self, "Preset", f"Failed to load preset: {e}")

    def apply_preset(self, cfg: JobConfig):
        self.cmb_mode.setCurrentText(cfg.resize_mode)
        self.spin_w.setValue(cfg.width); self.spin_h.setValue(cfg.height)
        self.spin_percent.setValue(cfg.scale_percent); self.spin_maxedge.setValue(cfg.max_edge)
        self.cmb_crop.setCurrentText(cfg.crop_mode)
        self.cmb_fmt.setCurrentText(cfg.out_format); self.spin_quality.setValue(cfg.quality)
        self.chk_keep_exif.setChecked(cfg.keep_exif); self.chk_strip_meta.setChecked(cfg.strip_metadata)
        self.dsp_bri.setValue(cfg.adjust_brightness); self.dsp_con.setValue(cfg.adjust_contrast)
        self.dsp_sat.setValue(cfg.adjust_saturation); self.dsp_sharp.setValue(cfg.sharpen_amount)
        self.chk_gray.setChecked(cfg.do_grayscale); self.dsp_blur.setValue(cfg.blur_radius)
        self.edit_wm_text.setText(cfg.wm_text); self.edit_wm_logo.setText(cfg.wm_logo_path)
        self.cmb_wm_pos.setCurrentText(cfg.wm_position); self.spin_wm_opacity.setValue(cfg.wm_opacity)
        self.spin_wm_size.setValue(cfg.wm_size_percent); self.spin_wm_margin.setValue(cfg.wm_margin)
        self.edit_bg.setText(','.join(map(str, cfg.bg_color)))
        self.edit_out.setText(cfg.out_dir); self.edit_name.setText(cfg.name_pattern)
        self.chk_subfmt.setChecked(cfg.make_subfolders_by_format); self.chk_overwrite.setChecked(cfg.overwrite)

    # Folder Watcher
    def start_watching(self, folder: str, cfg: JobConfig):
        if not _HAS_WATCHDOG:
            self.log.append("⚠️ watchdog not installed — folder watch disabled")
            return
        if self._watcher:
            try: self._watcher.stop(); self._watcher.join()
            except Exception: pass
        class Handler(FileSystemEventHandler):
            def __init__(self, mw: MainWindow, cfg: JobConfig):
                self.mw = mw; self.cfg = cfg
            def on_created(self, event):
                if event.is_directory: return
                if os.path.splitext(event.src_path)[1].lower() in SUPPORTED_EXTS:
                    self.mw.log.append(f"👀 New file detected: {os.path.basename(event.src_path)} — processing…")
                    task = ProcessTask([event.src_path], self.cfg)
                    task.signals.file_done.connect(lambda p: self.mw.log.append(f"✔ Auto: {os.path.basename(p)}"))
                    task.signals.error.connect(lambda m: self.mw.log.append(f"❌ Auto: {m}"))
                    self.mw.threadpool.start(task)
        self._watch_dir = folder
        observer = Observer()
        observer.schedule(Handler(self, cfg), folder, recursive=True)
        observer.start()
        self._watcher = observer
        self.log.append(f"👀 Watching folder: {folder}")

# ------------- main -------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
