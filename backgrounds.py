"""Saved background library, remembered selection and the "Sfondi" control window."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# Next to the .exe when frozen (PyInstaller unpacks code to a temp dir).
DATA_DIR = (Path(sys.executable).parent if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent)
BACKGROUND_DIR = DATA_DIR / "backgrounds"
SETTINGS_PATH = DATA_DIR / "settings.json"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
PANEL_NAME = "Sfondi"


def read_image(path: Path) -> np.ndarray | None:
    # imdecode instead of imread: imread fails on non-ASCII Windows paths.
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def list_backgrounds() -> list[Path]:
    """Saved backgrounds, newest first."""
    if not BACKGROUND_DIR.is_dir():
        return []
    files = [p for p in BACKGROUND_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _new_path(prefix: str, suffix: str) -> Path:
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = BACKGROUND_DIR / f"{prefix}_{stamp}{suffix}"
    counter = 1
    while path.exists():
        path = BACKGROUND_DIR / f"{prefix}_{stamp}_{counter}{suffix}"
        counter += 1
    return path


def save_capture(image: np.ndarray) -> Path:
    path = _new_path("cattura", ".png")
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Impossibile salvare lo sfondo catturato.")
    data.tofile(str(path))
    return path


def import_image(source: Path) -> Path:
    """Copy a user image into the library (re-encoded as PNG if not PNG/JPG)."""
    image = read_image(source)
    if image is None:
        raise RuntimeError(f"Immagine non leggibile: {source}")
    suffix = source.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        path = _new_path("immagine", suffix)
        shutil.copyfile(source, path)
    else:
        path = _new_path("immagine", ".png")
        cv2.imencode(".png", image)[1].tofile(str(path))
    return path


def pick_image_file() -> Path | None:
    """Native file dialog; blocks the video for as long as it is open."""
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        name = filedialog.askopenfilename(
            parent=root, title="Scegli un'immagine di sfondo",
            filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp *.webp"), ("Tutti i file", "*.*")],
        )
    finally:
        root.destroy()
    return Path(name) if name else None


def load_settings() -> dict:
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return settings if isinstance(settings, dict) else {}


def save_settings(**changes) -> None:
    """Merge `changes` into settings.json (background, permanent, ...)."""
    try:
        SETTINGS_PATH.write_text(json.dumps({**load_settings(), **changes}), encoding="utf-8")
    except OSError as error:
        print(f"[Sfondi] Impossibile salvare le impostazioni: {error}")


def remembered_background() -> Path | None:
    name = load_settings().get("background")
    if not isinstance(name, str):
        return None
    path = BACKGROUND_DIR / Path(name).name
    return path if path.is_file() else None


def remember_background(path: Path) -> None:
    save_settings(background=path.name)


def fit_background(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to cover the frame and crop the centre, keeping the aspect ratio."""
    if image.shape[:2] == (height, width):
        return image
    scale = max(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(width, round(image.shape[1] * scale)),
                                 max(height, round(image.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top:top + height, left:left + width]


class BackgroundPanel:
    """Small OpenCV window with buttons and a clickable thumbnail grid.

    Clicks are queued and handled by the main loop, so the file dialog and
    disk work never run inside the OpenCV mouse callback.
    """

    WIDTH = 720
    THUMB = (160, 90)
    COLUMNS = 4
    MAX_THUMBS = 16
    GAP = 12
    BUTTONS = (("capture", "Cattura sfondo [B]", 12), ("upload", "Carica immagine [U]", 250),
               ("permanent", "Sfondo fisso: {} [P]", 488))
    BUTTON_TOP, BUTTON_HEIGHT, BUTTON_WIDTH = 16, 52, 220
    GRID_TOP = 110

    def __init__(self):
        self.pending: tuple[str, Path | None] | None = None
        self.thumbs: dict[Path, tuple[float, np.ndarray]] = {}
        self.cells: list[tuple[Path, int, int]] = []
        self.drawn_state = None
        cv2.namedWindow(PANEL_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(PANEL_NAME, self._on_mouse)

    def poll(self) -> tuple[str, Path | None] | None:
        action, self.pending = self.pending, None
        return action

    def _on_mouse(self, event, x, y, flags, userdata) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        top = self.BUTTON_TOP
        for action, _, left in self.BUTTONS:
            if left <= x <= left + self.BUTTON_WIDTH and top <= y <= top + self.BUTTON_HEIGHT:
                self.pending = (action, None)
                return
        for path, left, top in self.cells:
            if left <= x <= left + self.THUMB[0] and top <= y <= top + self.THUMB[1]:
                self.pending = ("select", path)
                return

    def _thumbnail(self, path: Path) -> np.ndarray | None:
        mtime = path.stat().st_mtime
        cached = self.thumbs.get(path)
        if cached is None or cached[0] != mtime:
            image = read_image(path)
            if image is None:
                return None
            cached = (mtime, fit_background(image, *self.THUMB).copy())
            self.thumbs[path] = cached
        return cached[1]

    def show(self, files: list[Path], selected: Path | None, status: str,
             permanent: bool = False) -> None:
        """Redraw only when something visible changed."""
        files = files[:self.MAX_THUMBS]
        state = (tuple(files), selected, status, permanent)
        if state == self.drawn_state:
            return
        self.drawn_state = state
        rows = max(1, -(-len(files) // self.COLUMNS))
        height = self.GRID_TOP + rows * (self.THUMB[1] + self.GAP) + self.GAP
        canvas = np.full((height, self.WIDTH, 3), (34, 29, 26), np.uint8)
        top = self.BUTTON_TOP
        for action, label, left in self.BUTTONS:
            corner = (left + self.BUTTON_WIDTH, top + self.BUTTON_HEIGHT)
            active = action == "permanent" and permanent
            cv2.rectangle(canvas, (left, top), corner, (60, 140, 60) if active else (62, 54, 48), -1)
            cv2.rectangle(canvas, (left, top), corner, (153, 142, 111), 2)
            cv2.putText(canvas, label.format("ON" if permanent else "OFF"), (left + 12, top + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (247, 245, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, status, (20, 94), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (201, 195, 179), 1, cv2.LINE_AA)
        if not files:
            cv2.putText(canvas, "Nessuno sfondo salvato.", (20, self.GRID_TOP + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
        self.cells = []
        for index, path in enumerate(files):
            thumb = self._thumbnail(path)
            if thumb is None:
                continue
            left = self.GAP + (index % self.COLUMNS) * (self.THUMB[0] + 20)
            cell_top = self.GRID_TOP + (index // self.COLUMNS) * (self.THUMB[1] + self.GAP)
            canvas[cell_top:cell_top + self.THUMB[1], left:left + self.THUMB[0]] = thumb
            if path == selected:
                cv2.rectangle(canvas, (left - 3, cell_top - 3),
                              (left + self.THUMB[0] + 2, cell_top + self.THUMB[1] + 2),
                              (92, 228, 255), 3)
            self.cells.append((path, left, cell_top))
        cv2.imshow(PANEL_NAME, canvas)
