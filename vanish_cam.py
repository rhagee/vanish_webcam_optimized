"""Webcam locale: Naruto Vanish avviato unendo i palmi davanti alla webcam."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from backgrounds import (
    BackgroundPanel, fit_background, import_image, list_backgrounds, pick_image_file,
    read_image, remember_background, remembered_background, save_capture,
)
from global_hotkeys import GlobalHotkeys
from hand_debug import HandDebugView
from naruto_effect import (
    EFFECT_SECONDS, PERSON_FADE_SECONDS, hand_points, joined_hands_check,
    load_log, render_substitution, substitution_anchor,
)
from vanish_sound import SOUND_PATH, VanishSound


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Anteprima"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
)
HAND_MODEL = APP_DIR / "hand_landmarker.task"

CAPTURE_DELAY_SECONDS = 3.0  # Time to step out of the frame.
CAPTURE_SECONDS = 3.0
CAPTURE_SAMPLE_SECONDS = 0.15
HOLD_SECONDS = 0.2  # Palms must stay joined this long after first recognition.
HOLD_GRACE_SECONDS = 0.2  # Tolerate missed detections while holding.
# Region where joined hands usually are; retried zoomed-in when the full frame misses them.
HAND_CROP = (0.18, 1.0, 0.12, 0.88)  # top, bottom, left, right (fractions)


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("Gli FPS devono essere compresi tra 1 e 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naruto Vanish per la webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Indice della webcam")
    parser.add_argument("--width", type=int, default=960, help="Larghezza video")
    parser.add_argument("--height", type=int, default=540, help="Altezza video")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Avvia a schermo intero"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Non specchiare la webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="FPS della webcam e dell'uscita virtuale (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Richiede la webcam virtuale; per default si attiva se disponibile. --no-virtual-camera la disattiva",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Driver della webcam virtuale da usare",
    )
    parser.add_argument("--virtual-device", help="Nome del dispositivo virtuale da usare")
    parser.add_argument(
        "--debug", "--gesture-debug", dest="debug", action="store_true",
        help="Apre la finestra Debug mani (tasto D) per vedere ed etichettare il riconoscimento",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "Webcam virtuale non disponibile. Installa OBS Studio (con il driver "
            "OBS Virtual Camera) oppure Unity Capture. Se usi OBS Virtual Camera, "
            "ferma 'Avvia fotocamera virtuale' in OBS. "
            "Avvia da AVVIA_LOCALE.bat per installare pyvirtualcam. "
            f"Dettaglio: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Proseguo con l'anteprima locale.")
        return None
    print(f"[Webcam] Uscita virtuale attiva: {camera.device} ({width}x{height}, {args.fps:g} FPS)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def download_models() -> None:
    if SEGMENTER_MODEL.exists() and SEGMENTER_MODEL.stat().st_size > 100_000:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = SEGMENTER_MODEL.with_suffix(SEGMENTER_MODEL.suffix + ".part")
    print(f"[Vanish Lab] Scarico {SEGMENTER_MODEL.name}...")
    try:
        request = urllib.request.Request(SEGMENTER_URL, headers={"User-Agent": "Vanish-Lab/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 256):
                output.write(chunk)
        os.replace(temporary, SEGMENTER_MODEL)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Non riesco a scaricare il modello MediaPipe. "
            "Controlla la connessione internet e riprova."
        ) from error


def open_camera(index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} non disponibile. Chiudi le altre app oppure prova --camera 1."
        )

    # Without MJPG many USB webcams fall back to raw YUY2, which caps
    # 960x540/720p at 5-15 FPS over DirectShow.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_tasks() -> tuple[vision.ImageSegmenter, vision.HandLandmarker, vision.HandLandmarker]:
    """Segmenter, tracking hand model (VIDEO) and zoomed-crop hand model (IMAGE)."""
    if not HAND_MODEL.is_file():
        raise RuntimeError(f"Modello delle mani non trovato: {HAND_MODEL}")
    with ExitStack() as cleanup:
        # The segmenter only runs for a handful of frames per vanish, so IMAGE mode.
        segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        ))
        cleanup.callback(segmenter.close)
        tasks = [segmenter]
        for mode in (vision.RunningMode.VIDEO, vision.RunningMode.IMAGE):
            hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
                running_mode=mode,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.45,
            ))
            cleanup.callback(hands.close)
            tasks.append(hands)
        cleanup.pop_all()  # Success: the caller owns them now.
    return tuple(tasks)


def to_mp_image(frame: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def person_mask(segmenter: vision.ImageSegmenter, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    result = segmenter.segment(to_mp_image(frame))
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    # Soften at model resolution; it is much cheaper than blurring the full frame.
    raw = cv2.GaussianBlur(raw, (0, 0), sigmaX=0.6)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


@dataclass
class HandPass:
    """One run of a hand model: full frame or zoomed crop (points in frame pixels)."""
    name: str
    rect: tuple[int, int, int, int]
    hands: list[np.ndarray]
    check: dict


@dataclass
class HandResult:
    seen_at: float
    joined: bool
    count: int
    frame: np.ndarray | None = None
    passes: list[HandPass] = field(default_factory=list)


class HandWatcher:
    """Runs the hand models on a worker thread so the video never waits for them.

    The main loop submits its latest frame; the worker always analyses the most
    recent one (older frames are skipped) and publishes a HandResult.
    """

    def __init__(self, hands, zoomed_hands=None):
        self.hands = hands
        self.zoomed_hands = zoomed_hands
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.frame: np.ndarray | None = None
        self.frame_time = 0.0
        self.result: HandResult | None = None
        self.last_timestamp_ms = -1
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame: np.ndarray, now: float) -> None:
        with self.lock:
            self.frame, self.frame_time = frame, now
        self.wake.set()

    def take_result(self) -> HandResult | None:
        with self.lock:
            result, self.result = self.result, None
        return result

    def detect(self, frame: np.ndarray, now: float) -> HandResult:
        # VIDEO mode needs strictly increasing timestamps and tracks hands
        # between calls, so it only runs the palm detector when needed.
        timestamp_ms = max(int(now * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        height, width = frame.shape[:2]
        found = hand_points(
            self.hands.detect_for_video(to_mp_image(frame), timestamp_ms).hand_landmarks,
            width, height)
        passes = [HandPass("intera", (0, 0, width, height), found, joined_hands_check(found))]
        if not passes[0].check["joined"] and self.zoomed_hands is not None:
            # Pressed-together palms are small and easy to miss in the full frame.
            top, bottom, left, right = HAND_CROP
            x1, y1, x2, y2 = int(width * left), int(height * top), int(width * right), int(height * bottom)
            zoomed = hand_points(
                self.zoomed_hands.detect(to_mp_image(frame[y1:y2, x1:x2])).hand_landmarks,
                x2 - x1, y2 - y1)
            zoomed = [hand + (x1, y1) for hand in zoomed]  # Back to full-frame pixels.
            passes.append(HandPass("zoom", (x1, y1, x2, y2), zoomed, joined_hands_check(zoomed)))
        return HandResult(
            seen_at=now,
            joined=any(p.check["joined"] for p in passes),
            count=max(len(p.hands) for p in passes),
            frame=frame,
            passes=passes,
        )

    def _run(self) -> None:
        while True:
            self.wake.wait()
            self.wake.clear()
            if not self.running:
                return
            with self.lock:
                frame, now = self.frame, self.frame_time
            try:
                result = self.detect(frame, now)
            except Exception as error:  # Keep watching; one bad frame must not stop it.
                print(f"[Mani] Errore di riconoscimento: {error}")
                continue
            with self.lock:
                self.result = result

    def close(self) -> None:
        self.running = False
        self.wake.set()
        self.thread.join(2.0)


class NarutoVanish:
    """Per-frame state machine: (capturing) -> ready -> vanishing -> vanished."""

    def __init__(self, segmenter, watcher, sound, sprite: np.ndarray, on_captured=None):
        self.segmenter = segmenter
        self.watcher = watcher
        self.sound = sound
        self.sprite = sprite
        self.on_captured = on_captured
        self.debug_view = None  # HandDebugView while debug mode is on.
        self.phase = "waiting"  # No background yet.
        self.phase_started = 0.0
        self.now = 0.0
        self.background: np.ndarray | None = None
        self.samples: list[np.ndarray] = []
        self.last_sample = 0.0
        self.anchor: tuple[int, int, int] | None = None
        self.vanished_frame: np.ndarray | None = None
        self.joined_since: float | None = None
        self.last_joined = 0.0

    def set_background(self, image: np.ndarray) -> None:
        self.background = image
        self.phase = "ready"
        self.vanished_frame = None
        self.joined_since = None

    def capture_background(self, now: float) -> None:
        self.phase = "capturing"
        self.phase_started = now
        self.samples.clear()

    def reappear(self) -> None:
        if self.background is not None:
            self.set_background(self.background)

    def vanish(self, frame: np.ndarray, now: float) -> None:
        if self.background is None:
            return
        self.anchor = substitution_anchor(person_mask(self.segmenter, frame))
        self.phase = "vanishing"
        self.phase_started = now
        self.vanished_frame = None
        self.sound.play()

    def status(self) -> str:
        """One-line state for the terminal and the Sfondi window (never the video)."""
        if self.phase == "capturing":
            elapsed = self.now - self.phase_started
            if elapsed < CAPTURE_DELAY_SECONDS:
                left = int(np.ceil(CAPTURE_DELAY_SECONDS - elapsed))
                return f"Esci dall'inquadratura: cattura dello sfondo tra {left} s"
            left = int(np.ceil(CAPTURE_DELAY_SECONDS + CAPTURE_SECONDS - elapsed))
            return f"Cattura dello sfondo in corso... {left} s"
        if self.phase == "waiting":
            return "Nessuno sfondo: premi Cattura sfondo o scegline uno"
        if self.phase == "ready":
            return "Pronto: unisci i palmi per svanire"
        return "Sparito: Ctrl+Alt+R per riapparire"

    def process(self, frame: np.ndarray, now: float) -> np.ndarray:
        """Return the frame to show; never draws text or UI on the video."""
        self.now = now
        if self.background is not None and self.background.shape != frame.shape:
            # Custom images (or a changed webcam resolution) are fitted once.
            self.background = fit_background(self.background, frame.shape[1], frame.shape[0])
            self.vanished_frame = None
        age = now - self.phase_started
        if self.phase == "capturing":
            self._capture(frame, now, age)
            return frame
        if self.phase == "waiting":
            return frame
        if self.phase == "ready":
            self.watcher.submit(frame, now)
            result = self.watcher.take_result()
            if result is None:
                return frame
            hold = self._hold_progress(result)
            if self.debug_view is not None:
                self.debug_view.update(result, hold)
            if hold >= 1.0:
                self.vanish(frame, now)
                if self.debug_view is not None:
                    self.debug_view.freeze("attivato")  # Label what just fired the effect.
            return frame
        if self.vanished_frame is None:  # Vanishing, or vanished but needs a redraw.
            mask = person_mask(self.segmenter, frame) if age < PERSON_FADE_SECONDS else None
            output = render_substitution(frame, self.background, mask,
                                         self.sprite, self.anchor, age)
            if age >= EFFECT_SECONDS:
                self.phase = "vanished"
                self.vanished_frame = output
            return output
        return self.vanished_frame  # Static: background with the landed log.

    def _capture(self, frame: np.ndarray, now: float, elapsed: float) -> None:
        if elapsed < CAPTURE_DELAY_SECONDS:
            return
        if not self.samples or now - self.last_sample >= CAPTURE_SAMPLE_SECONDS:
            self.samples.append(frame.copy())
            self.last_sample = now
        if elapsed >= CAPTURE_DELAY_SECONDS + CAPTURE_SECONDS:
            # The median removes anything that moved during the capture.
            background = np.median(np.stack(self.samples), axis=0).astype(np.uint8)
            self.samples.clear()
            self.set_background(background)
            if self.on_captured is not None:
                self.on_captured(background)

    def _hold_progress(self, result: HandResult) -> float:
        """Update the hold timer (by frame time); 1.0 means the gesture fires."""
        seen_at = result.seen_at
        if result.joined:
            if self.joined_since is None:
                self.joined_since = seen_at
            self.last_joined = seen_at
        elif self.joined_since is not None and seen_at - self.last_joined > HOLD_GRACE_SECONDS:
            self.joined_since = None
        if self.joined_since is None:
            return 0.0
        return min(1.0, (seen_at - self.joined_since) / HOLD_SECONDS)


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    required = (SEGMENTER_MODEL, HAND_MODEL, SOUND_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Risorse mancanti: {', '.join(missing)}")
    load_log()
    import pygame.mixer  # noqa: F401  The distributed build must contain its MP3 decoder.
    for task in create_tasks():
        task.close()
    print("[Webcam] Self-test completato: modelli, asset e audio presenti.")
    return 0


def run() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    with ExitStack() as resources:
        return run_camera(args, resources)


KEY_ACTIONS = {  # Only while the preview is focused; see global_hotkeys for the rest.
    ord(" "): "capture", ord("b"): "capture", ord("u"): "upload",
    ord("r"): "reappear", ord("v"): "vanish", ord("f"): "fullscreen", ord("d"): "debug",
}


def run_camera(args: argparse.Namespace, resources: ExitStack) -> int:
    download_models()
    capture = open_camera(args.camera, args.width, args.height, args.fps)
    resources.callback(capture.release)
    segmenter, hands, zoomed_hands = create_tasks()
    resources.callback(segmenter.close)
    resources.callback(hands.close)
    resources.callback(zoomed_hands.close)
    watcher = HandWatcher(hands, zoomed_hands)
    resources.callback(watcher.close)  # Runs first: stop the thread before closing models.
    sound = VanishSound()
    resources.callback(sound.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    fullscreen = args.fullscreen
    if fullscreen:
        set_fullscreen(True)
    hotkeys = GlobalHotkeys()
    resources.callback(hotkeys.close)
    panel = BackgroundPanel()
    files = list_backgrounds()
    selected = remembered_background()

    def select(path) -> None:
        nonlocal selected
        image = read_image(path)
        if image is None:
            print(f"[Sfondi] Immagine non leggibile: {path}")
            return
        app.set_background(image)
        remember_background(path)
        selected = path

    def captured(image: np.ndarray) -> None:
        nonlocal files
        select(save_capture(image))
        files = list_backgrounds()

    app = NarutoVanish(segmenter, watcher, sound, load_log(), on_captured=captured)
    if args.debug:
        app.debug_view = HandDebugView()
    if selected is not None:
        select(selected)
    if app.background is None:
        selected = None
        app.capture_background(time.perf_counter())
    print("[Webcam] Tasti nella finestra: B/SPAZIO cattura sfondo, U carica immagine, "
          "R riappari, V svanisci, D debug mani, F fullscreen, Q esci.")
    virtual_camera = None
    first_frame = True
    last_status = None

    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("La webcam ha smesso di restituire immagini.")
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        if first_frame:
            # Open with the resolution the webcam actually delivers.
            virtual_camera = open_virtual_camera(args, frame.shape[1], frame.shape[0])
            if virtual_camera is not None:
                resources.callback(virtual_camera.close)
            first_frame = False

        now = time.perf_counter()
        display = app.process(frame, now)
        # Frames are paced by the webcam itself; no extra sleep is needed.
        if virtual_camera is not None:
            send_virtual_frame(virtual_camera, display)
        cv2.imshow(WINDOW_NAME, display)
        status = app.status()
        if status != last_status:
            print(f"[Webcam] {status}")
            last_status = status
        panel.show(files, selected, status)
        if app.debug_view is not None:
            app.debug_view.tick(now)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        actions = [(action, None) for action in hotkeys.poll()]
        if key in KEY_ACTIONS:
            actions.append((KEY_ACTIONS[key], None))
        if (clicked := panel.poll()) is not None:
            actions.append(clicked)
        for action, path in actions:
            if action == "capture":
                app.capture_background(now)
            elif action == "reappear":
                app.reappear()
            elif action == "vanish":
                app.vanish(frame, now)
            elif action == "select":
                select(path)
            elif action == "upload":
                source = pick_image_file()
                if source is not None:
                    try:
                        select(import_image(source))
                    except (RuntimeError, OSError) as error:
                        print(f"[Sfondi] {error}")
                    files = list_backgrounds()
            elif action == "debug":
                if app.debug_view is None:
                    app.debug_view = HandDebugView()
                else:
                    app.debug_view.close()
                    app.debug_view = None
            elif action == "fullscreen":
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[Vanish Lab] ERRORE: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
