"""Webcam locale: Naruto Vanish avviato unendo i palmi davanti alla webcam."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
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
from naruto_effect import (
    EFFECT_SECONDS, PERSON_FADE_SECONDS, joined_hands_from_landmarks,
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
HAND_CHECK_SECONDS = 0.12  # ~8 checks per second is plenty for a held pose.
HOLD_SECONDS = 0.5  # Palms must stay joined this long before the effect fires.
HOLD_GRACE_SECONDS = 0.25  # Tolerate a missed detection while holding.


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
        "--gesture-debug", action="store_true",
        help="Stampa nel terminale lo stato del riconoscimento (senza scritte nel video)",
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


def create_tasks() -> tuple[vision.ImageSegmenter, vision.HandLandmarker]:
    if not HAND_MODEL.is_file():
        raise RuntimeError(f"Modello delle mani non trovato: {HAND_MODEL}")
    # The segmenter only runs for a handful of frames per vanish, so IMAGE mode.
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    ))
    try:
        hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        ))
    except Exception:
        segmenter.close()
        raise
    return segmenter, hands


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


class NarutoVanish:
    """Per-frame state machine: (capturing) -> ready -> vanishing -> vanished."""

    def __init__(self, segmenter, hands, sound, sprite: np.ndarray, debug: bool = False,
                 on_captured=None):
        self.segmenter = segmenter
        self.hands = hands
        self.sound = sound
        self.sprite = sprite
        self.debug = debug
        self.on_captured = on_captured
        self.phase = "waiting"  # No background yet.
        self.phase_started = 0.0
        self.now = 0.0
        self.background: np.ndarray | None = None
        self.samples: list[np.ndarray] = []
        self.last_sample = 0.0
        self.anchor: tuple[int, int, int] | None = None
        self.vanished_frame: np.ndarray | None = None
        self.last_timestamp_ms = -1
        self.last_hand_check = -HAND_CHECK_SECONDS
        self.joined_since: float | None = None
        self.last_joined = 0.0
        self.last_debug = 0.0

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
            return "Pronto: tieni i palmi uniti per mezzo secondo"
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
            if now - self.last_hand_check >= HAND_CHECK_SECONDS:
                self.last_hand_check = now
                if self._gesture_held(frame, now):
                    self.vanish(frame, now)
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

    def _gesture_held(self, frame: np.ndarray, now: float) -> bool:
        """True once the palms have stayed joined for HOLD_SECONDS."""
        if self._hands_joined(frame, now):
            if self.joined_since is None:
                self.joined_since = now
            self.last_joined = now
        elif self.joined_since is not None and now - self.last_joined > HOLD_GRACE_SECONDS:
            self.joined_since = None
        return self.joined_since is not None and now - self.joined_since >= HOLD_SECONDS

    def _hands_joined(self, frame: np.ndarray, now: float) -> bool:
        # VIDEO mode needs strictly increasing timestamps and tracks hands
        # between calls, so it only runs the palm detector when needed.
        timestamp_ms = max(int(now * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        found = self.hands.detect_for_video(to_mp_image(frame), timestamp_ms).hand_landmarks
        height, width = frame.shape[:2]
        joined = joined_hands_from_landmarks(found, width, height)
        if self.debug and now - self.last_debug >= 1.0:
            print(f"[Mani] rilevate={len(found)}, unite={joined}")
            self.last_debug = now
        return joined


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
    segmenter, hands = create_tasks()
    hands.close()
    segmenter.close()
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
    ord("r"): "reappear", ord("v"): "vanish", ord("f"): "fullscreen",
}


def run_camera(args: argparse.Namespace, resources: ExitStack) -> int:
    download_models()
    capture = open_camera(args.camera, args.width, args.height, args.fps)
    resources.callback(capture.release)
    segmenter, hands = create_tasks()
    resources.callback(segmenter.close)
    resources.callback(hands.close)
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

    app = NarutoVanish(segmenter, hands, sound, load_log(),
                       debug=args.gesture_debug, on_captured=captured)
    if selected is not None:
        select(selected)
    if app.background is None:
        selected = None
        app.capture_background(time.perf_counter())
    print("[Webcam] Tasti nella finestra: B/SPAZIO cattura sfondo, U carica immagine, "
          "R riappari, V svanisci, F fullscreen, Q esci.")
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
