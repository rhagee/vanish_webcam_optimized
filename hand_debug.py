"""'Debug mani' window: shows what the hand models see and records labeled samples.

Nothing here touches the preview or the virtual camera. Labeled samples go to
debug/dataset.jsonl (one JSON object per line) plus the raw frame as JPG, so the
joined-hands rule can be tuned against real examples.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

from backgrounds import DATA_DIR
from naruto_effect import JOIN_RULES

DEBUG_NAME = "Debug mani"
DEBUG_DIR = DATA_DIR / "debug"
DATASET_PATH = DEBUG_DIR / "dataset.jsonl"
VIEW_WIDTH = 640
PANEL_HEIGHT = 230
FREEZE_DELAY_SECONDS = 3.0

# MediaPipe hand skeleton.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
PASS_COLORS = {"intera": (90, 220, 90), "zoom": (0, 170, 255)}
GREEN, RED, WHITE, GREY = (60, 190, 60), (50, 50, 210), (240, 240, 240), (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(image, text, origin, scale=0.5, color=WHITE, thickness=1):
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


class HandDebugView:
    """Live view; click the image to freeze, then label it with SI / NO."""

    def __init__(self):
        self.result = None
        self.hold = 0.0
        self.frozen = False
        self.reason = ""
        self.freeze_at: float | None = None
        self.message = ""
        self.buttons: dict[bool, tuple[int, int, int, int]] = {}
        self.image_height = 0
        self.dirty = True
        self.last_countdown = None
        cv2.namedWindow(DEBUG_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(DEBUG_NAME, self._on_mouse)
        print(f"[Debug] Finestra '{DEBUG_NAME}' aperta. Campioni salvati in {DEBUG_DIR}")

    def update(self, result, hold: float) -> None:
        if not self.frozen:
            self.result, self.hold, self.dirty = result, hold, True

    def freeze(self, reason: str) -> None:
        if self.result is not None:
            self.frozen, self.reason, self.freeze_at, self.dirty = True, reason, None, True

    def resume(self) -> None:
        self.frozen, self.freeze_at, self.dirty = False, None, True

    def _on_mouse(self, event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONUP:
            for label, (x1, y1, x2, y2) in self.buttons.items():
                if self.frozen and x1 <= x <= x2 and y1 <= y <= y2:
                    self.label(label)
                    return
            if y < self.image_height:
                if self.frozen:
                    self.resume()
                else:
                    self.freeze("manuale")
        elif event == cv2.EVENT_RBUTTONUP and not self.frozen:
            self.freeze_at = time.perf_counter() + FREEZE_DELAY_SECONDS
            self.dirty = True

    def tick(self, now: float) -> None:
        """Call once per frame: handles the delayed freeze and redraws if needed."""
        if self.freeze_at is not None:
            if now >= self.freeze_at:
                self.freeze("timer")
            else:
                countdown = int(np.ceil(self.freeze_at - now))
                self.dirty = self.dirty or countdown != self.last_countdown
                self.last_countdown = countdown
        if self.dirty:
            self.dirty = False
            cv2.imshow(DEBUG_NAME, self.render(now))

    def label(self, is_naruto: bool) -> None:
        """Save the frozen sample with the user's label, then resume live view."""
        result = self.result
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        name = f"{'si' if is_naruto else 'no'}_{stamp}"
        ok, jpg = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            jpg.tofile(str(DEBUG_DIR / f"{name}.jpg"))
        height, width = result.frame.shape[:2]
        record = {
            "image": f"{name}.jpg",
            "label": is_naruto,
            "detected": result.joined,
            "reason": self.reason,  # manuale / timer / attivato (effect fired)
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_size": [width, height],
            "hold": round(self.hold, 3),
            "rules": [list(rule) for rule in JOIN_RULES],
            "passes": [{
                "name": p.name,
                "rect": list(p.rect),
                "joined": p.check["joined"],
                "values": {k: round(v, 4) for k, v in p.check["values"].items()},
                "hands": [np.round(hand, 1).tolist() for hand in p.hands],
            } for p in result.passes],
        }
        with DATASET_PATH.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        count = DATASET_PATH.read_text(encoding="utf-8").count("\n")
        verdict = "corretto" if is_naruto == result.joined else "SBAGLIATO"
        self.message = f"Salvato #{count} come {'SI' if is_naruto else 'NO'} (algoritmo {verdict})"
        print(f"[Debug] {self.message}: {name}.jpg")
        self.resume()

    def render(self, now: float) -> np.ndarray:
        if self.result is None or self.result.frame is None:
            canvas = np.full((360 + PANEL_HEIGHT, VIEW_WIDTH, 3), 30, np.uint8)
            _text(canvas, "In attesa del riconoscimento mani (serve uno sfondo pronto)...",
                  (16, 180), 0.5, GREY)
            self.image_height = 360
            return canvas
        result = self.result
        height, width = result.frame.shape[:2]
        scale = VIEW_WIDTH / width
        view = cv2.resize(result.frame, (VIEW_WIDTH, round(height * scale)),
                          interpolation=cv2.INTER_AREA)
        self.image_height = view.shape[0]
        for p in result.passes:
            color = PASS_COLORS.get(p.name, WHITE)
            if p.name != "intera":
                x1, y1, x2, y2 = (round(v * scale) for v in p.rect)
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            for hand in p.hands:
                points = [tuple(int(v) for v in pt * scale) for pt in hand]
                for a, b in CONNECTIONS:
                    cv2.line(view, points[a], points[b], color, 2, cv2.LINE_AA)
                for point in points:
                    cv2.circle(view, point, 3, WHITE, -1, cv2.LINE_AA)

        # Verdict of the current algorithm for this exact frame.
        banner = GREEN if result.joined else RED
        cv2.rectangle(view, (0, 0), (250, 34), banner, -1)
        _text(view, "OK - rilevato" if result.joined else "NO - non rilevato",
              (10, 24), 0.7, WHITE, 2)
        if self.frozen:
            _text(view, f"FERMO ({self.reason})", (VIEW_WIDTH - 190, 24), 0.6, (0, 220, 255), 2)
        elif self.freeze_at is not None:
            left = int(np.ceil(self.freeze_at - now))
            _text(view, f"Foto tra {left} s", (VIEW_WIDTH - 150, 24), 0.7, (0, 220, 255), 2)

        panel = np.full((PANEL_HEIGHT, VIEW_WIDTH, 3), 30, np.uint8)
        y = 24
        for p in result.passes:
            color = PASS_COLORS.get(p.name, WHITE)
            _text(panel, f"{p.name}: mani {len(p.hands)}", (12, y), 0.5, color)
            x = 130
            for name, limit, below in JOIN_RULES:
                value = p.check["values"].get(name)
                if value is None:
                    text, passed = f"{name} -", False
                else:
                    passed = value < limit if below else value > limit
                    text = f"{name} {value:.2f}{'<' if below else '>'}{limit:g}"
                _text(panel, text, (x, y), 0.42, GREEN if passed else RED)
                x += 128
            y += 26
        cv2.rectangle(panel, (12, y), (12 + 200, y + 12), GREY, 1)
        cv2.rectangle(panel, (12, y), (12 + int(200 * self.hold), y + 12), GREEN, -1)
        _text(panel, "tenuta prima dell'effetto", (222, y + 11), 0.42, GREY)
        y += 34

        self.buttons = {}
        if self.frozen:
            _text(panel, "E' la posa Naruto giusta?", (12, y + 4), 0.55, WHITE)
            y += 14
            for is_naruto, label, left, color in ((True, "SI - e' Naruto", 12, GREEN),
                                                  (False, "NO - non lo e'", 330, RED)):
                top = y
                cv2.rectangle(panel, (left, top), (left + 298, top + 46), color, -1)
                _text(panel, label, (left + 60, top + 31), 0.7, WHITE, 2)
                offset = self.image_height
                self.buttons[is_naruto] = (left, top + offset, left + 298, top + 46 + offset)
            y += 62
        else:
            y += 6
        hint = ("Clic sull'immagine: riprendi senza salvare" if self.frozen else
                "Clic sull'immagine: blocca ora  |  Clic destro: blocca tra 3 s")
        _text(panel, hint, (12, min(y, PANEL_HEIGHT - 30)), 0.45, GREY)
        if self.message:
            _text(panel, self.message, (12, PANEL_HEIGHT - 10), 0.45, (0, 220, 255))
        return np.vstack([view, panel])

    def close(self) -> None:
        try:
            cv2.destroyWindow(DEBUG_NAME)
        except cv2.error:
            pass
