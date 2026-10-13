"""'Hand debug' window: shows what the hand models see and records labeled samples.

Nothing here touches the preview or the virtual camera. Labeled samples go to
debug/dataset.jsonl (one JSON object per line) plus the raw frame as JPG, so the
joined-hands rule can be tuned against real examples.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque

import cv2
import numpy as np

from backgrounds import DATA_DIR
from naruto_effect import JOIN_RULES

DEBUG_NAME = "Hand debug"
DEBUG_DIR = DATA_DIR / "debug"
DATASET_PATH = DEBUG_DIR / "dataset.jsonl"
VIEW_WIDTH = 640
PANEL_HEIGHT = 300
FREEZE_DELAY_SECONDS = 3.0
RECENT_SECONDS = 2.0
ONE_HAND = "only 1 hand"

# MediaPipe hand skeleton.
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
PASS_COLORS = {"full": (90, 220, 90), "zoom": (0, 170, 255)}
GREEN, RED, WHITE, GREY = (60, 190, 60), (50, 50, 210), (240, 240, 240), (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(image, text, origin, scale=0.5, color=WHITE, thickness=1):
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def blocking_rules(result) -> list[str]:
    """Rules failed by the closest pass (fewest failures); ONE_HAND if no pass saw two."""
    if result.joined:
        return []
    failures = [
        [name for name, limit, below in JOIN_RULES
         if not ((p.check["values"][name] < limit) if below else (p.check["values"][name] > limit))]
        for p in result.passes if p.check["values"]
    ]
    return min(failures, key=len) if failures else [ONE_HAND]


class HandDebugView:
    """Live view that freezes (on detection or click) until labeled YES / NO."""

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
        self.recent: deque = deque()  # (seen_at, joined, blocking rules) per result.
        cv2.namedWindow(DEBUG_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(DEBUG_NAME, self._on_mouse)
        print(f"[Debug] '{DEBUG_NAME}' window open. Samples are saved to {DEBUG_DIR}")

    def update(self, result, hold: float) -> None:
        if not self.frozen:
            self.result, self.hold, self.dirty = result, hold, True
            self.recent.append((result.seen_at, result.joined, blocking_rules(result)))
            while self.recent and result.seen_at - self.recent[0][0] > RECENT_SECONDS:
                self.recent.popleft()

    def recent_summary(self) -> dict:
        """Share of recognized frames in the last RECENT_SECONDS and what blocked the rest."""
        total = len(self.recent)
        if not total:
            return {"frames": 0, "ok": 0.0, "blocked_by": {}}
        counts = Counter(rule for _, _, rules in self.recent for rule in rules)
        return {
            "frames": total,
            "ok": sum(joined for _, joined, _ in self.recent) / total,
            "blocked_by": {rule: round(n / total, 2) for rule, n in counts.most_common()},
        }

    def freeze(self, reason: str) -> None:
        if self.result is not None:
            self.frozen, self.reason, self.freeze_at, self.dirty = True, reason, None, True

    def resume(self) -> None:
        self.frozen, self.freeze_at, self.dirty = False, None, True

    def answer(self, is_naruto: bool) -> None:
        """YES / NO (buttons or Y / N keys); only a frozen frame can be answered."""
        if self.frozen:
            self.label(is_naruto)

    def _on_mouse(self, event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frozen:
                for is_naruto, (x1, y1, x2, y2) in self.buttons.items():
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.answer(is_naruto)
                        return
            elif y < self.image_height:
                self.freeze("manual")
        elif event == cv2.EVENT_RBUTTONDOWN and not self.frozen:
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
        name = f"{'yes' if is_naruto else 'no'}_{stamp}"
        ok, jpg = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            jpg.tofile(str(DEBUG_DIR / f"{name}.jpg"))
        height, width = result.frame.shape[:2]
        record = {
            "image": f"{name}.jpg",
            "label": is_naruto,
            "detected": result.joined,
            "reason": self.reason,  # manual / timer / detected (algorithm fired)
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_size": [width, height],
            "hold": round(self.hold, 3),
            "last_2s": self.recent_summary(),  # How steady recognition was before the freeze.
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
        verdict = "right" if is_naruto == result.joined else "WRONG"
        self.message = f"Saved #{count} as {'YES' if is_naruto else 'NO'} (algorithm {verdict})"
        print(f"[Debug] {self.message}: {name}.jpg")
        self.resume()

    def render(self, now: float) -> np.ndarray:
        if self.result is None or self.result.frame is None:
            canvas = np.full((360 + PANEL_HEIGHT, VIEW_WIDTH, 3), 30, np.uint8)
            _text(canvas, "Waiting for hand recognition (needs a background to be ready)...",
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
            if p.name != "full":
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
        _text(view, "OK - detected" if result.joined else "NO - not detected",
              (10, 24), 0.7, WHITE, 2)
        if self.frozen:
            _text(view, f"FROZEN ({self.reason})", (VIEW_WIDTH - 190, 24), 0.6, (0, 220, 255), 2)
        elif self.freeze_at is not None:
            left = int(np.ceil(self.freeze_at - now))
            _text(view, f"Photo in {left} s", (VIEW_WIDTH - 150, 24), 0.7, (0, 220, 255), 2)

        panel = np.full((PANEL_HEIGHT, VIEW_WIDTH, 3), 30, np.uint8)
        y = 24
        for p in result.passes:
            color = PASS_COLORS.get(p.name, WHITE)
            _text(panel, f"{p.name}: hands {len(p.hands)}", (12, y), 0.5, color)
            for index, (name, limit, below) in enumerate(JOIN_RULES):
                value = p.check["values"].get(name)
                if value is None:
                    text, passed = f"{name} -", False
                else:
                    passed = value < limit if below else value > limit
                    text = f"{name} {value:.2f}{'<' if below else '>'}{limit:g}"
                column, row = index % 3, index // 3
                _text(panel, text, (125 + column * 172, y + row * 20), 0.42,
                      GREEN if passed else RED)
            y += 20 * ((len(JOIN_RULES) + 2) // 3) + 8
        cv2.rectangle(panel, (12, y), (12 + 200, y + 12), GREY, 1)
        cv2.rectangle(panel, (12, y), (12 + int(200 * self.hold), y + 12), GREEN, -1)
        _text(panel, "hold before the effect", (222, y + 11), 0.42, GREY)
        y += 34
        summary = self.recent_summary()
        if summary["frames"]:
            blocked = ", ".join(f"{rule} {share:.0%}"
                                for rule, share in list(summary["blocked_by"].items())[:3])
            _text(panel, f"Last 2 s: recognized {summary['ok']:.0%}"
                         + (f"  |  blocked by: {blocked}" if blocked else ""),
                  (12, y), 0.45, (0, 220, 255))
        y += 22

        self.buttons = {}
        if self.frozen:
            _text(panel, "Is this the right Naruto pose?", (12, y + 4), 0.55, WHITE)
            y += 14
            for is_naruto, label, left, color in ((True, "YES - Naruto", 12, GREEN),
                                                  (False, "NO - not it", 330, RED)):
                top = y
                cv2.rectangle(panel, (left, top), (left + 298, top + 46), color, -1)
                _text(panel, label, (left + 60, top + 31), 0.7, WHITE, 2)
                offset = self.image_height
                self.buttons[is_naruto] = (left, top + offset, left + 298, top + 46 + offset)
            y += 62
        else:
            y += 6
        hint = ("Choose with the buttons or the Y / N keys" if self.frozen else
                "Click the image: freeze now  |  Right-click: freeze in 3 s")
        _text(panel, hint, (12, min(y, PANEL_HEIGHT - 30)), 0.45, GREY)
        if self.message:
            _text(panel, self.message, (12, PANEL_HEIGHT - 10), 0.45, (0, 220, 255))
        return np.vstack([view, panel])

    def close(self) -> None:
        try:
            cv2.destroyWindow(DEBUG_NAME)
        except cv2.error:
            pass
