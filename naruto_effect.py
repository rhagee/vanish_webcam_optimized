"""Two-hand trigger and smoke/log compositing for body substitution."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


EFFECT_SECONDS = 1.45
PERSON_FADE_SECONDS = 0.13
LOG_MAX_HEIGHT = 720  # The source sprite is far larger than any on-screen log.
LOG_PATH = Path(__file__).resolve().parent / "assets" / "substitution_log.png"


# Joined-hands rule: (name, limit, True if the value must stay below the limit).
# Tuned on labeled samples from the "Hand debug" window: the old rule (only
# distances) accepted fists, hands lying flat on the desk and hands half out of
# frame; these limits rejected all 29 false positives and kept 11 of 15 real poses.
# Each limit sits about halfway between the real poses and the nearest false
# positive, leaving room for angled (oblique) views where the far hand looks smaller.
JOIN_RULES = (
    ("edge", 0.01, False),       # Nearest landmark to the frame edge / frame size.
    ("upright", 0.4, False),     # Wrist->middle knuckle pointing up (1 = vertical).
    ("finger", 1.05, False),     # Middle tip-to-wrist / knuckle-to-wrist (fist ~0.6).
    ("size", 2.0, True),         # Larger hand / smaller hand.
    ("palms", 1.42, True),       # Palm-centre distance / hand size.
    ("tips", 1.45, True),        # Fingertip-centre distance / hand size.
)
MIN_HAND_PIXELS = 8.0


def hand_points(hands, width: int, height: int) -> list[np.ndarray]:
    """Landmarks as (21, 2) pixel arrays, skipping incomplete hands."""
    return [np.array([(p.x * width, p.y * height) for p in hand], dtype=np.float64)
            for hand in hands if len(hand) == 21]


def joined_hands_check(points: list[np.ndarray], width: int, height: int) -> dict:
    """Measure the first two hands against JOIN_RULES; 'joined' is the verdict.

    Points are (21, 2) pixel arrays in a width x height frame. Landmark ids:
    0 wrist, 5/9/13/17 index/middle/ring/pinky knuckles, 8/12/16/20 their tips.
    """
    check = {"joined": False, "values": {}}
    if len(points) < 2 or not all(np.isfinite(hand).all() for hand in points[:2]):
        return check
    points = points[:2]
    scale = [max(float(np.linalg.norm(hand[5] - hand[17])),
                 float(np.linalg.norm(hand[0] - hand[9]))) for hand in points]
    if min(scale) < MIN_HAND_PIXELS:
        return check
    size = sum(scale) * 0.5
    every = np.vstack(points)
    palms = [hand[[0, 5, 9, 13, 17]].mean(axis=0) for hand in points]
    fingertips = [hand[[8, 12, 16, 20]].mean(axis=0) for hand in points]
    knuckle = [max(float(np.linalg.norm(hand[9] - hand[0])), 1e-6) for hand in points]
    values = {
        # The model invents landmarks for hands cut by the frame edge.
        "edge": float(min(every[:, 0].min() / width, every[:, 1].min() / height,
                           1 - every[:, 0].max() / width, 1 - every[:, 1].max() / height)),
        "upright": min(float(hand[0][1] - hand[9][1]) / k for hand, k in zip(points, knuckle)),
        "finger": min(float(np.linalg.norm(hand[12] - hand[0])) / k
                      for hand, k in zip(points, knuckle)),
        "size": max(scale) / min(scale),
        "palms": float(np.linalg.norm(palms[0] - palms[1])) / size,
        "tips": float(np.linalg.norm(fingertips[0] - fingertips[1])) / size,
    }
    check["values"] = values
    check["joined"] = all((values[name] < limit) if below else (values[name] > limit)
                          for name, limit, below in JOIN_RULES)
    return check


def joined_hands_from_landmarks(hands, width: int, height: int) -> bool:
    """Recognize two joined, upright hands with extended fingers (see JOIN_RULES)."""
    return joined_hands_check(hand_points(hands, width, height), width, height)["joined"]


def substitution_anchor(mask: np.ndarray) -> tuple[int, int, int]:
    """Place the log at the person's torso instead of at a fixed screen spot."""
    height, width = mask.shape
    binary = (mask > 0.45).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return width // 2, height // 2, int(height * 0.57)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[largest]
    size = int(np.clip(h * 0.58, height * 0.24, height * 0.70))
    center_x = int(np.clip(x + w * 0.5, size // 3, width - size // 3))
    center_y = int(np.clip(y + h * 0.52, size // 3, height - size // 3))
    return center_x, center_y, size


def load_log() -> np.ndarray:
    sprite = cv2.imread(str(LOG_PATH), cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise RuntimeError(f"Log sprite not available: {LOG_PATH}")
    if sprite.shape[0] > LOG_MAX_HEIGHT:
        width = round(sprite.shape[1] * LOG_MAX_HEIGHT / sprite.shape[0])
        sprite = cv2.resize(sprite, (width, LOG_MAX_HEIGHT), interpolation=cv2.INTER_AREA)
    return sprite


def _blend(destination: np.ndarray, color: np.ndarray, alpha: np.ndarray) -> None:
    """In-place alpha blend; callers pass only the region that changes."""
    destination[:] = (
        color.astype(np.float32) * alpha
        + destination.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def overlay_log(frame: np.ndarray, sprite: np.ndarray, anchor: tuple[int, int, int],
                opacity: float = 1.0, scale: float = 1.0) -> None:
    if opacity <= 0:
        return
    cx, cy, target_height = anchor
    height = max(1, int(target_height * scale))
    width = max(1, int(height * sprite.shape[1] / sprite.shape[0]))
    sized = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
    x1, y1 = cx - width // 2, cy - height // 2
    x2, y2 = x1 + width, y1 + height
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if left >= right or top >= bottom:
        return
    part = sized[top - y1:bottom - y1, left - x1:right - x1]
    alpha = (part[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    _blend(frame[top:bottom, left:right], part[:, :, :3], alpha)


def landed_log_anchor(anchor: tuple[int, int, int], frame_height: int) -> tuple[int, int, int]:
    cx, cy, size = anchor
    return cx, max(cy, frame_height - round(size * 0.48)), size


def falling_log_anchor(anchor: tuple[int, int, int], frame_height: int,
                       age: float) -> tuple[int, int, int]:
    """Accelerate downwards, touch the frame floor, then settle with a small bounce."""
    cx, cy, size = anchor
    floor_y = landed_log_anchor(anchor, frame_height)[1]
    distance = floor_y - cy
    if distance <= 0 or age <= 0.18:
        return anchor
    elapsed = age - 0.18
    gravity = frame_height * 3.2
    fall = gravity * elapsed * elapsed * 0.5
    if fall < distance:
        return cx, cy + round(fall), size
    landed_at = math.sqrt(2.0 * distance / gravity)
    bounce_age = elapsed - landed_at
    bounce = size * 0.075 * math.exp(-5.0 * bounce_age) * abs(math.sin(11.0 * bounce_age))
    return cx, floor_y - round(bounce), size


def _smoke_layer(frame: np.ndarray, anchor: tuple[int, int, int],
                 age: float, foreground: bool) -> None:
    if age >= EFFECT_SECONDS:
        return
    cx, cy, size = anchor
    rng = np.random.default_rng(9127 if foreground else 1934)
    layer = np.zeros((*frame.shape[:2], 4), dtype=np.uint8)
    reveal = min(1.0, age / 0.16)
    fade = max(0.0, 1.0 - max(0.0, age - 0.33) / 1.12)
    radius = size * (0.27 + 0.54 * min(age, 1.0))
    for _ in range(22 if foreground else 18):
        angle = rng.uniform(0, 2 * math.pi)
        distance = radius * rng.uniform(0.25, 1.0)
        x = int(cx + math.cos(angle) * distance)
        y = int(cy + math.sin(angle) * distance * 0.63 - age * size * 0.13)
        puff = max(2, int(size * rng.uniform(0.09, 0.20) * (0.7 + age * 0.7)))
        color = (229, 235, 235, 190) if foreground else (152, 162, 166, 125)
        cv2.ellipse(layer, (x, y), (puff, int(puff * 0.68)),
                    rng.uniform(-25, 25), 0, 360, color, -1, cv2.LINE_AA)
    # Long white winding smoke ribbons echo the supplied anime reference.
    for direction in (-1, 1):
        center = (int(cx + direction * size * 0.13), int(cy - size * 0.13 - age * size * 0.09))
        axes = (max(2, int(size * (0.30 + age * 0.27))),
                max(2, int(size * (0.09 + age * 0.12))))
        cv2.ellipse(layer, center, axes, direction * 15, 20, 230,
                    (244, 247, 245, 220), max(2, int(size * 0.025)), cv2.LINE_AA)
    # Blend only the smoke's bounding box, not the whole frame.
    x, y, w, h = cv2.boundingRect(layer[:, :, 3])
    if w == 0 or h == 0:
        return
    region = layer[y:y + h, x:x + w]
    alpha = (region[:, :, 3:4].astype(np.float32) / 255.0) * (reveal * fade)
    _blend(frame[y:y + h, x:x + w], region[:, :, :3], alpha)


def composite_person(frame: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Person (mask 1) over the background (mask 0), like a video-call virtual background."""
    alpha = np.ascontiguousarray(mask, dtype=np.float32)
    return cv2.blendLinear(frame, background, alpha, 1.0 - alpha)


def render_substitution(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                        sprite: np.ndarray, anchor: tuple[int, int, int], age: float) -> np.ndarray:
    """Quick body replacement, expanding smoke, then a persistent wooden log.

    `mask` is only needed while the person fades out (age < PERSON_FADE_SECONDS).
    """
    output = background.copy()
    if mask is not None and age < PERSON_FADE_SECONDS:
        _blend(output, frame, (mask * (1.0 - age / PERSON_FADE_SECONDS))[:, :, None])
    if age >= EFFECT_SECONDS:
        overlay_log(output, sprite, landed_log_anchor(anchor, output.shape[0]))
        return output
    _smoke_layer(output, anchor, age, foreground=False)
    log_opacity = min(1.0, max(0.0, (age - 0.05) / 0.15))
    moving_anchor = falling_log_anchor(anchor, output.shape[0], age)
    overlay_log(output, sprite, moving_anchor, log_opacity,
                scale=1.15 - min(age, 0.25) * 0.6)
    _smoke_layer(output, anchor, age, foreground=True)
    return output
