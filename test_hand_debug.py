import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import hand_debug
import vanish_cam as app
from naruto_effect import hand_points, joined_hands_check, load_log
from test_naruto_effect import SyncWatcher, hand


def make_result(joined_hands=True, now=1.0):
    frame = np.full((540, 960, 3), 70, np.uint8)
    found = [hand(0.45), hand(0.55)] if joined_hands else [hand(0.45)]
    points = hand_points(found, 960, 540)
    passes = [app.HandPass("intera", (0, 0, 960, 540), points, joined_hands_check(points, 960, 540))]
    if not joined_hands:
        passes.append(app.HandPass("zoom", (115, 97, 844, 540), [], joined_hands_check([], 960, 540)))
    return app.HandResult(now, joined_hands, len(found), frame, passes)


class HandDebugViewTests(unittest.TestCase):
    def setUp(self):
        folder = Path(tempfile.mkdtemp())
        self.shown = []
        for item in (
            patch.object(hand_debug, "DEBUG_DIR", folder),
            patch.object(hand_debug, "DATASET_PATH", folder / "dataset.jsonl"),
            patch.object(hand_debug.cv2, "namedWindow"),
            patch.object(hand_debug.cv2, "setMouseCallback"),
            patch.object(hand_debug.cv2, "imshow",
                         side_effect=lambda name, image: self.shown.append(image)),
            patch("builtins.print"),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.folder = folder
        self.view = hand_debug.HandDebugView()

    def test_live_view_shows_green_or_red_verdict(self):
        self.view.update(make_result(True), 0.5)
        self.view.tick(1.0)
        self.view.update(make_result(False), 0.0)
        self.view.tick(1.1)
        green, red = self.shown[0][10, 10], self.shown[1][10, 10]
        self.assertGreater(int(green[1]), int(green[2]))  # BGR: green banner.
        self.assertGreater(int(red[2]), int(red[1]))      # Red banner.

    def test_click_freezes_and_label_buttons_save_sample(self):
        self.view.update(make_result(False), 0.0)
        self.view.tick(1.0)
        self.view._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
        self.assertTrue(self.view.frozen)
        self.view.update(make_result(True), 1.0)  # Ignored while frozen.
        self.view._on_mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
        self.assertTrue(self.view.frozen)  # Only SI / NO unfreeze it.
        self.view.tick(1.1)
        x1, y1, x2, y2 = self.view.buttons[True]
        self.view._on_mouse(cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2, 0, None)
        self.assertFalse(self.view.frozen)
        record = json.loads((self.folder / "dataset.jsonl").read_text(encoding="utf-8"))
        self.assertEqual((record["label"], record["detected"], record["reason"]),
                         (True, False, "manuale"))
        self.assertEqual([p["name"] for p in record["passes"]], ["intera", "zoom"])
        self.assertEqual(len(record["passes"][0]["hands"][0]), 21)
        self.assertTrue((self.folder / record["image"]).is_file())
        self.assertIn("SBAGLIATO", self.view.message)  # Algorithm missed a real pose.

    def test_right_click_freezes_after_delay(self):
        self.view.update(make_result(True), 1.0)
        with patch.object(hand_debug.time, "perf_counter", return_value=10.0):
            self.view._on_mouse(cv2.EVENT_RBUTTONDOWN, 5, 5, 0, None)
        self.view.tick(11.0)
        self.assertFalse(self.view.frozen)
        self.view.tick(13.0)
        self.assertTrue(self.view.frozen)
        self.assertEqual(self.view.reason, "timer")

    def test_recent_summary_reports_share_and_blocking_rules(self):
        for now, joined in ((1.0, True), (1.5, False), (2.0, False), (2.5, True)):
            self.view.update(make_result(joined, now), 0.0)
        summary = self.view.recent_summary()
        self.assertEqual(summary["frames"], 4)
        self.assertEqual(summary["ok"], 0.5)
        self.assertEqual(summary["blocked_by"], {hand_debug.ONE_HAND: 0.5})
        self.view.update(make_result(True, 4.0), 0.0)  # Drops 1.0 and 1.5 (older than 2 s).
        self.assertEqual(self.view.recent_summary()["frames"], 3)

    def test_s_and_n_keys_answer_only_a_frozen_frame(self):
        self.view.update(make_result(True), 1.0)
        self.view.answer(False)  # Live: nothing to answer.
        self.assertFalse((self.folder / "dataset.jsonl").exists())
        self.view.freeze("manuale")
        self.view.answer(False)
        record = json.loads((self.folder / "dataset.jsonl").read_text(encoding="utf-8"))
        self.assertEqual((record["label"], record["detected"]), (False, True))
        self.assertFalse(self.view.frozen)

    def test_debug_mode_never_plays_effect_and_pauses_until_answer(self):
        vanish = app.NarutoVanish(Mock(), SyncWatcher(True), Mock(), load_log())
        vanish.set_background(np.zeros((480, 640, 3), np.uint8))
        vanish.debug_view = self.view
        frame = np.full((480, 640, 3), 90, np.uint8)
        for now in (1.0, 1.25):
            vanish.process(frame, now)
        self.assertEqual(vanish.phase, "ready")
        vanish.sound.play.assert_not_called()
        self.assertTrue(self.view.frozen)
        self.assertEqual(self.view.reason, "rilevato")
        self.assertEqual(self.view.hold, 1.0)
        self.assertIn("SI / NO", vanish.status())
        submitted = vanish.watcher.submitted
        vanish.process(frame, 1.5)
        self.assertEqual(vanish.watcher.submitted, submitted)  # Paused while waiting.
        self.view.answer(True)
        vanish.process(frame, 1.6)
        self.assertEqual(vanish.watcher.submitted, submitted + 1)
        self.assertFalse(self.view.frozen)  # Hold restarts from zero after answering.


if __name__ == "__main__":
    unittest.main()
