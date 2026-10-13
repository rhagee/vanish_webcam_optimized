import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from naruto_effect import (
    EFFECT_SECONDS, JOIN_RULES, LOG_MAX_HEIGHT, falling_log_anchor, hand_points,
    joined_hands_check, joined_hands_from_landmarks,
    landed_log_anchor, load_log, overlay_log, render_substitution, substitution_anchor,
)


def hand(center_x, upward=True):
    points = [SimpleNamespace(x=center_x, y=0.53) for _ in range(21)]
    points[0] = SimpleNamespace(x=center_x, y=0.70)
    for index, offset in ((5, -0.045), (9, -0.015), (13, 0.015), (17, 0.045)):
        points[index] = SimpleNamespace(x=center_x + offset, y=0.54 if upward else 0.84)
    for index, offset in ((8, -0.045), (12, -0.015), (16, 0.015), (20, 0.045)):
        points[index] = SimpleNamespace(x=center_x + offset, y=0.34 if upward else 1.04)
    return points


class JoinedHandsTests(unittest.TestCase):
    def test_two_nearby_hands_recognized_but_one_or_separated_hands_are_not(self):
        left, right = hand(0.45), hand(0.55)
        self.assertTrue(joined_hands_from_landmarks([left, right], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, hand(0.83)], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, hand(0.55, upward=False)], 640, 480))

    def test_horizontal_hands_are_rejected(self):
        # Hands lying flat (e.g. on the desk) were the most common false positive.
        hands = [hand(0.45), hand(0.55)]
        rotated = [[SimpleNamespace(x=point.y, y=1 - point.x) for point in h] for h in hands]
        self.assertFalse(joined_hands_from_landmarks(rotated, 480, 640))

    def test_fists_are_rejected(self):
        fists = [hand(0.45), hand(0.55)]
        for h in fists:
            for tip in (8, 12, 16, 20):  # Fingertips curled back near the knuckles.
                h[tip] = SimpleNamespace(x=h[tip].x, y=0.60)
        self.assertFalse(joined_hands_from_landmarks(fists, 640, 480))

    def test_hands_cut_by_the_frame_edge_are_rejected(self):
        low = [[SimpleNamespace(x=p.x, y=p.y + 0.32) for p in h] for h in (hand(0.45), hand(0.55))]
        self.assertFalse(joined_hands_from_landmarks(low, 640, 480))  # Wrists below the frame.

    def test_check_reports_every_rule_value(self):
        points = hand_points([hand(0.45), hand(0.55)], 640, 480)
        check = joined_hands_check(points, 640, 480)
        self.assertTrue(check["joined"])
        self.assertEqual(set(check["values"]), {name for name, _, _ in JOIN_RULES})


class SyncWatcher:
    """Stand-in for HandWatcher: answers immediately, one value per submitted frame."""

    def __init__(self, joined):
        self.joined = iter(joined) if not isinstance(joined, bool) else None
        self.always = joined if isinstance(joined, bool) else None
        self.submitted = 0
        self.result = None
        self.close = Mock()

    def submit(self, frame, now):
        self.submitted += 1
        joined = self.always if self.joined is None else next(self.joined)
        self.result = app.HandResult(now, joined, 2 if joined else 1, frame)

    def take_result(self):
        result, self.result = self.result, None
        return result


class StateMachineTests(unittest.TestCase):
    frame = np.full((480, 640, 3), 90, np.uint8)

    def make_app(self, joined):
        vanish = app.NarutoVanish(Mock(), SyncWatcher(joined), Mock(), load_log())
        vanish.set_background(np.zeros((480, 640, 3), np.uint8))
        return vanish

    def run_frames(self, vanish, times):
        with patch.object(app, "person_mask", return_value=np.ones((480, 640), np.float32)):
            for now in times:
                vanish.process(self.frame, now)

    def test_effect_fires_after_holding_the_gesture_briefly(self):
        vanish = self.make_app(True)
        self.run_frames(vanish, (1.0, 1.125, 1.1875))
        self.assertEqual(vanish.phase, "ready")
        self.run_frames(vanish, (1.25,))
        self.assertEqual(vanish.phase, "vanishing")
        self.assertEqual(vanish.phase_started, 1.25)
        vanish.sound.play.assert_called_once()
        self.assertEqual(vanish.watcher.submitted, 4)  # Every ready frame is analysed.

    def test_brief_dropout_is_tolerated_but_releasing_resets_the_hold(self):
        vanish = self.make_app([True, False, True])
        self.run_frames(vanish, (1.0, 1.125, 1.25))
        self.assertEqual(vanish.phase, "vanishing")  # One missed frame kept the hold.

        vanish = self.make_app([True, False, False, False, True, True])
        self.run_frames(vanish, (1.0, 1.1, 1.2, 1.3, 1.4, 1.5))
        self.assertEqual(vanish.phase, "ready")  # Hold restarted at 1.4.

    def test_effect_never_fires_on_a_frame_that_is_not_recognized(self):
        vanish = self.make_app([True, True, False, True])
        self.run_frames(vanish, (1.0, 1.125, 1.25))
        self.assertEqual(vanish.phase, "ready")  # Hold is complete but 1.25 is not joined.
        self.run_frames(vanish, (1.3125,))
        self.assertEqual(vanish.phase, "vanishing")

    def test_permanent_background_cuts_out_the_person_while_ready(self):
        vanish = self.make_app(False)
        mask = np.zeros((480, 640), np.float32)
        mask[:, :320] = 1.0  # Person on the left half.
        with patch.object(app, "person_mask", return_value=mask) as segment:
            np.testing.assert_array_equal(vanish.process(self.frame, 1.0), self.frame)
            segment.assert_not_called()  # Off by default: no per-frame segmentation.
            vanish.permanent = True
            shown = vanish.process(self.frame, 1.1)
        self.assertEqual(int(shown[0, 0, 0]), 90)   # Person kept from the camera.
        self.assertEqual(int(shown[0, -1, 0]), 0)   # Background elsewhere.

    def test_vanished_frame_is_computed_once(self):
        vanish = self.make_app(False)
        with patch.object(app, "person_mask", return_value=np.zeros((480, 640), np.float32)):
            vanish.vanish(self.frame, 0.0)
        final = vanish.process(self.frame, EFFECT_SECONDS)
        self.assertEqual(vanish.phase, "vanished")
        self.assertIs(vanish.process(self.frame, 5.0), final)


class HandWatcherTests(unittest.TestCase):
    frame = np.zeros((480, 640, 3), np.uint8)

    @staticmethod
    def models(full, zoomed):
        hands, zoomed_hands = Mock(), Mock()
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=full)
        zoomed_hands.detect.return_value = SimpleNamespace(hand_landmarks=zoomed)
        return hands, zoomed_hands

    def make_watcher(self, full, zoomed):
        watcher = app.HandWatcher(*self.models(full, zoomed))
        self.addCleanup(watcher.close)
        return watcher

    def test_zoomed_crop_recovers_hands_missed_in_full_frame(self):
        watcher = self.make_watcher([hand(0.45)], [hand(0.45), hand(0.55)])
        result = watcher.detect(self.frame, 1.0)
        self.assertEqual((result.joined, result.count), (True, 2))
        self.assertEqual([p.name for p in result.passes], ["intera", "zoom"])
        self.assertFalse(result.passes[0].check["joined"])
        # Zoomed points are mapped back into full-frame pixels.
        x1, y1 = result.passes[1].rect[:2]
        self.assertGreater(result.passes[1].hands[0][:, 0].min(), x1)
        crop = watcher.zoomed_hands.detect.call_args.args[0]
        self.assertLess(crop.width, 640)
        self.assertLess(crop.height, 480)

    def test_full_frame_hit_skips_zoomed_retry_and_timestamps_increase(self):
        watcher = self.make_watcher([hand(0.45), hand(0.55)], [])
        result = watcher.detect(self.frame, 1.0)
        self.assertEqual((result.joined, len(result.passes)), (True, 1))
        watcher.detect(self.frame, 1.0)
        watcher.zoomed_hands.detect.assert_not_called()
        stamps = [c.args[1] for c in watcher.hands.detect_for_video.call_args_list]
        self.assertEqual(stamps, [1000, 1001])

    def test_worker_thread_publishes_latest_result_and_stops(self):
        watcher = self.make_watcher([hand(0.45), hand(0.55)], [])
        watcher.submit(self.frame, 2.5)
        deadline = time.monotonic() + 5
        result = None
        while result is None and time.monotonic() < deadline:
            result = watcher.take_result()
            time.sleep(0.01)
        self.assertEqual((result.seen_at, result.joined, result.count), (2.5, True, 2))
        watcher.close()
        self.assertFalse(watcher.thread.is_alive())


class EffectTests(unittest.TestCase):
    def test_generated_asset_has_real_transparency_and_is_downscaled(self):
        sprite = load_log()
        self.assertEqual(sprite.shape[2], 4)
        self.assertLessEqual(sprite.shape[0], LOG_MAX_HEIGHT)
        self.assertEqual(int(sprite[:, :, 3].min()), 0)
        self.assertGreaterEqual(int(sprite[:, :, 3].max()), 250)

    def test_smoke_animation_leaves_log_behind(self):
        frame = np.full((160, 240, 3), 130, np.uint8)
        background = np.full_like(frame, 35)
        mask = np.ones(frame.shape[:2], np.float32)
        sprite = load_log()
        anchor = (120, 85, 100)
        early = render_substitution(frame, background, mask, sprite, anchor, 0.04)
        middle = render_substitution(frame, background, None, sprite, anchor, 0.55)
        final = render_substitution(frame, background, None, sprite, anchor, EFFECT_SECONDS)
        self.assertTrue(np.any(early != frame))
        self.assertTrue(np.any(early == frame))  # Person still partly visible.
        self.assertTrue(np.any(middle != frame))
        self.assertTrue(np.any(middle != background))
        self.assertTrue(np.any(final != background))
        np.testing.assert_array_equal(final[0, 0], background[0, 0])

    def test_log_anchor_tracks_person_region(self):
        mask = np.zeros((160, 240), np.float32)
        mask[20:130, 140:210] = 1.0
        x, y, size = substitution_anchor(mask)
        self.assertGreater(x, 140)
        self.assertGreater(size, 35)
        self.assertTrue(20 < y < 130)

    def test_log_falls_then_stays_on_floor(self):
        anchor = (120, 80, 90)
        ys = [falling_log_anchor(anchor, 240, age)[1] for age in (0.1, 0.3, 0.5, 0.7)]
        self.assertEqual(ys[0], anchor[1])
        self.assertGreater(ys[2], ys[1])
        self.assertGreater(ys[3], ys[2])
        floor = landed_log_anchor(anchor, 240)
        self.assertGreater(floor[1], ys[3])
        background = np.full((240, 240, 3), 35, np.uint8)
        output = render_substitution(background, background, None, load_log(), anchor, EFFECT_SECONDS)
        expected = background.copy()
        overlay_log(expected, load_log(), floor)
        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    unittest.main()
