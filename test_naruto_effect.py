import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from naruto_effect import (
    EFFECT_SECONDS, LOG_MAX_HEIGHT, falling_log_anchor, joined_hands_from_landmarks,
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

    def test_rotated_joined_hands_still_recognized(self):
        hands = [hand(0.45), hand(0.55)]
        rotated = [[SimpleNamespace(x=point.y, y=1 - point.x) for point in h] for h in hands]
        self.assertTrue(joined_hands_from_landmarks(rotated, 480, 640))


class StateMachineTests(unittest.TestCase):
    def make_app(self, found):
        hands = Mock()
        hands.detect_for_video.return_value = SimpleNamespace(hand_landmarks=found)
        vanish = app.NarutoVanish(Mock(), hands, Mock(), load_log())
        vanish.phase = "ready"
        vanish.background = np.zeros((480, 640, 3), np.uint8)
        return vanish

    def test_effect_fires_only_after_holding_the_gesture(self):
        frame = np.full((480, 640, 3), 90, np.uint8)
        vanish = self.make_app([hand(0.45), hand(0.55)])
        with patch.object(app, "person_mask", return_value=np.ones((480, 640), np.float32)):
            for now in (1.0, 1.15, 1.3, 1.45):
                vanish.process(frame, now)
                self.assertEqual(vanish.phase, "ready")
            vanish.process(frame, 1.6)
        self.assertEqual(vanish.phase, "vanishing")
        self.assertEqual(vanish.phase_started, 1.6)
        vanish.sound.play.assert_called_once()

    def test_brief_dropout_is_tolerated_but_releasing_resets_the_hold(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        joined = SimpleNamespace(hand_landmarks=[hand(0.45), hand(0.55)])
        apart = SimpleNamespace(hand_landmarks=[hand(0.45)])
        vanish = self.make_app([])
        vanish.hands.detect_for_video.side_effect = [joined, apart, joined, joined]
        with patch.object(app, "person_mask", return_value=np.ones((480, 640), np.float32)):
            for now in (1.0, 1.15, 1.3, 1.5):
                vanish.process(frame, now)
        self.assertEqual(vanish.phase, "vanishing")  # One missed check kept the hold.

        vanish = self.make_app([])
        vanish.hands.detect_for_video.side_effect = [joined, apart, apart, apart, joined, joined]
        for now in (1.0, 1.15, 1.3, 1.45, 1.6, 1.75):
            vanish.process(frame, now)
        self.assertEqual(vanish.phase, "ready")  # Hold restarted at 1.6.

    def test_hand_model_is_not_run_on_every_frame(self):
        vanish = self.make_app([])
        frame = np.zeros((480, 640, 3), np.uint8)
        for now in (1.0, 1.033, 1.066, 1.1, 1.133, 1.166):
            vanish.process(frame, now)
        stamps = [c.args[1] for c in vanish.hands.detect_for_video.call_args_list]
        self.assertEqual(stamps, [1000, 1133])

    def test_vanished_frame_is_computed_once(self):
        vanish = self.make_app([])
        frame = np.zeros((480, 640, 3), np.uint8)
        with patch.object(app, "person_mask", return_value=np.zeros((480, 640), np.float32)):
            vanish.vanish(frame, 0.0)
        final = vanish.process(frame, EFFECT_SECONDS)
        self.assertEqual(vanish.phase, "vanished")
        self.assertIs(vanish.process(frame, 5.0), final)


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
