import argparse
from pathlib import Path
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from test_naruto_effect import SyncWatcher


def arguments(**overrides):
    values = dict(
        camera=0, width=960, height=540, fps=30.0, fullscreen=False,
        no_mirror=True, virtual_camera=None, virtual_backend=None, virtual_device=None,
        debug=False, self_test=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera), PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIs(app.open_virtual_camera(arguments(), 640, 480), camera)
        module.Camera.assert_called_once_with(
            width=640, height=480, fps=30.0, fmt="BGR", backend=None, device=None,
        )

    def test_unavailable_driver_auto_falls_back_but_explicit_mode_fails(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), 640, 480))
            for options in (dict(virtual_camera=True), dict(virtual_backend="obs")):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), 640, 480)

    def test_disabled_camera_does_not_open_device(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIsNone(app.open_virtual_camera(arguments(virtual_camera=False), 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_handles_noncontiguous_frames(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: non scambiare rosso e blu.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)

    def test_send_resizes_if_input_resolution_changes(self):
        camera = Mock(width=40, height=24)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.shape, (24, 40, 3))
        self.assertTrue((sent == 42).all())


def run_mock_video(test, frames, times, keys, joined=False, send_error=None,
                   remembered=None, hotkey_actions=(), settings=None, person=1.0):
    """Drive run() with fake devices; returns (virtual camera, shown frames, mocks)."""
    saved = {}  # Fake background library: path -> image.
    if remembered is not None:
        saved[Path("ricordato.png")] = remembered

    def save_capture(image):
        saved[Path("cattura.png")] = image
        return Path("cattura.png")

    hotkeys = Mock()
    hotkeys.poll.side_effect = [list(hotkey_actions)] + [[]] * len(frames)
    panel = Mock()
    panel.poll.return_value = None
    capture = Mock()
    capture.read.side_effect = [(True, frame) for frame in frames]
    height, width = frames[0].shape[:2]
    segmenter, hands, zoomed_hands, sound = Mock(), Mock(), Mock(), Mock()
    watcher = SyncWatcher(joined)
    camera = Mock(width=width, height=height)
    camera.send.side_effect = send_error
    shown = []
    with ExitStack() as mocks:
        mocks.enter_context(patch.object(app, "parse_args", return_value=arguments()))
        mocks.enter_context(patch.object(app, "download_models"))
        mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
        mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, hands, zoomed_hands)))
        mocks.enter_context(patch.object(app, "HandWatcher", return_value=watcher))
        mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
        opening = mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
        mask = mocks.enter_context(patch.object(
            app, "person_mask", return_value=np.full((height, width), person, np.float32)))
        mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=times))
        mocks.enter_context(patch.object(app, "GlobalHotkeys", return_value=hotkeys))
        mocks.enter_context(patch.object(app, "BackgroundPanel", return_value=panel))
        mocks.enter_context(patch.object(app, "list_backgrounds", side_effect=lambda: list(saved)))
        mocks.enter_context(patch.object(
            app, "remembered_background", return_value=next(iter(saved), None)))
        save = mocks.enter_context(patch.object(app, "save_capture", side_effect=save_capture))
        mocks.enter_context(patch.object(app, "read_image", side_effect=saved.get))
        remember = mocks.enter_context(patch.object(app, "remember_background"))
        mocks.enter_context(patch.object(app, "load_settings", return_value=dict(settings or {})))
        save_settings = mocks.enter_context(patch.object(app, "save_settings"))
        mocks.enter_context(patch.object(app.cv2, "namedWindow"))
        mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
        mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
        destroy = mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
        mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: shown.append(frame.copy())))
        mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=keys))
        mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Overlay vietato")))
        mocks.enter_context(patch("builtins.print"))
        if send_error is None:
            test.assertEqual(app.run(), 0)
        else:
            with test.assertRaisesRegex(RuntimeError, "send failed"):
                app.run()
        opening.assert_called_once()
        test.assertEqual(opening.call_args.args[1:], (width, height))
        capture.release.assert_called_once()
        segmenter.close.assert_called_once()
        hands.close.assert_called_once()
        zoomed_hands.close.assert_called_once()
        watcher.close.assert_called_once()
        sound.close.assert_called_once()
        camera.close.assert_called_once()
        destroy.assert_called_once()
        hotkeys.close.assert_called_once()
    return camera, shown, SimpleNamespace(watcher=watcher, sound=sound, mask=mask,
                                          save=save, remember=remember,
                                          save_settings=save_settings)


class VideoLoopTests(unittest.TestCase):
    frames = [np.full((24, 40, 3), v, np.uint8) for v in (10, 40, 80, 120, 160)]

    def test_preview_and_virtual_output_are_clean_while_ready(self):
        # start, frame0 countdown, frame1 ends the 3+3 s capture, frames 2-4 ready.
        times = (0, 0.1, 6.1, 6.3, 6.4, 6.5)
        camera, shown, mocks = run_mock_video(self, self.frames, times, (0, 0, 0, 0, ord("q")))
        self.assertEqual(len(shown), 5)
        for sent, preview in zip(camera.send.call_args_list, shown):
            np.testing.assert_array_equal(sent.args[0], preview)
        for frame, preview in zip(self.frames, shown):
            np.testing.assert_array_equal(preview, frame)
        self.assertEqual(mocks.watcher.submitted, 3)  # Every ready frame is analysed.
        mocks.mask.assert_not_called()  # No segmentation until the effect starts.
        mocks.sound.play.assert_not_called()
        mocks.save.assert_called_once()  # Captured background saved and remembered.
        mocks.remember.assert_called_once_with(Path("cattura.png"))

    def test_remembered_background_skips_capture_and_hotkey_vanishes(self):
        remembered = np.full((24, 40, 3), 200, np.uint8)
        _, shown, mocks = run_mock_video(self, self.frames[:3], (0.1, 0.2, 2.0), (0, 0, ord("q")),
                                         remembered=remembered, hotkey_actions=["vanish"])
        mocks.save.assert_not_called()
        mocks.sound.play.assert_called_once()
        np.testing.assert_array_equal(shown[0], self.frames[0])
        self.assertTrue(np.any(shown[2] != self.frames[2]))
        self.assertTrue(np.any(shown[2] != remembered))  # Background plus log.

    def test_held_joined_hands_play_effect_then_leave_static_log(self):
        frames = [np.full((24, 40, 3), v, np.uint8) for v in (10, 40, 80, 100, 120, 140, 160)]
        # frame2 starts the hold, frame3 completes it, frame4 mid-effect, frame5-6 done.
        times = (0, 0.1, 6.1, 6.3, 6.85, 7.4, 8.5, 8.6)
        camera, shown, mocks = run_mock_video(
            self, frames, times, (0, 0, 0, 0, 0, 0, ord("q")), joined=True)
        self.assertEqual(mocks.watcher.submitted, 2)
        mocks.sound.play.assert_called_once()
        background = frames[1]
        np.testing.assert_array_equal(shown[2], frames[2])
        np.testing.assert_array_equal(shown[3], frames[3])
        self.assertTrue(np.any(shown[4] != frames[4]))
        self.assertTrue(np.any(shown[4] != background))
        self.assertTrue(np.any(shown[5] != background))  # Tronco persistente.
        np.testing.assert_array_equal(shown[6], shown[5])
        self.assertEqual(mocks.mask.call_count, 1)  # Only for the anchor; fade already over.
        for sent, preview in zip(camera.send.call_args_list, shown):
            np.testing.assert_array_equal(sent.args[0], preview)

    def test_v_key_starts_effect_and_r_brings_person_back(self):
        times = (0, 0.1, 6.1, 6.3, 8.0, 8.1)
        keys = (0, 0, ord("v"), ord("r"), ord("q"))
        _, shown, mocks = run_mock_video(self, self.frames, times, keys)
        mocks.sound.play.assert_called_once()
        self.assertTrue(np.any(shown[3] != self.frames[3]))
        np.testing.assert_array_equal(shown[4], self.frames[4])

    def test_permanent_background_is_remembered_and_p_key_toggles_it(self):
        remembered = np.full((24, 40, 3), 200, np.uint8)
        _, shown, mocks = run_mock_video(
            self, self.frames[:3], (0.1, 0.2, 0.3, 0.4), (ord("p"), 0, ord("q")),
            remembered=remembered, settings={"permanent": True}, person=0.0)
        np.testing.assert_array_equal(shown[0], remembered)  # Person mask empty: only background.
        mocks.save_settings.assert_called_once_with(permanent=False)
        np.testing.assert_array_equal(shown[1], self.frames[1])  # Toggled off: raw camera.

    def test_all_devices_close_on_send_error(self):
        run_mock_video(self, self.frames, (0, 0.1), (0,), send_error=RuntimeError("send failed"))


if __name__ == "__main__":
    unittest.main()
