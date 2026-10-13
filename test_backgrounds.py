import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import backgrounds


class BackgroundLibraryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        patches = (
            patch.object(backgrounds, "BACKGROUND_DIR", self.folder / "backgrounds"),
            patch.object(backgrounds, "SETTINGS_PATH", self.folder / "settings.json"),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_capture_is_saved_listed_and_remembered(self):
        image = np.full((24, 40, 3), (10, 70, 200), np.uint8)
        path = backgrounds.save_capture(image)
        self.assertEqual(path.suffix, ".png")
        np.testing.assert_array_equal(backgrounds.read_image(path), image)
        self.assertEqual(backgrounds.list_backgrounds(), [path])
        self.assertIsNone(backgrounds.remembered_background())
        backgrounds.remember_background(path)
        self.assertEqual(backgrounds.remembered_background(), path)
        path.unlink()
        self.assertIsNone(backgrounds.remembered_background())  # Deleted file is ignored.

    def test_import_copies_jpg_and_converts_other_formats(self):
        image = np.full((30, 50, 3), 120, np.uint8)
        jpg, bmp = self.folder / "foto vacanze.jpg", self.folder / "custom.bmp"
        cv2.imencode(".jpg", image)[1].tofile(str(jpg))
        cv2.imencode(".bmp", image)[1].tofile(str(bmp))
        self.assertEqual(backgrounds.import_image(jpg).suffix, ".jpg")
        self.assertEqual(backgrounds.import_image(bmp).suffix, ".png")
        self.assertEqual(len(backgrounds.list_backgrounds()), 2)
        (self.folder / "broken.png").write_bytes(b"not an image")
        with self.assertRaises(RuntimeError):
            backgrounds.import_image(self.folder / "broken.png")

    def test_settings_are_merged_and_survive_a_broken_file(self):
        path = backgrounds.save_capture(np.zeros((8, 8, 3), np.uint8))
        backgrounds.remember_background(path)
        backgrounds.save_settings(permanent=True)
        self.assertEqual(backgrounds.load_settings(), {"background": path.name, "permanent": True})
        self.assertEqual(backgrounds.remembered_background(), path)
        backgrounds.SETTINGS_PATH.write_text("{not json", encoding="utf-8")
        self.assertEqual(backgrounds.load_settings(), {})
        self.assertIsNone(backgrounds.remembered_background())

    def test_fit_background_covers_frame_without_distortion(self):
        image = np.zeros((100, 400, 3), np.uint8)
        image[:, 150:250] = 255  # Centre stripe must stay centred.
        fitted = backgrounds.fit_background(image, 160, 90)
        self.assertEqual(fitted.shape, (90, 160, 3))
        self.assertEqual(int(fitted[45, 80, 0]), 255)
        self.assertEqual(int(fitted[45, 2, 0]), 0)


if __name__ == "__main__":
    unittest.main()
