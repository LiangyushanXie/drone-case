"""Exercise the installed runtime's preprocessing without loading a detector."""

import importlib.util
import json
import unittest
from types import SimpleNamespace

from model_arch import ROOT, build_square_predictor, load_config


class RuntimePreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("ultralytics") is None:
            raise unittest.SkipTest("Run on MyGPU with the installed author runtime")
        import cv2
        import numpy as np
        import torch
        from ultralytics.utils import ops

        cls.cv2, cls.np, cls.torch, cls.ops = cv2, np, torch, ops

    def setUp(self):
        self.config = load_config()
        # Only the preprocessing interface is needed; no model is constructed.
        self.predictor = object.__new__(build_square_predictor())
        self.predictor.imgsz = [self.config["image_size"]] * 2
        self.predictor.model = SimpleNamespace(
            pt=True, dynamic=False, stride=32, fp16=False
        )
        self.predictor.device = self.torch.device(
            "cuda:0" if self.torch.cuda.is_available() else "cpu"
        )

    def assert_square_tensor(self, image):
        tensor = self.predictor.preprocess([image])
        self.assertEqual(list(tensor.shape), [1, 3, 640, 640])
        self.assertEqual(tensor.dtype, self.torch.float32)
        self.assertEqual(tensor.device, self.predictor.device)
        return tensor

    def test_landscape_portrait_square_and_odd_dimensions(self):
        for height, width in [(1080, 1920), (1920, 1080), (640, 640), (1025, 1921)]:
            with self.subTest(height=height, width=width):
                image = self.np.zeros((height, width, 3), dtype=self.np.uint8)
                image[..., 2] = 255  # BGR red should become normalized RGB red.
                tensor = self.assert_square_tensor(image)
                self.assertEqual(tensor[0, :, 320, 320].tolist(), [1.0, 0.0, 0.0])

    def test_all_committed_preview_images_have_square_input(self):
        manifest = json.loads((ROOT / self.config["sample_manifest"]).read_text())
        self.assertEqual(len(manifest["samples"]), 30)
        for sample in manifest["samples"]:
            with self.subTest(image=sample["image"]):
                image = self.cv2.imread(str(ROOT / sample["image"]))
                self.assertIsNotNone(image)
                self.assert_square_tensor(image)

    def test_boxes_restore_to_original_image_coordinates(self):
        for height, width in [(1080, 1920), (1920, 1080), (640, 640), (1025, 1921)]:
            with self.subTest(height=height, width=width):
                image = self.np.zeros((height, width, 3), dtype=self.np.uint8)
                expected = [width // 4, height // 4, width * 3 // 4, height * 3 // 4]
                x1, y1, x2, y2 = expected
                image[y1:y2, x1:x2] = 255
                padded = self.predictor.pre_transform([image])[0]
                ys, xs = self.np.where(padded[..., 0] > 200)
                observed_box = self.torch.tensor(
                    [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]],
                    dtype=self.torch.float32,
                )
                restored = self.ops.scale_boxes(
                    padded.shape[:2], observed_box, image.shape
                )[0]
                error = (restored - self.torch.tensor(expected)).abs().max().item()
                # Rasterization can shift an edge by one pixel in the resized image.
                self.assertLessEqual(error, max(height, width) / 640 + 1)


if __name__ == "__main__":
    unittest.main()
