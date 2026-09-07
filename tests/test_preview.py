import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data_utils
from data_utils import (
    choose_dispersed_images,
    parse_annotations,
    sha256,
    verify_manifest,
)
from model_arch import load_config, prediction_arguments
from run_inference import render_pair

try:
    from PIL import Image
except ImportError:
    Image = None


class AnnotationTests(unittest.TestCase):
    def test_original_visdrone_car_mapping_and_coordinates(self):
        text = "10,20,30,40,1,4,0,2\n3,4,8,9,1,2,0,0\n0,0,20,20,0,0,0,0\n"
        cars, ignored = parse_annotations(text)
        self.assertEqual(len(cars), 1)
        self.assertEqual(cars[0]["xyxy"], [10, 20, 40, 60])
        self.assertEqual(cars[0]["occlusion"], 2)
        self.assertEqual(len(ignored), 1)

    def test_unscored_car_is_ignored(self):
        cars, ignored = parse_annotations("1,2,3,4,0,4,0,0\n")
        self.assertEqual(cars, [])
        self.assertEqual(len(ignored), 1)

    def test_invalid_scored_car_is_not_silently_corrected(self):
        with self.assertRaises(ValueError):
            parse_annotations("1,2,0,4,1,4,0,0\n")


class SamplingTests(unittest.TestCase):
    def test_dispersed_selection_is_reproducible_and_order_independent(self):
        images = [
            Path(f"{group:07d}_{frame:05d}.jpg")
            for group in range(40)
            for frame in range(4)
        ]
        selected = choose_dispersed_images(images, 30, 20260907)
        self.assertEqual(
            selected, choose_dispersed_images(list(reversed(images)), 30, 20260907)
        )
        self.assertEqual(len({p.name.split("_")[0] for p in selected}), 30)

    def test_insufficient_groups_fail_instead_of_repeating_frames(self):
        with self.assertRaises(ValueError):
            choose_dispersed_images(
                [Path("0000001_00001.jpg"), Path("0000001_00002.jpg")], 2, 0
            )

    def test_manifest_detects_changed_input_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "val"
            directory.mkdir()
            image, annotation = directory / "a.jpg", directory / "a.txt"
            image.write_bytes(b"original image fixture")
            annotation.write_text("1,2,3,4,1,4,0,0\n")
            manifest = {
                "samples": [
                    {
                        "image": "val/a.jpg",
                        "annotation": "val/a.txt",
                        "image_sha256": sha256(image),
                        "annotation_sha256": sha256(annotation),
                    }
                ]
            }
            with patch.object(data_utils, "ROOT", root):
                verify_manifest(
                    manifest, {"sample_count": 1, "validation_directory": "val"}
                )
                image.write_bytes(b"changed image fixture")
                with self.assertRaises(ValueError):
                    verify_manifest(
                        manifest, {"sample_count": 1, "validation_directory": "val"}
                    )


class ConfigurationTests(unittest.TestCase):
    def test_agreed_prediction_interface(self):
        config = load_config()
        arguments = prediction_arguments(config)
        self.assertEqual((arguments["imgsz"], arguments["batch"]), (640, 1))
        self.assertEqual(arguments["classes"], [2])
        self.assertFalse(arguments["rect"])
        self.assertFalse(arguments["half"])
        self.assertEqual((arguments["conf"], arguments["iou"]), (0.25, 0.70))

    def test_confidence_outside_probability_range_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            config["confidence"] = 1.2
            path = Path(temporary) / "bad.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)


@unittest.skipIf(Image is None, "Optional Pillow is required for rendering checks")
class RenderingTests(unittest.TestCase):
    def test_pair_preserves_two_original_size_panels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "fixture.png"
            Image.new("RGB", (160, 100), "#202020").save(image_path)
            output = root / "pair.png"
            render_pair(
                image_path,
                [{"xyxy": [10, 20, 40, 50]}],
                [],
                [{"xyxy": [12, 22, 42, 52], "confidence": 0.8}],
                output,
            )
            with Image.open(output) as result:
                self.assertEqual(result.size, (328, 148))
                self.assertEqual(result.getpixel((10, 68)), (24, 199, 122))
                self.assertEqual(result.getpixel((180, 70)), (255, 157, 46))


if __name__ == "__main__":
    unittest.main()
