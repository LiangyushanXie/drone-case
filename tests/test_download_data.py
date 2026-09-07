import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts/download_data.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArchiveSafetyTests(unittest.TestCase):
    def test_valid_archive_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "input.zip", "w") as archive:
                archive.writestr("original/labels.txt", "0 1 2 3\n")
            MODULE.extract_archive(root / "input.zip", root / "raw")
            self.assertEqual(
                (root / "raw/original/labels.txt").read_bytes(), b"0 1 2 3\n"
            )

    def test_rejects_path_traversal_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "bad.zip", "w") as archive:
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaises(ValueError):
                MODULE.extract_archive(root / "bad.zip", root / "raw")
            self.assertFalse((root / "outside.txt").exists())

    def test_rejects_symbolic_link_archive_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            with zipfile.ZipFile(root / "bad.zip", "w") as archive:
                archive.writestr(info, "../outside")
            with self.assertRaises(ValueError):
                MODULE.extract_archive(root / "bad.zip", root / "raw")


if __name__ == "__main__":
    unittest.main()
