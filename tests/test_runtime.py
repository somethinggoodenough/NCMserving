from pathlib import Path
import tempfile
import unittest

from ncm_serving.runtime import verify_ncm_checkout


class RuntimeTests(unittest.TestCase):
    def test_invalid_checkout_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "not an NCM checkout"):
                verify_ncm_checkout(Path(folder))


if __name__ == "__main__":
    unittest.main()
