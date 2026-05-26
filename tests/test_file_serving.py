import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import app as webapp


class FileServingTests(unittest.TestCase):
    def test_serves_generated_file(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            file_path = output_dir / "Example Video [abc123].txt"
            file_path.write_text("hello\n", encoding="utf-8")

            with patch.object(webapp.ripper, "OUTPUT_DIR", output_dir):
                response = webapp.app.test_client().get("/files/Example%20Video%20%5Babc123%5D.txt")
                response_text = response.get_data(as_text=True)
                response.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_text, "hello\n")

    def test_rejects_path_traversal(self):
        with TemporaryDirectory() as temp_dir:
            with patch.object(webapp.ripper, "OUTPUT_DIR", Path(temp_dir)):
                response = webapp.app.test_client().get("/files/../app.py")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
