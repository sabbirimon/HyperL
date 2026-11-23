# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Exercise native discovery, preferences, admission and publication via Tk."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

import hyperl_studio as studio


class StudioTest(unittest.TestCase):
    def test_desktop_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            file = directory / "local weights.safetensors"
            header = json.dumps({"x": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]},
                                 "w": {"dtype": "F32", "shape": [1, 2], "data_offsets": [8, 16]}}).encode()
            header += b" " * (-len(header) % 8)
            raw = struct.pack("<Q", len(header)) + header + struct.pack("<4f", 2, 3, 5, 7)
            file.write_bytes(raw)
            output = directory / "out.safetensors"
            prefs = directory / "studio.json"
            root = tk.Tk()
            root.withdraw()
            self.addCleanup(root.destroy)
            with patch.object(studio, "preferences_path", return_value=prefs):
                panel = studio.Studio(root)
            for key, val in dict(engine=os.environ.get("HYPERL_ENGINE", studio.engine_path()),
                                 model=str(file), output=str(output), budget="1",
                                 digest=hashlib.sha256(raw).hexdigest()).items():
                panel.fields[key].set(val)

            def run(action):
                panel.start(action)
                deadline = time.monotonic() + 30
                while panel.process and time.monotonic() < deadline:
                    root.update()
                    time.sleep(0.01)
                self.assertIsNone(panel.process)
                return panel.status.get()

            self.assertIn("logical_processors=", run("processors"))
            self.assertIn("workers=1", run("plan"))
            self.assertFalse(output.exists())
            self.assertIn("output_bytes=4", run("project"))
            self.assertEqual(struct.unpack("<f", output.read_bytes()[-4:]), (31.0,))
            self.assertEqual(studio.read_settings(prefs), {"device": "auto", "budget": "1"})
            self.assertEqual(run("project"), "Choose a new output file.")


if __name__ == "__main__":
    unittest.main()
