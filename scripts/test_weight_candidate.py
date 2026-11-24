# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from hyperl.weight_candidate import clip_candidate


@unittest.skipUnless(find_spec("numpy"), "optional NumPy runtime")
class CandidateTest(unittest.TestCase):
    def test_only_selected_tensor_changes_and_original_stays_verifiable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = json.dumps({k: dict(dtype="F32", shape=[2], data_offsets=[n, n+8])
                                 for k, n in [("a", 0), ("b", 8)]}).encode()
            original = struct.pack("<Q", len(header)) + header + struct.pack("<4f", -7, 2, 8, 9)
            sha = hashlib.sha256(original).hexdigest()
            source, candidate = root / "input.safetensors", root / "new.safetensors"
            source.write_bytes(original)
            for digest, name, clip in [("0"*64, "a", 3), (sha, "missing", 3), (sha, "a", 10)]:
                with self.assertRaises(ValueError):
                    clip_candidate(source, candidate, name, clip, digest, 4096)
                self.assertFalse(candidate.exists())
            with self.assertRaises(InterruptedError):
                clip_candidate(source, candidate, "a", 3, sha, 4096, cancelled=lambda: True)
            with self.assertRaises(ValueError):
                clip_candidate(source, candidate, "a", 3, sha, len(original) - 1)
            self.assertFalse(candidate.exists())
            result = clip_candidate(source, candidate, "a", 3, sha, 4096)
            self.assertEqual(result["values_changed"], 1)
            self.assertEqual(struct.unpack("<4f", candidate.read_bytes()[-16:]), (-3, 2, 8, 9))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(), result["candidate_sha256"])
            with self.assertRaises(ValueError):
                clip_candidate(source, candidate, "a", 1, sha, 4096)
