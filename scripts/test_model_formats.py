# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import dataclasses
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.model_formats import inspect_model, registry, _json
from hyperl.jax_llm import DecoderConfig


class ModelFormatsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
    def safe(self, name="model.safetensors", tensors=None, payload=b"\0" * 8, raw=None):
        header = raw if raw is not None else json.dumps(tensors or {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
        path = self.root / name; path.write_bytes(struct.pack("<Q", len(header)) + header + payload); return path
    def test_header_validation_is_separate_from_engine_and_sharding(self):
        path = self.root / "model.gguf"
        for version, level in [(2,"header-only"),(3,"header-only"),(99,"unsupported-version")]:
            path.write_bytes(b"GGUF" + struct.pack("<IQQ", version, 1, 0))
            result=inspect_model(path);self.assertEqual(level,result["validation"])
            self.assertFalse(result["execution_ready"]);self.assertFalse(result["sharding_qualified"])
            self.assertFalse(result["payload_integrity_verified"]);self.assertTrue(result["opaque_storage_eligible"])
        path.write_bytes(b"GGUF"+b"\0"*5)
        with self.assertRaises(ValueError):inspect_model(path)
    def test_safetensors_shape_and_offset_validation(self):
        result=inspect_model(self.safe());self.assertEqual("structure-validated",result["validation"])
        self.assertEqual(1,result["details"]["tensor_count"]);self.assertEqual(["F32"],result["details"]["dtypes"])
        for descriptor in [dict(dtype="F32",shape=[3],data_offsets=[0,8]),dict(dtype="F32",shape=[True],data_offsets=[0,8]),
                           dict(dtype="FUTURE",shape=[2],data_offsets=[0,8]),dict(dtype="F32",shape=[2],data_offsets=[1,9])]:
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):inspect_model(self.safe(tensors={"w":descriptor}))
        overlap={"a":dict(dtype="F32",shape=[1],data_offsets=[0,4]),"b":dict(dtype="F32",shape=[1],data_offsets=[0,4])}
        with self.assertRaises(ValueError):inspect_model(self.safe(tensors=overlap))
        with self.assertRaises(ValueError):inspect_model(self.safe(payload=b"\0"*9))
    def test_scalar_empty_tensor_and_metadata_are_valid(self):
        header={"scalar":dict(dtype="F32",shape=[],data_offsets=[0,4]),"empty":dict(dtype="F32",shape=[0],data_offsets=[0,0]),"__metadata__":{"producer":"fixture"}}
        result=inspect_model(self.safe(tensors=header,payload=b"\0"*4));self.assertEqual(2,result["details"]["tensor_count"])
    def test_duplicate_nonfinite_giant_integer_depth_and_header_bounds_reject(self):
        for raw in [b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":1e999}',b'{"x":123456789012345678901}',b'{"x":'+b'['*17+b'0'+b']'*17+b'}']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):_json(raw)
        with self.assertRaises(ValueError):inspect_model(self.safe(raw=b'{"__metadata__":{"x":false}}'))
        path=self.root/"oversize.safetensors";path.write_bytes(struct.pack("<Q",2**63)+b"{}")
        with self.assertRaises(ValueError):inspect_model(path)
    def test_shard_index_checks_names_complete_mapping_sizes_and_confined_paths(self):
        self.safe("a.safetensors")
        index=self.root/"model.safetensors.index.json"
        index.write_text(json.dumps(dict(metadata=dict(total_size=8),weight_map={"weight":"a.safetensors"})))
        result=inspect_model(index);self.assertEqual(1,result["details"]["shard_count"])
        for mapping in [{"missing":"a.safetensors"},{"weight":"../a.safetensors"},{"weight":"/tmp/a.safetensors"},{"weight":"https://example.test/a.safetensors"}]:
            index.write_text(json.dumps(dict(weight_map=mapping)))
            with self.subTest(mapping=mapping),self.assertRaises(ValueError):inspect_model(index)
        (self.root/"link.safetensors").symlink_to(self.root/"a.safetensors")
        index.write_text(json.dumps(dict(weight_map={"weight":"link.safetensors"})))
        with self.assertRaises(ValueError):inspect_model(index)
        index.write_text(json.dumps(dict(metadata=dict(total_size=9),weight_map={"weight":"a.safetensors"})))
        with self.assertRaises(ValueError):inspect_model(index)
    def test_existing_jax_manifest_validates_without_loading_or_hashing_payload(self):
        config=DecoderConfig(vocabulary=8,hidden=4,intermediate=8,layers=1,heads=2,kv_heads=1,context=8)
        records={}
        for name in config.shapes():
            file=name+".npy";(self.root/file).write_bytes(b"unloaded opaque fixture")
            records[name]=dict(file=file,sha256="a"*64)
        document=dict(format="hyperl-llama-f32/1",config=dataclasses.asdict(config),tensors=records)
        path=self.root/"manifest.json";path.write_text(json.dumps(document))
        jax_loaded_before="jax" in sys.modules
        result=inspect_model(self.root);self.assertEqual("manifest-validated-only",result["validation"])
        self.assertFalse(result["payload_integrity_verified"]);self.assertEqual(jax_loaded_before,"jax" in sys.modules)
        document["provenance"]=dict(importer="hyperl-safetensors-llama/1",sources={"config.json":"a"*64,"model.safetensors":"b"*64},source_format="safetensors",output_dtype="float32",tokenizer="not imported; caller supplies matching token IDs",original_context=8,source_dtypes=["F32"])
        path.write_text(json.dumps(document));self.assertEqual("manifest-validated-only",inspect_model(path)["validation"])
        document["provenance"]["load_plugin"]="untrusted.py";path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):inspect_model(path)
        del document["provenance"]["load_plugin"]
        records["norm"]["file"]="../escape.npy";path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):inspect_model(path)
    def test_recognition_never_trusts_extension_or_loads_pickle_and_unknown_stays_storable(self):
        for suffix,kind in [(".onnx","onnx"),(".tflite","tflite"),(".mlmodel","coreml"),(".pt","pytorch"),(".pth","pytorch"),(".future","unknown"),(".bin","unknown")]:
            path=self.root/("fixture"+suffix);path.write_bytes(b"not executable model data")
            result=inspect_model(path);self.assertEqual(kind,result["detected_format"]);self.assertFalse(result["execution_ready"])
        package=self.root/"fixture.mlpackage";package.mkdir();self.assertEqual("extension-hint-only",inspect_model(package)["validation"])
        source=self.safe();link=self.root/"link.safetensors";link.symlink_to(source)
        with self.assertRaises(ValueError):inspect_model(link)
        first=registry();first["entries"]["gguf"]["versions"].append(999)
        self.assertNotIn(999,registry()["entries"]["gguf"]["versions"])
    def test_existing_cluster_cli_exposes_read_only_inspection(self):
        environment=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/"python"))
        result=subprocess.run([sys.executable,"-m","hyperl.cluster","model-inspect",str(self.safe())],env=environment,capture_output=True,text=True,timeout=15)
        self.assertEqual(0,result.returncode,result.stderr);self.assertEqual("safetensors",json.loads(result.stdout)["detected_format"])


if __name__ == "__main__":
    unittest.main()
