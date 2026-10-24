import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from qualify_android_vulkan import select_gpu, validate_result, verify_bundle

class AndroidVulkanQualificationTest(unittest.TestCase):
    def setUp(self):
        self.gpu={"index":0,"name":"Owner test GPU","hardwareGpu":True,"computeQueue":True}
        self.probe={"format":"hyperl-vulkan-probe/1","inferenceQualified":False,"devices":[self.gpu]}
        self.case={"elements":3}
        self.result={"format":"hyperl-vulkan-result/1","backend":"VULKAN_GPU","device":self.gpu["name"],
                     "gpuCompletionConfirmed":True,"elements":3,"storageMode":"host-visible-coherent",
                     "allocatedBytes":4096,"compileMs":1,"submitAndWaitMs":1}

    def test_software_and_unknown_devices_never_selected(self):
        for value in (False,None,"true"):
            probe=copy.deepcopy(self.probe); probe["devices"][0]["hardwareGpu"]=value
            with self.assertRaises(ValueError): select_gpu(probe)

    def test_multiple_devices_require_exact_name(self):
        self.probe["devices"].append(dict(self.gpu,index=1,name="Other GPU"))
        with self.assertRaises(ValueError): select_gpu(self.probe)
        self.assertEqual(select_gpu(self.probe,"Other GPU")["index"],1)
        with self.assertRaises(ValueError): select_gpu(self.probe,"Unavailable")

    def test_missing_completion_identity_and_shape_fail(self):
        for key in ("format","backend","device","gpuCompletionConfirmed","elements","storageMode"):
            metadata=dict(self.result); metadata.pop(key)
            with self.assertRaises(ValueError): validate_result(metadata,self.gpu,self.case,[0,6,12],[0,6,12])
        metadata=dict(self.result,backend="CPU_REFERENCE")
        with self.assertRaises(ValueError): validate_result(metadata,self.gpu,self.case,[0,6,12],[0,6,12])

    def test_nonfinite_mismatched_and_truncated_results_fail(self):
        for actual in ([0,float("nan"),12],[0,float("inf"),12],[0,5,12],[0,6]):
            with self.assertRaises(ValueError): validate_result(self.result,self.gpu,self.case,actual,[0,6,12])
        validate_result(self.result,self.gpu,self.case,[0,6,12],[0,6,12])

    def test_timing_and_budget_evidence_fail_closed(self):
        for key,value in (("allocatedBytes",True),("allocatedBytes",33*1024*1024),("compileMs",float("nan")),("submitAndWaitMs",-1)):
            with self.assertRaises(ValueError): validate_result(dict(self.result,**{key:value}),self.gpu,self.case,[0,6,12],[0,6,12])

    def bundle(self,folder):
        (folder/"test").mkdir()
        files={"hyperl-vulkan":b"fixture-not-executed","test/kernel.spv":b"fixture-not-executed","test/input_0.bin":b"123456789012","test/reference.json":b"[0,6,12]"}
        for path,data in files.items(): (folder/path).write_bytes(data)
        manifest={"format":"hyperl-android-vulkan-bundle/1","abi":"arm64-v8a","files":{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
                  "cases":[{"name":"test","elements":3,"inputs":1,"expectedNonfiniteRejection":False}]}
        (folder/"manifest.json").write_text(json.dumps(manifest)); return manifest

    def test_bundle_rejects_tampering_before_execution(self):
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path); self.bundle(folder); verify_bundle(folder)
            (folder/"test/kernel.spv").write_bytes(b"changed")
            with self.assertRaises(ValueError): verify_bundle(folder)

    def test_bundle_requires_checked_inputs_and_reference(self):
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path); manifest=self.bundle(folder)
            manifest["files"].pop("test/reference.json"); (folder/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): verify_bundle(folder)

    def test_bundle_rejects_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path); manifest=self.bundle(folder)
            manifest["cases"][0]["name"]="../test"; (folder/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): verify_bundle(folder)
            self.bundle_manifest_restore(folder)
            target=folder/"test/input_0.bin"; target.unlink(); target.symlink_to(folder/"hyperl-vulkan")
            with self.assertRaises(ValueError): verify_bundle(folder)

    def bundle_manifest_restore(self,folder):
        manifest=json.loads((folder/"manifest.json").read_text()); manifest["cases"][0]["name"]="test"
        (folder/"manifest.json").write_text(json.dumps(manifest))

if __name__=="__main__": unittest.main()
