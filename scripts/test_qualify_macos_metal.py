"""Metadata contract fixtures only: no GPU execution or hardware qualification."""
import copy,unittest
from qualify_macos_metal import result_evidence,validate_result_metadata

class MetalQualificationContractTest(unittest.TestCase):
    name='TEST SERIALIZATION ONLY'
    def metadata(self):
        return {'backend':'METAL_GPU','device':self.name,'cpuVerified':True,'result':[0,6,12],
                'metrics':{'format':'hyperl-metal-result/1','device':self.name,'storageMode':'managed','gpuCompletionConfirmed':True,'compileMs':1.0,'submitAndWaitMs':1.0}}
    def test_explicit_metadata_contract(self):
        validate_result_metadata(self.metadata(),self.name)
    def test_missing_or_false_verification_never_passes(self):
        for key in ('backend','device','cpuVerified','metrics'):
            with self.subTest(missing=key):
                value=self.metadata();del value[key]
                with self.assertRaises(ValueError):validate_result_metadata(value,self.name)
        for flag in (False,None,1,'true'):
            with self.subTest(cpuVerified=flag):
                value=self.metadata();value['cpuVerified']=flag
                with self.assertRaises(ValueError):validate_result_metadata(value,self.name)
    def test_completion_and_exact_device_required(self):
        for key,value in (('format','other'),('device','OTHER TEST DEVICE'),('storageMode','unknown'),('gpuCompletionConfirmed',False),('gpuCompletionConfirmed',1)):
            with self.subTest(key=key,value=value):
                result=self.metadata();result['metrics'][key]=value
                with self.assertRaises(ValueError):validate_result_metadata(result,self.name)
        result=self.metadata();del result['metrics']['gpuCompletionConfirmed']
        with self.assertRaises(ValueError):validate_result_metadata(result,self.name)
    def test_evidence_excludes_arrays_and_bounds_text(self):
        value=self.metadata();value['device']='x'*1024
        original=copy.deepcopy(value);evidence=result_evidence(value,3)
        self.assertNotIn('result',evidence);self.assertEqual(256,len(evidence['device']))
        self.assertEqual(original,value)

if __name__=='__main__':unittest.main()
