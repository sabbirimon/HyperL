import sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "studio")]
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None or not hasattr(np.ndarray, "__dlpack_device__"), "requires NumPy DLPack arrays")
class PortableKernel(unittest.TestCase):
    def test_device_reference_and_no_broadcast(self):
        from hyperl.portable_kernel import execute
        from cuda_kernel import lower
        _, info = lower((ROOT / "examples/cuda/add.cu").read_text())
        report = {"format":"hyperl-migration/1","interface":info}
        a = np.arange(67,dtype=np.float32)
        b = a*2
        def run(x,y,device=(1,0)):
            return execute(report,{0:x,1:y},device=device,to_host=np.asarray)
        np.testing.assert_array_equal(run(a,b),a*3)
        for x,y,device in [(a,b,(2,0)),(a,b[:1],(1,0)),(a,b.astype(np.float64),(1,0))]:
            with self.assertRaises(ValueError):
                run(x,y,device)
