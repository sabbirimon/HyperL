# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
import importlib.util
import os
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperl.ir import DType, Graph, Type


@unittest.skipUnless(importlib.util.find_spec("jax"), "optional JAX compiler")
class CompilerTest(unittest.TestCase):
    def test_composed_graph_and_device_signature(self):
        import jax
        import numpy as np
        graph = Graph()
        matrix = graph.add_input("matrix", Type(DType.F32, (3, 2)))
        x = graph.add_input("x", Type(DType.F32, (2,)))
        b = graph.add_input("bias", Type(DType.F32, (3,)))
        output = graph.softmax(graph.rmsnorm(graph.gemv(matrix, x, b)))
        graph.set_outputs([output])
        module = graph.lower()
        self.assertIn("stablehlo.dot_general", module.text)
        run = module.compile(jax.devices("cpu")[0])
        a = np.arange(6, dtype=np.float32).reshape(3, 2)
        v = np.array([-3, 2], np.float32)
        bias = np.array([0, 1, -1], np.float32)
        z = a @ v + bias
        z /= np.sqrt(np.mean(z*z) + 1e-5)
        np.testing.assert_allclose(run(a, v, bias)[0], np.exp(z) / np.exp(z).sum(), rtol=1e-5)
        with self.assertRaises(ValueError):
            run(a, v, v)
        with self.assertRaises(ValueError):
            graph.lower(max_tensor_bytes=10)

    def test_causal_attention(self):
        import jax
        import numpy as np
        g = Graph()
        q = g.add_input("q", Type(DType.F32, (2, 3, 4)))
        v = g.add_input("v", Type(DType.F32, (2, 3, 2)))
        g.set_outputs([g.attention(q, q, v, causal=True)])
        compiled = g.lower().compile(jax.devices("cpu")[0])
        rng = np.random.default_rng(7)
        x, values = (rng.normal(size=s).astype(np.float32) for s in [(2,3,4), (2,3,2)])
        expected = np.empty_like(values)
        for h in range(2):
            for t in range(3):
                scores = x[h,:t+1] @ x[h,t] / 2
                exp = np.exp(scores - scores.max())
                expected[h,t] = exp @ values[h,:t+1] / exp.sum()
        np.testing.assert_allclose(compiled(x, values)[0], expected, atol=2e-6)

    @unittest.skipUnless(os.environ.get("HYPERL_TEST_IREE") == "1", "explicit IREE native runtime test")
    def test_iree_cpu_executes_bf16_input_matmul(self):
        import numpy as np
        from iree import runtime as rt
        g = Graph()
        # F32 inputs, a separate BF16 graph below qualifies widening at compile time.
        a = g.add_input("a", Type(DType.F32, (2,3)))
        b = g.add_input("b", Type(DType.F32, (3,2)))
        g.set_outputs([g.matmul(a,b)])
        cfg = rt.Config("local-task")
        ctx = rt.SystemContext(config=cfg)
        vm = rt.VmModule.copy_buffer(cfg.vm_instance, g.lower().iree("llvm-cpu"))
        ctx.add_vm_module(vm)
        x = np.arange(6,dtype=np.float32).reshape(2,3)
        y = x.T.copy()
        out = ctx.modules[vm.name].main(x,y).to_host()
        np.testing.assert_array_equal(out, x @ y)
        a.type.dtype = DType.BF16
        self.assertTrue(g.lower().iree("llvm-cpu"))
