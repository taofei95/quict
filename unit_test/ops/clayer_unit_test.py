import os
import unittest
import numpy as np


if os.environ.get("test_with_gpu", True):
    import cupy as cp

    import QuICT.ops.linalg.gpu_calculator as GPUCalculator
    from QuICT.ops.utils.calculation_layer import CalculationLayer


@unittest.skipUnless(os.environ.get("test_with_gpu", True), "require GPU")
class TestCalculationLayer(unittest.TestCase):
    def test_calculation_layer(self):
        A = np.random.random((1 << 5, 1 << 5)).astype(np.complex64)
        B = np.random.random((1 << 5, 1 << 5)).astype(np.complex64)

        based_result = GPUCalculator.dot(A, B, gpu_out=True)

        mempool = cp.get_default_memory_pool()
        before_used_bytes = mempool.used_bytes()

        with CalculationLayer() as CL:
            gpu_A = CL.htod(A)
            gpu_B = CL.htod(B)

            layer_result = CL.dot(gpu_A, gpu_B, gpu_out=True)

        assert np.allclose(based_result, layer_result)

        after_used_bytes = mempool.used_bytes()

        # Check for memory release, maybe failure caused by the mulit-process.
        self.assertEqual(before_used_bytes, after_used_bytes)


if __name__ == "__main__":
    unittest.main()
