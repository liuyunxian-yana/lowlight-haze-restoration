import unittest
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .core import box_sum, cp_ahe, fuse, restore


class CEEFTests(unittest.TestCase):
    def test_box_sum_matches_direct_window(self):
        x = np.random.default_rng(1).random((13, 17))
        expected = sliding_window_view(np.pad(x, 3), (7, 7)).sum((2, 3))
        np.testing.assert_allclose(box_sum(x, 3), expected, atol=1e-13)

    def test_contract_and_weights(self):
        x = np.random.default_rng(2).integers(0, 256, (32, 35, 3), dtype=np.uint8)
        self.assertEqual(cp_ahe(x).shape, x.shape)
        y = restore(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.uint8)
        np.testing.assert_array_equal(x, x.copy())

    def test_five_exposures_and_reconstruction_are_finite(self):
        rng = np.random.default_rng(3)
        x = rng.random((32, 33, 3, 5))
        y = fuse(x)
        self.assertEqual(y.shape, (32, 33, 3))
        self.assertTrue(np.isfinite(y).all())
        with self.assertRaises(ValueError):
            fuse(x[:8])


if __name__ == '__main__':
    unittest.main()
