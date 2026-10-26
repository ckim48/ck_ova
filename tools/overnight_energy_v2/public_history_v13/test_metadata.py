import unittest
import numpy as np
from radio import reserve_symbols,control_resources


class MetadataTests(unittest.TestCase):
    def test_paid_time_keeps_original_frame_and_mapping(self):
        original=np.full(41,4096);dims=np.full(41,512)
        n,each,total=reserve_symbols(original,dims,3,9)
        self.assertEqual(each,192);self.assertEqual(total,1728)
        self.assertEqual(int(n.sum())+total,int(original.sum()))
        self.assertTrue(np.all(n>=dims))
        with self.assertRaises(ValueError):reserve_symbols(np.array([8]),np.array([8]),1,1)
    def test_control_rate_power_and_energy(self):
        h=np.array([2.,4.]);p,e=control_resources(h,96,192,1e6)
        np.testing.assert_allclose(h*h*p,np.sqrt(2)-1)
        np.testing.assert_allclose(e,p*.000192)
        self.assertTrue(np.all(p<=.32))
        p,e=control_resources(h,0,0,1e6);np.testing.assert_array_equal(e,0)


if __name__=='__main__':unittest.main()
