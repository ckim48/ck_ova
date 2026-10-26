import unittest

import numpy as np

import codec
from wireless import quantize


class CodecTests(unittest.TestCase):
    def test_original_quantizer_is_preserved_exactly(self):
        rng=np.random.default_rng(21)
        for bits in [2,4,6,8,12,16]:
            data=rng.normal(0,.001,(4,17));data[1]=0;data[-1,1:]=0
            expected,_,_=quantize(data[None],bits,32)
            packet=codec.encode(data,bits)
            np.testing.assert_array_equal(codec.decode(packet,4,17),expected[0])
            payload=codec.fragment_bits(packet,4)
            self.assertEqual(int(payload.sum()),8*len(packet))
            self.assertLessEqual(int(payload.max()-payload.min()),8)

    def test_zero_client_has_no_payload(self):
        data=np.zeros((41,512));packet=codec.encode(data,8)
        self.assertEqual(packet,b'')
        np.testing.assert_array_equal(codec.decode(packet,41,512),data)

    def test_truncated_or_wrong_shape_packet_is_rejected(self):
        packet=codec.encode(np.ones((2,512)),8)
        with self.assertRaises(ValueError):codec.decode(packet[:-1],2,512)
        with self.assertRaises(ValueError):codec.decode(packet,3,512)


if __name__=='__main__':
    unittest.main()
