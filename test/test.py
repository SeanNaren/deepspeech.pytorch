import torch
import unittest
from torch.autograd import Variable

from decoder import ArgMaxDecoder


class TestCases(unittest.TestCase):
    def test_decoder(self):
        input = Variable(
            torch.FloatTensor(
                [[[0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]],
                 [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]])
                .transpose(0, 1))  # seqLength x batch x outputDim
        decoder = ArgMaxDecoder(alphabet="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        decoded = decoder.decode(input.data, None)
        expected_decoding = ['BAD', 'D']
        self.assertItemsEqual(expected_decoding, decoded)


if __name__ == '__main__':
    unittest.main()
