import torch
import unittest
from torch.autograd import Variable

from CTCLoss import ctc_loss

precision = 1e-5


class CTCTestCase(unittest.TestCase):
    def testAdd(self):  ## test method names begin 'test*'
        input = Variable(torch.FloatTensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]]).transpose(0, 1))
        target = Variable(torch.FloatTensor([3, 3]))
        sizes = Variable(torch.FloatTensor(input.size(1)).fill_(input.size(0)))
        label_lens = Variable(torch.FloatTensor([2]))
        loss = ctc_loss(input, target, sizes, label_lens).data[0]
        self.assertAlmostEqual(loss, 7.355742931366)

        input = Variable(
            torch.FloatTensor([[[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6], [-15, -14, -13, -12, -11]]]).transpose(0,
                                                                                                                    1))
        target = Variable(torch.FloatTensor([2, 3]))
        sizes = Variable(torch.FloatTensor(input.size(1)).fill_(input.size(0)))
        label_lens = Variable(torch.FloatTensor([2]))
        loss = ctc_loss(input, target, sizes, label_lens).data[0]
        self.assertAlmostEqual(loss, 4.9388499259949)

        input = Variable(torch.FloatTensor([
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            [[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6], [-15, -14, -13, -12, -11]]
        ]).transpose(0, 1).contiguous())
        target = Variable(torch.FloatTensor([1, 3, 3, 2, 3]))
        sizes = Variable(torch.FloatTensor([1, 3, 3]))
        label_lens = Variable(torch.FloatTensor([1, 2, 2]))
        loss = torch.sum(ctc_loss(input, target, sizes, label_lens)).data[0]
        self.assertAlmostEqual(loss, 13.904030799866)

if __name__ == '__main__':
    unittest.main()
