# functions/add.py
import torch
from ctc import cpu_ctc_np
from torch.autograd import Function
from torch.nn import Module

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class CPUCTC(Function):
    def forward(self, input, target, sizes, label_lens):
        """
        input: Tensor of (seqLength x batch x outputDim) containing output from network
        target: 1 dimensional Tensor containing all the targets of the batch in one large sequence
        sizes: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        act_lens = sizes.cpu().numpy()
        acts = input.cpu().numpy()
        labels = target.cpu().numpy()
        label_lens = label_lens.cpu().numpy()
        self.cost, self.grads = cpu_ctc_np(acts, act_lens, labels, label_lens)
        self.grads = torch.FloatTensor(self.grads)
        self.cost = torch.FloatTensor([torch.sum(torch.FloatTensor(self.cost))])
        if input.is_cuda:
            self.grads = self.grads.cuda()
            self.cost = self.cost.cuda()
        return self.cost

    def backward(self, grad_output):
        return self.grads, None, None, None


class CTC(Module):

    def __init__(self):
        super(CTC, self).__init__()

    def forward(self, input, target, sizes, label_lens):
        _assert_no_grad(target)
        _assert_no_grad(sizes)
        _assert_no_grad(label_lens)
        return CPUCTC()(input, target, sizes, label_lens)