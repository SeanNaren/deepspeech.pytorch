# functions/add.py
import torch
from ctc import cpu_ctc_np
from torch.autograd import Function


class CPU_CTC(Function):
    def forward(self, input, target, sizes, label_lens):
        """
        input: Tensor of (seqLength x batch x outputDim) containing output from network
        target: 1 dimensional Tensor containing all the targets of the batch in one large sequence
        sizes: Tensor of size (batch) containing size of each output sequence from the network
        label_lens: Tensor of (batch) containing label length of each example
        """
        is_cuda = True if input.is_cuda else False
        self.target = target
        self.sizes = sizes
        self.label_lens = label_lens
        act_lens = sizes
        acts = input.cpu().numpy()
        act_lens = act_lens.numpy()
        labels = target.numpy()
        label_lens = label_lens.numpy()
        self.cost, self.grads = cpu_ctc_np(acts, act_lens, labels, label_lens)
        self.grads = torch.FloatTensor(self.grads)
        self.cost = torch.FloatTensor(self.cost)
        if is_cuda:
            self.grads = self.grads.cuda()
            self.cost = self.cost.cuda()
        return self.cost

    def backward(self, gradOutput):
        return self.grads, self.target, self.sizes, self.label_lens


def ctc_loss(input, target, sizes, label_lens):
    return CPU_CTC()(input, target, sizes, label_lens)
