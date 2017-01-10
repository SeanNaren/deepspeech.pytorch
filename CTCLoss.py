# functions/add.py
import torch
from ctc import cpu_ctc_np
from torch.autograd import Function


class CPU_CTC(Function):
    def forward(self, input, target, sizes, label_lens):
        is_cuda = True if input.is_cuda else False
        self.target = target
        self.sizes = sizes
        self.label_lens = label_lens
        act_lens = sizes
        label_lens = torch.FloatTensor(target.size(0))
        target = target.cpu().numpy()
        targets = []
        for i in xrange(len(target)):
            x = target[i]
            label = x[x != 0]  # Due to padding, we remove zeros
            targets.append(torch.FloatTensor(label))
            label_lens[i] = label.size
        labels = torch.zeros(int(torch.sum(label_lens)))
        pos = 0
        for target in targets:
            labels[pos:pos + target.size(0)] = target
            pos = pos + target.size(0)
        acts = input.cpu().numpy()
        act_lens = act_lens.numpy()
        labels = labels.numpy()
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


def ctc_loss(input, target, sizes):
    return CPU_CTC()(input, target, sizes)
