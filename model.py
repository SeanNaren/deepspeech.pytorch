from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

from CTCLoss import ctc_loss


class SequenceWise(nn.Container):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        sizes = x.size()
        x = x.view(sizes[0] * sizes[1], -1)
        x = self.module(x)
        x = x.view(sizes[0], sizes[1], -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchLSTM(nn.Container):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(BatchLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.batch_norm = SequenceWise(nn.BatchNorm1d(hidden_size))

    def forward(self, x, (h0, c0)):
        h0 = Variable(h0.data.resize_(1 * self.num_directions, x.size(1), self.hidden_size).zero_())
        c0 = Variable(c0.data.resize_(1 * self.num_directions, x.size(1), self.hidden_size).zero_())
        x, _ = self.rnn(x, (h0, c0))
        x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        x = self.batch_norm(x)
        return x, (h0, c0)


class StateSequential(nn.Container):
    def __init__(self, *args):
        super(StateSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def forward(self, input, (h0, c0)):
        for module in self._modules.values():
            input, (h0, c0) = module(input, (h0, c0))
        return input, (h0, c0)


class DeepSpeech(nn.Container):
    def __init__(self, num_classes=29, rnn_hidden_size=200, nb_layers=2, bidirectional=True):
        super(DeepSpeech, self).__init__()
        rnn_input_size = 32 * 41
        self.rnn_hidden_size = rnn_hidden_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        rnns = []
        rnn = BatchLSTM(input_size=rnn_input_size, hidden_size=self.rnn_hidden_size,
                        bidirectional=bidirectional)
        rnns.append(('0', rnn))
        for x in xrange(nb_layers - 1):
            rnn = BatchLSTM(input_size=rnn_hidden_size, hidden_size=self.rnn_hidden_size,
                            bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = StateSequential(OrderedDict(rnns))
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.rnn_hidden_size),
            nn.Linear(self.rnn_hidden_size, num_classes)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

    def forward(self, x, hidden, cell):
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1)  # seqLength x batch x features

        x, _ = self.rnns(x, (hidden, cell))

        x = self.fc(x) # seqLength x batch x features
        return x
