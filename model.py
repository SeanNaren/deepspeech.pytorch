from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable


class SequenceWise(nn.Module):
    def __init__(self, module, batch_first=False):
        super(SequenceWise, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        sizes = x.size()
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()
        x = x.view(sizes[0] * sizes[1], -1)
        x = self.module(x)
        x = x.view(sizes[0], sizes[1], -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, batch_norm=True):
        super(BatchLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm_activate = batch_norm
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size))
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_directions, x.size(1), self.hidden_size).type_as(x.data))
        c0 = Variable(torch.zeros(self.num_directions, x.size(1), self.hidden_size).type_as(x.data))
        if self.batch_norm_activate:
            x = self.batch_norm(x)
        x, _ = self.rnn(x, (h0, c0))
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxD*2) -> (TxNxD) by sum
        return x


class DeepSpeech(nn.Module):
    def __init__(self, num_classes=29, rnn_hidden_size=400, nb_layers=4, bidirectional=True):
        super(DeepSpeech, self).__init__()
        rnn_input_size = 32 * 41  # TODO this is only for 16khz, work this out for any window_size/stride/sample_rate
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        rnns = []
        rnn = BatchLSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                        bidirectional=bidirectional)
        rnns.append(('0', rnn))
        for x in range(nb_layers - 1):
            rnn = BatchLSTM(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size,
                            bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected, batch_first=True),
        )

    def forward(self, x):
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # seqLength x batch x features

        x = self.rnns(x)

        x = self.fc(x)
        return x
