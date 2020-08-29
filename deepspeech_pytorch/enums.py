from enum import Enum

from torch import nn


class DecoderType(Enum):
    greedy: str = 'greedy'
    beam: str = 'beam'


class MultiGPUType(Enum):
    disabled = None
    distributed = 'ddp'
    dataparallel = 'dp'


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'
    bartlett = 'bartlett'


class RNNType(Enum):
    lstm = nn.LSTM
    rnn = nn.RNN
    gru = nn.GRU


class Precision(Enum):
    half = 16
    full = 32
