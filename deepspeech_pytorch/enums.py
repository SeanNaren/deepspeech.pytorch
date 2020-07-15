from enum import Enum


class DecoderType(Enum):
    greedy: str = 'greedy'
    beam: str = 'beam'


class DistributedBackend(Enum):
    gloo = 'gloo'
    mpi = 'mpi'
    nccl = 'nccl'


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'
    bartlett = 'bartlett'


class RNNType(Enum):
    lstm = 'lstm'
    rnn = 'rnn'
    gru = 'gru'
