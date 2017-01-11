import torch
from aeon import DataLoader
from neon.backends import gen_backend
import numpy as np
from torch.autograd import Variable
import argparse

from CTCLoss import ctc_loss
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech pytorch params')
parser.add_argument('--noise_manifest', metavar='DIR',
                    help='path to noise manifest csv', default='noise_manifest.csv')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='train_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--max_transcript_length', default=1300, type=int, help='Maximum size of transcript in training')
parser.add_argument('--frame_length', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--frame_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--max_duration', default=15, type=float, help='The maximum duration of the training data in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--noise_probability', default=0.4, type=float, help='Window type for spectrogram generation')
parser.add_argument('--noise_min', default=0.5, type=float, help='Minimum noise to add')
parser.add_argument('--noise_max', default=1, type=float, help='Maximum noise to add (1 is an SNR of 0 (pure noise)')
parser.add_argument('--hidden_size', default=200, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=2, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

args = parser.parse_args()
sample_rate = args.sample_rate
minibatch_size = args.batch_size
alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
nout = len(alphabet)
spect_size = (args.frame_length * sample_rate / 2) + 1
be = gen_backend(batch_size=minibatch_size)

audio_config = dict(sample_freq_hz=sample_rate,
                    max_duration="%f seconds" % args.max_duration,
                    frame_length="%f seconds" % args.frame_length,
                    frame_stride="%f seconds" % args.frame_stride,
                    window_type=args.window,
                    noise_index_file=args.noise_manifest,
                    add_noise_probability=args.noise_probability,
                    noise_level=(args.noise_min, args.noise_max)
                    )

transcription_config = dict(alphabet=alphabet,
                            max_length=args.max_transcript_length,
                            pack_for_ctc=True)

dataloader_config = dict(type="audio,transcription",
                         audio=audio_config,
                         transcription=transcription_config,
                         manifest_filename=args.train_manifest,
                         macrobatch_size=be.bsz,
                         minibatch_size=be.bsz)

train_loader = DataLoader(dataloader_config, be)

model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers)
hidden = Variable(torch.randn(2, minibatch_size, args.hidden_size))
cell = Variable(torch.randn(2, minibatch_size, args.hidden_size))
inputBuffer = torch.FloatTensor()
targetBuffer = torch.FloatTensor()

if args.cuda:
    model = model.cuda()
    inputBuffer = inputBuffer.cuda()
    targetBuffer = targetBuffer.cuda()
    hidden = hidden.cuda()
    cell = cell.cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum)

for x in xrange(args.epochs - 1):
    model.train()

    for i, (data) in enumerate(train_loader):
        input = data[0].reshape(minibatch_size, 1, spect_size,
                                -1)  # Puts the data into the form of batch x channels x freq x time
        # TODO we could probably use the valid percentage to find out the real size
        label_lengths = Variable(torch.FloatTensor(data[2].get().astype(dtype=np.float32)).view(-1))

        # refresh the tape for input and cell states for the new epoch
        input = torch.FloatTensor(input.get().astype(dtype=np.float32))
        inputBuffer.resize_(input.size()).copy_(input)
        input = Variable(inputBuffer)
        target = torch.FloatTensor(data[1].get().astype(dtype=np.float32)).view(-1)
        targetBuffer.resize_(target.size()).copy_(target)
        target = Variable(targetBuffer)

        # refresh the tape for the hidden and cell states for the new epoch
        hidden = Variable(hidden.data)
        cell = Variable(cell.data)

        out = model(input, hidden, cell)

        sizes = Variable(torch.FloatTensor(out.size(1)).fill_(out.size(0)), requires_grad=False)
        loss = ctc_loss(out, target, sizes, label_lengths)
        print(x, loss)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()