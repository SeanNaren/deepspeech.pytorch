from aeon import DataLoader
from aeon import gen_backend
import numpy as np
from CTCLoss import CTC

import torch
from torch.autograd import Variable
import argparse

from decoder import ArgMaxDecoder
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech pytorch params')
parser.add_argument('--noise_manifest', metavar='DIR',
                    help='path to noise manifest csv', default='noise_manifest.csv')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='train_manifest.csv')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='test_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=5, type=int, help='Batch size for training')
parser.add_argument('--max_transcript_length', default=1300, type=int, help='Maximum size of transcript in training')
parser.add_argument('--frame_length', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--frame_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--max_duration', default=6.4, type=float,
                    help='The maximum duration of the training data in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--noise_probability', default=0.4, type=float, help='Probability of adding injecting noise')
parser.add_argument('--noise_min', default=0.5, type=float, help='Minimum noise to add')
parser.add_argument('--noise_max', default=1, type=float, help='Maximum noise to add (1 is an SNR of 0 (pure noise)')
parser.add_argument('--hidden_size', default=512, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')

iterations = 70


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    args = parser.parse_args()
    alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    nout = len(alphabet)
    spect_size = (args.frame_length * args.sample_rate / 2) + 1
    criterion = CTC()

    model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers, num_classes=nout)
    decoder = ArgMaxDecoder(alphabet=alphabet)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum)

    minibatch_size = args.batch_size
    audio_config = dict(sample_freq_hz=args.sample_rate,
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
    train_dataloader_config = dict(type="audio,transcription",
                                   audio=audio_config,
                                   transcription=transcription_config,
                                   manifest_filename=args.train_manifest,
                                   macrobatch_size=minibatch_size,
                                   minibatch_size=minibatch_size)
    be = gen_backend()
    train_loader = DataLoader(train_dataloader_config, be)
    data = train_loader.next()
    for x in range(iterations):
        model.train()
        label_lengths = Variable(torch.FloatTensor(data[2].astype(dtype=np.float32)).view(-1))
        input = data[0].reshape(int(minibatch_size), 1, int(spect_size),
                                -1)  # batch x channels x freq x time
        input = Variable(torch.FloatTensor(input.astype(dtype=np.float32)))
        target = Variable(torch.FloatTensor(data[1].astype(dtype=np.float32)).view(
            -1))

        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        out = model(input)

        seq_length = out.size(0)
        seq_percentage = torch.FloatTensor(data[3].astype(dtype=np.float32)).view(-1)
        sizes = Variable(seq_percentage.mul_(int(seq_length)) / 100)

        loss = criterion(out, target, sizes, label_lengths)
        loss = loss / input.size(0)  # average the loss by minibatch

        loss_sum = loss.data.sum()
        inf = float("inf")
        if loss_sum == inf or loss_sum == -inf:
            print("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0
        else:
            loss_value = loss.data[0]

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # rescale gradients if necessary
        total_norm = torch.FloatTensor([0])
        for param in model.parameters():
            param = Variable(param.data).cpu()
            total_norm.add_(param.norm().pow(2).data)
        total_norm = total_norm.sqrt()
        if total_norm[0] > args.max_norm:
            for param in model.parameters():
                param.grad.mul_(args.max_norm / total_norm[0])

        # SGD step
        optimizer.step()

        print('Epoch: [{0}]\t'
              'Loss {loss:.4f}\t'.format(
            (x + 1), 1, loss=loss_value))
    decoded_output = decoder.decode(out.data, sizes)
    print(decoded_output)

if __name__ == '__main__':
    main()
