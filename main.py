import time
import torch
from aeon import DataLoader, gen_backend
import numpy as np
from torch.autograd import Variable
import argparse

from CTCLoss import ctc_loss
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
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--max_transcript_length', default=1300, type=int, help='Maximum size of transcript in training')
parser.add_argument('--frame_length', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--frame_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--max_duration', default=6.4, type=float,
                    help='The maximum duration of the training data in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--noise_probability', default=0.4, type=float, help='Window type for spectrogram generation')
parser.add_argument('--noise_min', default=0.5, type=float, help='Minimum noise to add')
parser.add_argument('--noise_max', default=1, type=float, help='Maximum noise to add (1 is an SNR of 0 (pure noise)')
parser.add_argument('--hidden_size', default=512, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')


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
    minibatch_size = args.batch_size
    alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    nout = len(alphabet)
    spect_size = (args.frame_length * args.sample_rate / 2) + 1
    be = gen_backend()
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
    transcription_config = dict(alphabet=alphabet,
                                max_length=args.max_transcript_length,
                                pack_for_ctc=False)
    test_dataloader_config = dict(type="audio,transcription",
                                  audio=audio_config,
                                  transcription=transcription_config,
                                  manifest_filename=args.test_manifest,
                                  macrobatch_size=minibatch_size,
                                  minibatch_size=minibatch_size)
    train_loader = DataLoader(train_dataloader_config, be)
    test_loader = DataLoader(test_dataloader_config, be)

    model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers, num_classes=nout)
    decoder = ArgMaxDecoder(alphabet=alphabet)
    if args.cuda:
        model = model.cuda()
    print(model)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in xrange(args.epochs - 1):
        model.train()
        end = time.time()
        avg_loss = 0
        for i, (data) in enumerate(train_loader):  # train
            # measure data loading time
            data_time.update(time.time() - end)
            input = data[0].reshape(minibatch_size, 1, spect_size,
                                    -1)  # batch x channels x freq x time
            label_lengths = Variable(torch.FloatTensor(data[2].astype(dtype=np.float32)).view(-1))

            input = Variable(torch.FloatTensor(input.astype(dtype=np.float32)))
            target = Variable(torch.FloatTensor(data[1].astype(dtype=np.float32)).view(-1))

            if args.cuda:
                input = input.cuda()
                target = target.cuda()

            out = model(input)

            max_seq_length = out.size(0)
            seq_percentage = torch.FloatTensor(data[3].astype(dtype=np.float32)).view(-1)
            sizes = Variable(seq_percentage.mul_(int(max_seq_length) / 100))

            loss = ctc_loss(out, target, sizes, label_lengths)
            loss = loss / input.size(0)  # average the loss by minibatch
            avg_loss = avg_loss + loss.data[0]

            losses.update(loss.data[0], input.size(0))

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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                (epoch + 1), (i + 1), train_loader.nbatches, batch_time=batch_time,
                data_time=data_time, loss=losses))

        avg_loss = avg_loss / train_loader.nbatches
        print('Training Summary Epoch: [{0}]\t'
              'Average Loss {loss:.3f}\t'.format(
            (epoch + 1), loss=avg_loss))

        total_cer, total_wer = 0, 0

        for i, (data) in enumerate(test_loader):  # test
            input = data[0].reshape(minibatch_size, 1, spect_size,
                                    -1)  # batch x channels x freq x time

            input = Variable(torch.FloatTensor(input.astype(dtype=np.float32)))
            target = Variable(torch.FloatTensor(
                data[1].astype(dtype=np.float32).reshape(args.max_transcript_length, minibatch_size, order='F').T))

            if args.cuda:
                input = input.cuda()
                target = target.cuda()

            out = model(input)

            decoded_output = decoder.decode(out.data)
            target_strings = decoder.process_strings(decoder.convert_to_strings(target.data))
            wer, cer = 0, 0
            for x in xrange(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x])
                cer += decoder.cer(decoded_output[x], target_strings[x])
            total_cer += cer
            total_wer += wer
            batch_size = input.size(0)
            wer = wer / batch_size
            cer = cer / batch_size

            print('Validation Epoch: [{0}][{1}/{2}]\t'
                  'Average WER {wer:.0f}\t'
                  'Average CER {cer:.0f}\t'.format(
                (epoch + 1), (i + 1), test_loader.nbatches, wer=wer, cer=cer))
        wer = total_wer / test_loader.ndata
        cer = total_cer / test_loader.ndata

        # We need to format the targets into actual sentences
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.0f}\t'
              'Average CER {cer:.0f}\t'.format(
            (epoch + 1), wer=wer, cer=cer))


if __name__ == '__main__':
    main()
