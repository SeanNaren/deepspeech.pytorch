import argparse
import time

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from decoder import ArgMaxDecoder
from model import DeepSpeech
from torchaudio.data_loader import AudioDataLoader, AudioDataset

parser = argparse.ArgumentParser(description='DeepSpeech pytorch params')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='train_manifest.csv')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='test_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--frame_length', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--frame_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=512, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', default=True, type=bool, help='Turn off progress tracking per iteration')


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

    criterion = CTCLoss()
    alphabet = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "

    audio_config = dict(sample_rate=16000,
                        window_size=0.02,
                        window_stride=0.01,
                        window_type='hamming',
                        )

    train_dataloader_config = dict(type="audio,transcription",
                                   audio=audio_config,
                                   manifest_filename='train_manifest.csv',
                                   alphabet=alphabet,
                                   normalize=True)
    test_dataloader_config = dict(type="audio,transcription",
                                  audio=audio_config,
                                  manifest_filename='test_manifest.csv',
                                  alphabet=alphabet,
                                  normalize=True)
    train_loader = AudioDataLoader(AudioDataset(train_dataloader_config), args.batch_size,
                                   num_workers=args.num_workers)
    test_loader = AudioDataLoader(AudioDataset(test_dataloader_config), args.batch_size,
                                  num_workers=args.num_workers)

    model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers, num_classes=len(alphabet))
    decoder = ArgMaxDecoder(alphabet=alphabet)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(args.epochs - 1):
        model.train()
        end = time.time()
        avg_loss = 0
        for i, (data) in enumerate(train_loader):
            inputs, targets, input_percentages, target_sizes = data
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(inputs)
            target_sizes = Variable(target_sizes)
            targets = Variable(targets)

            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # seqLength x batchSize x alphabet

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int())

            loss = criterion(out, targets, sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # rescale gradients if necessary
            total_norm = torch.FloatTensor([0])
            for param in model.parameters():
                param = param.norm().pow(2).data.cpu()
                total_norm.add_(param)
            total_norm = total_norm.sqrt()
            if total_norm[0] > args.max_norm:
                for param in model.parameters():
                    param.grad.mul_(args.max_norm / total_norm[0])

            # SGD step
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

        avg_loss /= len(train_loader)
        print('Training Summary Epoch: [{0}]\t'
              'Average Loss {loss:.3f}\t'.format(
            (epoch + 1), loss=avg_loss))

        total_cer, total_wer = 0, 0
        for i, (data) in enumerate(test_loader):  # test
            inputs, targets, input_percentages, target_sizes = data

            inputs = Variable(inputs)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # seqLength x batchSize x alphabet
            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int())

            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x])
                cer += decoder.cer(decoded_output[x], target_strings[x])
            total_cer += cer
            total_wer += wer

        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)

        # We need to format the targets into actual sentences
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.0f}\t'
              'Average CER {cer:.0f}\t'.format(
            (epoch + 1), wer=wer * 100, cer=cer * 100))
        decoded_output = decoder.decode(out.data, sizes)
        print (decoded_output)


if __name__ == '__main__':
    main()
