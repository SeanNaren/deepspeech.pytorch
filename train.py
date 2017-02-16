import argparse
import errno
import json
import os
import time

import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import ArgMaxDecoder
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for prediction')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=400, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', default=True, type=bool, help='Turn off progress tracking per iteration')
parser.add_argument('--epoch_save', default=False, type=bool, help='Save model every epoch')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--final_model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save final model')


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


def checkpoint(model, args, nout, epoch=None):
    package = {
        'epoch': epoch if epoch else 'N/A',
        'hidden_size': args.hidden_size,
        'hidden_layers': args.hidden_layers,
        'nout': nout,
        'state_dict': model.state_dict(),
    }
    return package


def main():
    args = parser.parse_args()
    save_folder = args.save_folder
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    criterion = CTCLoss()

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window)

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True)
    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers, num_classes=len(labels))
    decoder = ArgMaxDecoder(labels)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    print(model)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(args.epochs):
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
            out = out.transpose(0, 1)  # TxNxH

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
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int())

            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
            total_cer += cer
            total_wer += wer

        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)

        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.0f}\t'
              'Average CER {cer:.0f}\t'.format(
            (epoch + 1), wer=wer * 100, cer=cer * 100))
        if args.epoch_save:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch)
            torch.save(checkpoint(model, args, len(labels), epoch), file_path)
    torch.save(checkpoint(model, args, len(labels)), args.final_model_path)


if __name__ == '__main__':
    main()
