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
from model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for prediction')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=400, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--final_model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save final model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--bidirectional', dest='bidirectional', action='store_true', help='Use bi-directional rnns')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.set_defaults(cuda=False, silent=False, checkpoint=False, visdom=False, bidirectional=False, augment=False)


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


def checkpoint(model, optimizer, args, nout, epoch=None, iteration=None, loss_results=None, cer_results=None,
               wer_results=None, avg_loss=None):
    package = {
        'hidden_size': args.hidden_size,
        'hidden_layers': args.hidden_layers,
        'nout': nout,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict()
    }
    if avg_loss is not None:
        package['avg_loss'] = avg_loss
    if epoch is not None:
        package['epoch'] = epoch + 1  # increment for readability
    if iteration is not None:
        package['iteration'] = iteration
    if loss_results is not None:
        package['loss_results'] = loss_results
        package['cer_results'] = cer_results
        package['wer_results'] = wer_results
    return package


def main():
    args = parser.parse_args()
    save_folder = args.save_folder

    loss_results, cer_results, wer_results = None, None, None
    if args.visdom:
        from visdom import Visdom
        viz = Visdom()

        opts = [dict(title='Loss', ylabel='Loss', xlabel='Epoch'),
                dict(title='WER', ylabel='WER', xlabel='Epoch'),
                dict(title='CER', ylabel='CER', xlabel='Epoch')]

        viz_windows = [None, None, None]
        loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
            args.epochs)
        epochs = torch.arange(1, args.epochs + 1)

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
                                       normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers, num_classes=len(labels),
                       rnn_type=supported_rnns[rnn_type],
                       sample_rate=args.sample_rate, window_size=args.window_size,
                       bidirectional=args.bidirectional)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True)
    decoder = ArgMaxDecoder(labels)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    if args.continue_from:
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', None) or 1) - 1  # Python index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))
        if args.visdom and \
                        package['loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
            epoch = start_epoch
            loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], package[
                'wer_results']
            x_axis = epochs[0:epoch]
            y_axis = [loss_results[0:epoch], wer_results[0:epoch], cer_results[0:epoch]]
            for x in range(len(viz_windows)):
                viz_windows[x] = viz.line(
                    X=x_axis,
                    Y=y_axis[x],
                    opts=opts[x],
                )
    else:
        avg_loss = 0
        start_epoch = 0
        start_iter = 0

    print(model)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_loader):
                break
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

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

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
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth.tar' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(
                    checkpoint(model, optimizer, args, len(labels), epoch=epoch, iteration=i, loss_results=loss_results,
                               wer_results=wer_results, cer_results=cer_results, avg_loss=avg_loss),
                    file_path)
        avg_loss /= len(train_loader)

        print('Training Summary Epoch: [{0}]\t'
              'Average Loss {loss:.3f}\t'.format(
            epoch + 1, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
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

            if args.cuda:
                torch.cuda.synchronize()

        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100

        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.0f}\t'
              'Average CER {cer:.0f}\t'.format(
            epoch + 1, wer=wer, cer=cer))

        if args.visdom:
            loss_results[epoch] = avg_loss
            wer_results[epoch] = wer
            cer_results[epoch] = cer
            epoch += 1
            x_axis = epochs[0:epoch]
            y_axis = [loss_results[0:epoch], wer_results[0:epoch], cer_results[0:epoch]]
            for x in range(len(viz_windows)):
                if viz_windows[x] is None:
                    viz_windows[x] = viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        opts=opts[x],
                    )
                else:
                    viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        win=viz_windows[x],
                        update='replace',
                    )
        if args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(checkpoint(model, optimizer, args, len(labels), epoch, loss_results=loss_results,
                                  wer_results=wer_results, cer_results=cer_results),
                       file_path)
        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        avg_loss = 0
    torch.save(checkpoint(model, optimizer, args, len(labels)), args.final_model_path)


if __name__ == '__main__':
    main()
