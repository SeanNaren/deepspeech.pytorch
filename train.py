import argparse
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from warpctc_pytorch import CTCLoss
from torch.nn.parallel import DistributedDataParallel

from data.data_loader import SpectrogramDataset, DSRandomSampler, DSElasticDistributedSampler, AudioDataLoader
from decoder import GreedyDecoder
from logger import VisdomLogger, TensorBoardLogger
from model import DeepSpeech, supported_rnns
from state import TrainingState
from test import evaluate
from utils import check_loss, CheckpointHandler

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=1024, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true', help='Enable CPU only training')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                    help='Enables epoch checkpoint saving of model')
parser.add_argument('--checkpoint-per-iteration', default=0, type=int,
                    help='Save checkpoint per N number of iterations. Default is disabled')
parser.add_argument('--save-n-recent-models', default=0, type=int,
                    help='Maximum number of checkpoints to save. If the max is reached, we delete older checkpoints.'
                         'Default is there is no maximum number, so we save all checkpoints.')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--best-val-model-name', default='deepspeech_final.pth',
                    help='Location to save best validated model within the save folder')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true',
                    help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true',
                    help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--load_auto_checkpoint', dest='load_auto_checkpoint', action='store_true',
                    help='Enable when handling interruptions. Automatically load the latest checkpoint from the '
                         'save folder')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--opt-level', type=str,
                    help='Apex optimization level,'
                         'check https://nvidia.github.io/apex/amp.html for more information')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None,
                    help='Overrides Apex keep_batch_norm_fp32 flag')
parser.add_argument('--loss-scale', default=1,
                    help='Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients')


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


if __name__ == '__main__':
    args = parser.parse_args()

    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main_proc = True
    device = torch.device("cpu" if args.no_cuda else "cuda")
    args.distributed = os.environ.get("LOCAL_RANK")  # If local rank exists, distributed env
    if args.distributed:
        # when using NCCL, on failures, surviving nodes will deadlock on NCCL ops
        # because NCCL uses a spin-lock on the device. Set this env var and
        # to enable a watchdog thread that will destroy stale NCCL communicators
        os.environ["NCCL_BLOCKING_WAIT"] = "1"

        device_id = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device_id)
        print(f"Setting CUDA Device to {device_id}")

        dist.init_process_group(backend=args.dist_backend)
        main_proc = device_id == 0  # Main process handles saving of models and reporting

    checkpoint_handler = CheckpointHandler(save_folder=args.save_folder,
                                           best_val_model_name=args.best_val_model_name,
                                           checkpoint_per_iteration=args.checkpoint_per_iteration,
                                           save_n_recent_models=args.save_n_recent_models)

    if main_proc and args.visdom:
        visdom_logger = VisdomLogger(args.id, args.epochs)
    if main_proc and args.tensorboard:
        tensorboard_logger = TensorBoardLogger(args.id, args.log_dir, args.log_params)

    if args.load_auto_checkpoint:
        latest_checkpoint = checkpoint_handler.find_latest_checkpoint()
        if latest_checkpoint:
            args.continue_from = latest_checkpoint

    if args.continue_from:  # Starting from previous model
        state = TrainingState.load_state(state_path=args.continue_from)
        model = state.model
        if args.finetune:
            state.init_finetune_states(args.epochs)

        if main_proc and args.visdom:  # Add previous scores to visdom graph
            visdom_logger.load_previous_values(state.epoch, state.results)
        if main_proc and args.tensorboard:  # Previous scores to tensorboard logs
            tensorboard_logger.load_previous_values(state.epoch, state.results)
    else:
        # Initialise new model training
        with open(args.labels_path) as label_file:
            labels = json.load(label_file)

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)

        state = TrainingState(model=model)
        state.init_results_tracking(epochs=args.epochs)

    # Data setup
    evaluation_decoder = GreedyDecoder(model.labels)  # Decoder used for validation
    train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                       manifest_filepath=args.train_manifest,
                                       labels=model.labels,
                                       normalize=True,
                                       speed_volume_perturb=args.speed_volume_perturb,
                                       spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=args.val_manifest,
                                      labels=model.labels,
                                      normalize=True,
                                      speed_volume_perturb=False,
                                      spec_augment=False)
    if not args.distributed:
        train_sampler = DSRandomSampler(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        start_index=state.training_step)
    else:
        train_sampler = DSElasticDistributedSampler(dataset=train_dataset,
                                                    batch_size=args.batch_size,
                                                    start_index=state.training_step)
    train_loader = AudioDataLoader(dataset=train_dataset,
                                   num_workers=args.num_workers,
                                   batch_sampler=train_sampler)
    test_loader = AudioDataLoader(dataset=test_dataset,
                                  num_workers=args.num_workers,
                                  batch_size=args.batch_size)

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters,
                                lr=args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=1e-5)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)
    if state.optim_state is not None:
        optimizer.load_state_dict(state.optim_state)
        amp.load_state_dict(state.amp_state)

    # Track states for optimizer/amp
    state.track_optim_state(optimizer)
    state.track_amp_state(amp)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[device_id])
    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion = CTCLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(state.epoch, args.epochs):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        state.set_epoch(epoch=epoch)
        train_sampler.set_epoch(epoch=epoch)
        train_sampler.reset_training_step(training_step=state.training_step)
        for i, (data) in enumerate(train_loader, start=state.training_step):
            state.set_training_step(training_step=i)
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.to(device)

            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH

            float_out = out.float()  # ensure float32 for loss
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_value = loss.item()

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()

                # compute gradient
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
                optimizer.step()
            else:
                print(error)
                print('Skipping grad update')
                loss_value = 0

            state.avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                (epoch + 1), (i + 1), len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))

            if main_proc and args.checkpoint_per_iteration:
                checkpoint_handler.save_iter_checkpoint_model(epoch=epoch, i=i, state=state)
            del loss, out, float_out

        state.avg_loss /= len(train_dataset)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=state.avg_loss))

        with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=evaluation_decoder,
                                             target_decoder=evaluation_decoder)

        state.add_results(epoch=epoch,
                          loss_result=state.avg_loss,
                          wer_result=wer,
                          cer_result=cer)

        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))

        if main_proc and args.visdom:
            visdom_logger.update(epoch, state.result_state)
        if main_proc and args.tensorboard:
            tensorboard_logger.update(epoch, state.result_state, model.named_parameters())

        if main_proc and args.checkpoint:  # Save epoch checkpoint
            checkpoint_handler.save_checkpoint_model(epoch=epoch, state=state)
        # anneal lr
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / args.learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

        if main_proc and (state.best_wer is None or state.best_wer > wer):
            checkpoint_handler.save_best_model(epoch=epoch, state=state)
            state.set_best_wer(wer)
            state.reset_avg_loss()
        state.reset_training_step()  # Reset training step for next epoch
