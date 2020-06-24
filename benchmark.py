import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.utils.data.distributed
from apex import amp
from apex.parallel import DistributedDataParallel
from tqdm import tqdm
from tqdm import trange
from warpctc_pytorch import CTCLoss

from deepspeech_pytorch.model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser(description="Benchmark script to check stability of training with CUDA. "
                                             "Assumes a few hardcoded defaults which shouldn't make a large difference")
parser.add_argument('--batch-size', type=int, default=32, help='Size of input')
parser.add_argument('--seconds', type=int, default=15,
                    help='The size of the fake input in seconds using default stride of 0.01, '
                         '15s is usually the maximum duration')
parser.add_argument('--dry-runs', type=int, default=2, help='Dry runs before measuring performance')
parser.add_argument('--runs', type=int, default=5, help='How many benchmark runs to measure performance')
parser.add_argument('--labels-path', default='labels.json', help='Path to the labels to infer over in the model')
parser.add_argument('--hidden-size', default=1024, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--num-samples', default=1024, type=int, help='Number of samples to go through')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--opt-level', type=str,
                    help='Apex optimization level,'
                         'check https://nvidia.github.io/apex/amp.html for more information')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None,
                    help='Overrides Apex keep_batch_norm_fp32 flag')
parser.add_argument('--loss-scale', default=1,
                    help='Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
args = parser.parse_args()
device = torch.device("cuda")

args.distributed = os.environ.get("LOCAL_RANK")  # If local rank exists, distributed env
if args.distributed:
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    print(f"Setting CUDA Device to {device_id}")

    dist.init_process_group(backend=args.dist_backend)

if args.distributed:
    input_data = torch.randn(int(args.num_samples / dist.get_world_size()), 1, 161, args.seconds * 100)
else:
    input_data = torch.randn(args.num_samples, 1, 161, args.seconds * 100)
input_data = input_data.to(device)
input_data = torch.chunk(input_data, int(len(input_data) / args.batch_size))

rnn_type = args.rnn_type.lower()
assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

with open(args.labels_path) as label_file:
    labels = str(''.join(json.load(label_file)))

audio_conf = dict(sample_rate=args.sample_rate,
                  window_size=args.window_size)

model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                   nb_layers=args.hidden_layers,
                   audio_conf=audio_conf,
                   labels=labels,
                   rnn_type=supported_rnns[rnn_type],
                   bidirectional=args.bidirectional)

model = model.to(device)
parameters = model.parameters()
optimizer = torch.optim.SGD(parameters, lr=3e-4, momentum=0.9, nesterov=True, weight_decay=1e-5)

model, optimizer = amp.initialize(model, optimizer,
                                  opt_level=args.opt_level,
                                  keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                  loss_scale=args.loss_scale)
print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

if args.distributed:
    model = DistributedDataParallel(model, device_ids=[device_id])

criterion = CTCLoss()

seconds = int(args.seconds)
batch_size = int(args.batch_size)


def iteration(inputs):
    # targets, align half of the audio
    targets = torch.ones(int(batch_size * ((seconds * 100) / 2)))
    target_sizes = torch.empty(batch_size, dtype=torch.int).fill_(int((seconds * 100) / 2))
    input_percentages = torch.ones(batch_size).fill_(1)
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

    out, output_sizes = model(inputs, input_sizes)
    out = out.transpose(0, 1)  # TxNxH

    float_out = out.float()  # ensure float32 for loss
    loss = criterion(float_out, targets, output_sizes, target_sizes)
    loss = loss / inputs.size(0)  # average the loss by minibatch
    optimizer.zero_grad()
    # compute gradient
    optimizer.zero_grad()

    # compute gradient
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 400)
    optimizer.step()
    del loss, out, float_out


def run_benchmark():
    print("Running dry runs...")
    for n in trange(args.dry_runs):
        for data in tqdm(input_data, total=len(input_data)):
            iteration(data)

    print("\n Running measured runs...")
    running_time = 0
    for n in trange(args.runs):
        start_time = time.time()
        for data in tqdm(input_data, total=len(input_data)):
            iteration(data)
        end_time = time.time()
        running_time += (end_time - start_time)

    return running_time / float(args.runs)


run_time = run_benchmark()

print("\n Average run time: %.2fs" % run_time)
