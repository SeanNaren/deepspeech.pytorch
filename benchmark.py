import argparse
import json
import time
import torch
from tqdm import tqdm
from warpctc_pytorch import CTCLoss
from tqdm import trange
from model import DeepSpeech, supported_rnns
import torch.distributed as dist
import torch.utils.data.distributed

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, help='Size of input')
parser.add_argument('--seconds', type=int, default=15,
                    help='The size of the fake input in seconds using default stride of 0.01, '
                         '15s is usually the maximum duration')
parser.add_argument('--dry-runs', type=int, default=2, help='Dry runs before measuring performance')
parser.add_argument('--runs', type=int, default=5, help='How many benchmark runs to measure performance')
parser.add_argument('--labels-path', default='labels.json', help='Path to the labels to infer over in the model')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--num-samples', default=1024, type=int, help='Number of samples to go through')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int, help='The rank of this process')
args = parser.parse_args()

args.distributed = args.world_size > 1
if args.distributed:
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

if args.distributed:
    input_data = torch.randn(int(args.num_samples / args.world_size), 1, 161, args.seconds * 100).cuda()
else:
    input_data = torch.randn(args.num_samples, 1, 161, args.seconds * 100).cuda()
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
                   rnn_type=supported_rnns[rnn_type])

print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

parameters = model.parameters()
optimizer = torch.optim.SGD(parameters, lr=3e-4,
                            momentum=0.9, nesterov=True)
if args.distributed:
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
else:
    model = torch.nn.DataParallel(model).cuda()

criterion = CTCLoss()

seconds = int(args.seconds)
batch_size = int(args.batch_size)


def iteration(inputs):
    # targets, align half of the audio
    targets = torch.ones(int(batch_size * ((seconds * 100) / 2)))
    target_sizes = torch.empty(batch_size, dtype=torch.int).fill_(int((seconds * 100) / 2))
    input_percentages = torch.ones(batch_size).fill_(1)

    out = model(inputs)
    out = out.transpose(0, 1)  # TxNxH

    seq_length = out.size(0)
    sizes = input_percentages.mul_(int(seq_length)).int()
    loss = criterion(out, targets, sizes, target_sizes)
    loss = loss / inputs.size(0)  # average the loss by minibatch
    # compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    del loss
    del out


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
