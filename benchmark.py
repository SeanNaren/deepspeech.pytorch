import argparse
import time
import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from data.utils import update_progress
from model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='Size of input')
parser.add_argument('--seconds', type=int, default=15,
                    help='The size of the fake input in seconds using default stride of 0.01, '
                         '15s is usually the maximum duration')
parser.add_argument('--dry_runs', type=int, default=20, help='Dry runs before measuring performance')
parser.add_argument('--runs', type=int, default=20, help='Hidden size of RNNs')
parser.add_argument('--hidden_size', default=400, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
args = parser.parse_args()

input = torch.randn(args.batch_size, 1, 161, args.seconds * 100).cuda()

rnn_type = args.rnn_type.lower()
assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                   nb_layers=args.hidden_layers, num_classes=29,
                   rnn_type=supported_rnns[rnn_type],
                   sample_rate=args.sample_rate, window_size=args.window_size)

parameters = model.parameters()
optimizer = torch.optim.SGD(parameters, lr=3e-4,
                            momentum=0.9, nesterov=True)
model = torch.nn.DataParallel(model).cuda()
criterion = CTCLoss()


seconds = int(args.seconds)
batch_size = int(args.batch_size)

def iteration(input_data):
    target = torch.IntTensor(int(batch_size * ((seconds * 100) / 2))).fill_(1)  # targets, align half of the audio
    target_size = torch.IntTensor(batch_size).fill_(int((seconds * 100) / 2))
    input_percentages = torch.IntTensor(batch_size).fill_(1)

    inputs = Variable(input_data)
    target_sizes = Variable(target_size)
    targets = Variable(target)
    start = time.time()
    out = model(inputs)
    out = out.transpose(0, 1)  # TxNxH

    seq_length = out.size(0)
    sizes = Variable(input_percentages.mul_(int(seq_length)).int())
    loss = criterion(out, targets, sizes, target_sizes)
    loss = loss / inputs.size(0)  # average the loss by minibatch
    # compute gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    return start, end


def run_benchmark(input_data):
    print("Running dry runs...")
    for n in range(args.dry_runs):
        iteration(input_data)
        update_progress(n / (float(args.dry_runs) - 1))

    print("\n Running measured runs...")
    running_time = 0
    for n in range(args.runs):
        start, end = iteration(input_data)
        running_time += end - start
        update_progress(n / (float(args.runs) - 1))

    return running_time / float(args.runs)


run_time = run_benchmark(input)

print("\n Average run time: %.2fs" % run_time)
