import argparse
import json

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from decoder import ArgMaxDecoder
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
args = parser.parse_args()

if __name__ == '__main__':
    package = torch.load(args.model_path)
    model = DeepSpeech(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                       num_classes=package['nout'])
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(package['state_dict'])
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window)
    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    decoder = ArgMaxDecoder(labels)
    parser = SpectrogramParser(audio_conf, normalize=True)
    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect))
    out = out.transpose(0, 1)  # TxNxH
    decoded_output = decoder.decode(out.data)
    print(decoded_output[0])
