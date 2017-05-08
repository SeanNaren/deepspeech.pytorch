import argparse

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from decoder import ArgMaxDecoder
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
args = parser.parse_args()

if __name__ == '__main__':
    package = torch.load(args.model_path)
    model = DeepSpeech.load_model(package, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    decoder = ArgMaxDecoder(labels)
    parser = SpectrogramParser(audio_conf, normalize=True)
    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect))
    out = out.transpose(0, 1)  # TxNxH
    decoded_output = decoder.decode(out.data)
    print(decoded_output[0])
