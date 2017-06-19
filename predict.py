import argparse
import sys
import time

import torch
from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from decoder import GreedyDecoder, BeamCTCDecoder
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech prediction')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
args = parser.parse_args()

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, top_paths=1, blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)

    t0 = time.time()
    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect, volatile=True))
    out = out.transpose(0, 1)  # TxNxH
    decoded_output = decoder.decode(out.data)
    t1 = time.time()

    print(decoded_output[0])
    print("Decoded {0:.2f} seconds of audio in {1:.2f} seconds".format(spect.size(3)*audio_conf['window_stride'], t1-t0), file=sys.stderr)
