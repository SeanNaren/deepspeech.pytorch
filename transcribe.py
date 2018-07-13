import argparse
import warnings

from opts import add_decoder_args, add_inference_args

warnings.simplefilter('ignore')

from decoder import GreedyDecoder

import torch

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os.path
import json

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--audio-path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
parser = add_decoder_args(parser)
args = parser.parse_args()


def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }
    results['_meta']['acoustic_model'].update(DeepSpeech.get_meta(model))

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe(audio_path, parser, model, decoder, cuda=False):
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    if cuda:
        spect = spect.cuda()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    model = DeepSpeech.load_model(args.model_path)
    if args.cuda:
        model.cuda()
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)

    decoded_output, decoded_offsets = transcribe(args.audio_path, parser, model, decoder, args.cuda)
    print(json.dumps(decode_results(model, decoded_output, decoded_offsets)))
