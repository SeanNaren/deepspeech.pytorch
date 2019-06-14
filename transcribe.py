import argparse
import warnings

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

from decoder import GreedyDecoder

import torch

from data.data_loader import SpectrogramParser
import os.path
import json


def decode_results(decoded_output, decoded_offsets):
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

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe(audio_path, spect_parser, model, decoder, device, use_half):
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    arg_parser = add_inference_args(arg_parser)
    arg_parser.add_argument('--audio-path', default='audio.wav',
                              help='Audio file to predict on')
    arg_parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
    arg_parser = add_decoder_args(arg_parser)
    args = arg_parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

    decoded_output, decoded_offsets = transcribe(audio_path=args.audio_path,
                                                 spect_parser=spect_parser,
                                                 model=model,
                                                 decoder=decoder,
                                                 device=device,
                                                 use_half=args.half)
    print(json.dumps(decode_results(decoded_output, decoded_offsets)))
