import argparse
import json
import os

import torch

from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.inference import transcribe
from deepspeech_pytorch.opts import add_decoder_args, add_inference_args
from deepspeech_pytorch.utils import load_model, load_decoder


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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    arg_parser = add_inference_args(arg_parser)
    arg_parser.add_argument('--audio-path',
                            default='audio.wav',
                            help='Audio file to predict on')
    arg_parser.add_argument('--offsets',
                            dest='offsets',
                            action='store_true',
                            help='Returns time offset information')
    arg_parser = add_decoder_args(arg_parser)
    args = arg_parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    decoder = load_decoder(decoder_type=args.decoder,
                           labels=model.labels,
                           lm_path=args.lm_path,
                           alpha=args.alpha,
                           beta=args.beta,
                           cutoff_top_n=args.cutoff_top_n,
                           cutoff_prob=args.cutoff_prob,
                           beam_width=args.beam_width,
                           lm_workers=args.lm_workers)

    spect_parser = SpectrogramParser(audio_conf=model.audio_conf,
                                     normalize=True)

    decoded_output, decoded_offsets = transcribe(audio_path=args.audio_path,
                                                 spect_parser=spect_parser,
                                                 model=model,
                                                 decoder=decoder,
                                                 device=device,
                                                 use_half=args.half)
    print(json.dumps(decode_results(decoded_output, decoded_offsets)))
