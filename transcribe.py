import argparse
import warnings
warnings.simplefilter('ignore')

from decoder import GreedyDecoder

from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os.path
import numpy as np
import json

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--top_paths', default=1, type=int, help='Number of paths to return')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--dict_path', default=None, type=str,
                       help='Path to an (optional) trie dictionary for use with beam search (req\'d with LM)')
beam_args.add_argument('--lm_alpha', default=1.9, type=float, help='Language model weight')
beam_args.add_argument('--lm_beta', default=0, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--label_size', default=0, type=int, help='Label selection size controls how many items in '
                                                                 'each beam are passed through to the beam scorer')
beam_args.add_argument('--label_margin', default=-1, type=float, help='Controls difference between minimal input score '
                                                                      'for an item to be passed to the beam scorer.')
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
parser.add_argument('--word_json', dest='word_json', action='store_true', help='Return word-level results as a json object')
args = parser.parse_args()

def word_decode(decoder, data, time_div=50, window=5, model=None):
    strings, aligns, conf, char_probs = decoder.decode(data)

    results = {
        "one_best": "",
        "num_paths": decoder._top_n,
        "top_paths": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path),
                **DeepSpeech.get_meta(model)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
                "dict": os.path.basename(args.dict_path) if args.dict_path else None
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "dict": args.dict_path is not None,
                "alpha": args.lm_alpha if args.lm_path is not None else None,
                "beta": args.lm_beta if args.lm_path is not None else None,
                "type": args.decoder,
                "label_size": args.label_size,
                "label_margin": args.label_margin
            }
        }
    }

    for pi in range(len(strings)):
        for i in range(len(strings[pi])):
            path = {"rank": pi, "conf": float(conf[pi][i]), "tokens": []}
            word = ''
            word_prob = 0.0
            start_idx = -1
            for idx, c in enumerate(strings[pi][i]):
                if c == ' ' and word != '':
                    start_align = aligns[pi][i][start_idx]
                    end_align = aligns[pi][i][idx-1]+window
                    path['tokens'].append({"token": word, "start": start_align/time_div, "end": end_align/time_div, "conf": word_prob})
                    word = ''
                    start_idx = -1
                else:
                    if start_idx == -1:
                        start_idx = idx
                    word += c
                    word_prob += char_probs[pi][i][idx]
            if word != '':
                path['tokens'].append({"token": word, "start": (aligns[pi][i][start_idx])/time_div, "end": (aligns[pi][i][len(strings[pi][i])-1]+window)/time_div, "conf": word_prob})
        results['top_paths'].append(path)
    if len(results['top_paths']) > 0:
        results['one_best'] = " ".join([x['token'] for x in results['top_paths'][0]['tokens']])
    return results

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, top_paths=args.top_paths,
                                 space_index=labels.index(' '),
                                 blank_index=labels.index('_'), lm_path=args.lm_path,
                                 dict_path=args.dict_path, lm_alpha=args.lm_alpha, lm_beta=args.lm_beta,
                                 label_size=args.label_size, label_margin=args.label_margin)
    else:
        decoder = GreedyDecoder(labels, space_index=labels.index(' '), blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)

    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect, volatile=True))
    out = out.transpose(0, 1)  # TxNxH

    if args.word_json:
        results = word_decode(decoder, out.data, model=model, window=1/(10*audio_conf['window_size']), time_div=1/audio_conf['window_size'])
        print(json.dumps(results))
    else:
        decoded_output, decoded_offsets, confs, char_probs = decoder.decode(out.data)
        for pi in range(args.top_paths):
            print(decoded_output[pi][0])
            if args.offsets:
                print(decoded_offsets[pi][0])
