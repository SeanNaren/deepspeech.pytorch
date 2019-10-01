import argparse
import json
import sys
from multiprocessing.pool import Pool

import numpy as np
import torch
from tqdm import tqdm

from decoder import BeamCTCDecoder
from model import DeepSpeech
from opts import add_decoder_args

parser = argparse.ArgumentParser(description='Tune an ARPA LM based on a pre-trained acoustic model output')
parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                    help='Path to model file created by training')
parser.add_argument('--saved-output', default="", type=str, help='Path to output from test.py')
parser.add_argument('--num-workers', default=16, type=int, help='Number of parallel decodes to run')
parser.add_argument('--output-path', default="tune_results.json", help="Where to save tuning results")
parser.add_argument('--lm-alpha-from', default=0.0, type=float, help='Language model weight start tuning')
parser.add_argument('--lm-alpha-to', default=3.0, type=float, help='Language model weight end tuning')
parser.add_argument('--lm-beta-from', default=0.0, type=float,
                    help='Language model word bonus (all words) start tuning')
parser.add_argument('--lm-beta-to', default=0.5, type=float,
                    help='Language model word bonus (all words) end tuning')
parser.add_argument('--lm-num-alphas', default=45, type=float, help='Number of alpha candidates for tuning')
parser.add_argument('--lm-num-betas', default=8, type=float, help='Number of beta candidates for tuning')
parser = add_decoder_args(parser)
args = parser.parse_args()

if args.lm_path is None:
    print("error: LM must be provided for tuning")
    sys.exit(1)

model = DeepSpeech.load_model(args.model_path)

saved_output = np.load(args.saved_output)


def init(beam_width, blank_index, lm_path):
    global decoder
    decoder = BeamCTCDecoder(model.labels, lm_path=lm_path, beam_width=beam_width, num_processes=args.lm_workers,
                             blank_index=blank_index)


def decode_dataset(params):
    lm_alpha, lm_beta = params
    global decoder
    decoder._decoder.reset_params(lm_alpha, lm_beta)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    for out, sizes, target_strings in saved_output:
        out = torch.Tensor(out).float()
        sizes = torch.Tensor(sizes).int()
        decoded_output, _, = decoder.decode(out, sizes)
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_cer += cer_inst
            total_wer += wer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference)

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars

    return [lm_alpha, lm_beta, wer * 100, cer * 100]


if __name__ == '__main__':
    p = Pool(args.num_workers, init, [args.beam_width, model.labels.index('_'), args.lm_path])

    cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
    cand_betas = np.linspace(args.lm_beta_from, args.lm_beta_to, args.lm_num_betas)
    params_grid = [(float(alpha), float(beta)) for alpha in cand_alphas
                   for beta in cand_betas]

    scores = []
    for params in tqdm(p.imap(decode_dataset, params_grid), total=len(params_grid)):
        scores.append(list(params))
    print("Saving tuning results to: {}".format(args.output_path))
    with open(args.output_path, "w") as fh:
        json.dump(scores, fh)
