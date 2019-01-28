import argparse

import numpy as np
import torch
from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
parser = add_decoder_args(parser)
parser.add_argument('--save-output', action="store_true", help="Saves output of model from test")
args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.cuda)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
                                      labels=model.labels, normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes)

        if args.save_output:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy()))

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference)
            if args.verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()), "CER:", float(cer_inst) / len(reference), "\n")

    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))
    if args.save_output:
        np.save(args.output_path, output_data)
