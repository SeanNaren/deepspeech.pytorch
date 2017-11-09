import argparse

from torch.autograd import Variable
from tqdm import tqdm

from decoder import GreedyDecoder, BeamCTCDecoder

from data.data_loader import SpectrogramDataset, AudioDataLoader
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--logits', default=None, type=str, help='Path to logits from test.py')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--trie_path', default=None, type=str,
                       help='Path to an (optional) trie dictionary for use with beam search (req\'d with LM)')
beam_args.add_argument('--lm_alpha_from', default=1, type=float, help='Language model weight start tuning')
beam_args.add_argument('--lm_alpha_to', default=3.2, type=float, help='Language model weight end tuning')
beam_args.add_argument('--lm_beta_from', default=0.0, type=float, help='Language model word bonus (all words) start tuning')
beam_args.add_argument('--lm_beta_to', default=0.45, type=float, help='Language model word bonus (all words) end tuning')
beam_args.add_argument('--lm_num_alphas', default=45, type=float, help='Number of alpha candidates for tuning')
beam_args.add_argument('--lm_num_betas', default=8, type=float, help='Number of beta candidates for tuning')
beam_args.add_argument('--label_size', default=0, type=int, help='Label selection size controls how many items in '
                                                                 'each beam are passed through to the beam scorer')
beam_args.add_argument('--label_margin', default=-1, type=float, help='Controls difference between minimal input score '
                                                                      'for an item to be passed to the beam scorer.')
args = parser.parse_args()

def decode_dataset(logits, test_loader, lm_alpha, lm_beta, labels):
    target_decoder = GreedyDecoder(labels, space_index=labels.index(' '), blank_index=labels.index('_'))
    decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, top_paths=1, space_index=labels.index(' '),
                             blank_index=labels.index('_'), lm_path=args.lm_path,
                             trie_path=args.trie_path, lm_alpha=lm_alpha, lm_beta=lm_beta,
                             label_size=args.label_size, label_margin=args.label_margin)
    total_cer, total_wer = 0, 0
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out = torch.from_numpy(logits[i][0])
        sizes = torch.from_numpy(logits[i][1])

        decoded_output, _, _, _ = decoder.decode(out.data, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            wer_inst = decoder.wer(decoded_output[0][x], target_strings[x]) / float(len(target_strings[x].split()))
            cer_inst = decoder.cer(decoded_output[0][x], target_strings[x]) / float(len(target_strings[x]))
            wer += wer_inst
            cer += cer_inst
        total_cer += cer
        total_wer += wer

    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)

    return (lm_alpha, lm_beta, wer, cer)


if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                      normalize=True)

    logits = np.load(args.logits)
    batch_size = len(logits[0][0])
    test_loader = AudioDataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers)

    results = decode_dataset(logits, test_loader, lm_alpha, lm_beta, labels)
    print(results)
