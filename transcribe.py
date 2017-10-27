import argparse

from decoder import GreedyDecoder

from torch.autograd import Variable

from data.data_loader import SpectrogramParser
from model import DeepSpeech

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--audio_path', default='audio.wav',
                    help='Audio file to predict on')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam"], type=str, help="Decoder to use")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--trie_path', default=None, type=str,
                       help='Path to an (optional) trie dictionary for use with beam search (req\'d with LM)')
beam_args.add_argument('--lm_alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--lm_beta1', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--lm_beta2', default=1, type=float, help='Language model word bonus (IV words)')
parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
args = parser.parse_args()

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, top_paths=1, space_index=labels.index(' '),
                                 blank_index=labels.index('_'), lm_path=args.lm_path,
                                 trie_path=args.trie_path, lm_alpha=args.lm_alpha, lm_beta1=args.lm_beta1,
                                 lm_beta2=args.lm_beta2)
    else:
        decoder = GreedyDecoder(labels, space_index=labels.index(' '), blank_index=labels.index('_'))

    parser = SpectrogramParser(audio_conf, normalize=True)

    spect = parser.parse_audio(args.audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    out = model(Variable(spect, volatile=True))
    out = out.transpose(0, 1)  # TxNxH
    decoded_output, decoded_offsets = decoder.decode(out.data)

    print(decoded_output[0])
    if args.offsets:
        print(decoded_offsets[0])
