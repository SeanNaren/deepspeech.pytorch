import ctcdecode
import json
import argparse

parser = argparse.ArgumentParser(description='LM Dictionary Generation')
parser.add_argument('--labels', help='path to label json file', default='labels.json')
parser.add_argument('--dict_path', help='path to text dictionary (one word per line)', default='vocab.txt')
parser.add_argument('--lm_path', help='path to the kenlm language model (optional)', default=None)
parser.add_argument('--output_path', help='path of output dictionary trie', default='vocab.dic')


def main():
    args = parser.parse_args()
    with open(args.labels, "r") as fh:
        label_data = json.load(fh)

    labels = ''.join(label_data)

    ctcdecode.generate_lm_dict(args.dict_path, args.output_path, labels, kenlm_path=args.lm_path,
                               blank_index=labels.index('_'), space_index=labels.index(' '))


if __name__ == '__main__':
    main()
