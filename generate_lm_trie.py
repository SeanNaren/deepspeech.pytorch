import pytorch_ctc
import json
import argparse

parser = argparse.ArgumentParser(description='LM Trie Generation')
parser.add_argument('--labels', help='path to label json file', default='labels.json')
parser.add_argument('--dictionary', help='path to text dictionary (one word per line)', default='vocab.txt')
parser.add_argument('--kenlm', help='path to binary kenlm language model', default="lm.kenlm")
parser.add_argument('--trie', help='path of trie to output', default='vocab.trie')


def main():
    args = parser.parse_args()
    with open(args.labels, "r") as fh:
        label_data = json.load(fh)

    labels = ''.join(label_data)

    pytorch_ctc.generate_lm_trie(args.dictionary, args.kenlm, args.trie, labels, labels.index('_'), labels.index(' '))


if __name__ == '__main__':
    main()
