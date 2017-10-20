import pytorch_ctc
import json
import argparse

parser = argparse.ArgumentParser(description='LM Dictionary Generation')
parser.add_argument('--labels', help='path to label json file', default='labels.json')
parser.add_argument('--dict_path', help='path to text dictionary (one word per line)', default='vocab.txt')
parser.add_argument('--model_path', help='path to the kenlm language model', default="lm.kenlm")
parser.add_argument('--output_path', help='path of output dictionary', default='vocab.dic')


def main():
    args = parser.parse_args()
    with open(args.labels, "r") as fh:
        label_data = json.load(fh)

    labels = ''.join(label_data)

    pytorch_ctc.generate_lm_dict(args.dict_path, args.model_path, args.output_path, labels, labels.index('_'),
                                 labels.index(' '))


if __name__ == '__main__':
    main()
