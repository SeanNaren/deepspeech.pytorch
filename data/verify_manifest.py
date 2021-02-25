import argparse
import json
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("manifests", metavar="m", nargs="+", help="Manifests to verify")
args = parser.parse_args()

def main():
    for manifest_path in tqdm(args.manifests):
        with open(manifest_path, "r") as manifest_file:
            manifest_json = json.load(manifest_file)

        root_path = Path(manifest_json['root_path'])
        for sample in tqdm(manifest_json['samples']):
            assert (root_path / Path(sample['wav_path'])).exists(), f"{sample['wav_path']} does not exist"
            assert (root_path / Path(sample['transcript_path'])).exists(), f"{sample['transcript_path']} does not exist"


if __name__ == "__main__":
    main()

