import argparse
import io
import json
import os
from pathlib import Path

from tqdm import tqdm


parser = argparse.ArgumentParser(description="Merges all manifest files in specified folder.")
parser.add_argument("manifests", metavar="m", nargs="+", help="Path to all manifest files you want to merge.")
parser.add_argument("-e", "--extension", default="wav", type=str, help="Audio file extension")
parser.add_argument("--name", default="merged_manifest", type=str, help="Merged dataset name")
parser.add_argument("--out", default="./", type=str, help="Output directory")
args = parser.parse_args()


def main():
    new_manifest_path = Path(args.out) / Path(args.name)
    new_manifest_path.mkdir(parents=True, exist_ok=True)
    (new_manifest_path / args.extension).mkdir(parents=True, exist_ok=True)
    (new_manifest_path / 'txt').mkdir(parents=True, exist_ok=True)

    new_manifest = {
        'root_path': new_manifest_path.absolute().as_posix(),
        'samples': []
    }
    for manifest in tqdm(args.manifests, desc="Manifests"):
        with open(manifest, "r") as manifest_file:
            manifest_json = json.load(manifest_file)

        root_path = Path(manifest_json['root_path'])
        for sample in tqdm(manifest_json['samples'], desc="Samples"):
            try:
                old_audio_path = root_path / Path(sample['wav_path'])
                new_audio_path = new_manifest_path.absolute() / Path(sample['wav_path'])
                os.symlink(old_audio_path, new_audio_path)
                old_txt_path = root_path / Path(sample['transcript_path'])
                new_txt_path = new_manifest_path.absolute() / Path(sample['transcript_path'])
                os.symlink(old_txt_path, new_txt_path)
            except FileExistsError:
                continue

        new_manifest['samples'] += manifest_json['samples']
    
    with open(f"{args.name}_manifest.json", "w") as json_file:
        json.dump(new_manifest, json_file)


if __name__ == "__main__":
    main()
