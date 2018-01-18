from __future__ import print_function

import argparse
import io
import os

from tqdm import tqdm
from utils import order_and_prune_files

parser = argparse.ArgumentParser(description='Merges all manifest CSV files in specified folder.')
parser.add_argument('--merge-dir', default='manifests/', help='Path to all manifest files you want to merge')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes any samples shorter than the min duration (given in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes any samples longer than the max duration (given in seconds, default 15)')
parser.add_argument('--output-path', default='merged_manifest.csv', help='Output path to merged manifest')

args = parser.parse_args()

file_paths = []
for file in os.listdir(args.merge_dir):
    if file.endswith(".csv"):
        with open(os.path.join(args.merge_dir, file), 'r') as fh:
            file_paths += fh.readlines()
file_paths = [file_path.split(',')[0] for file_path in file_paths]
file_paths = order_and_prune_files(file_paths, args.min_duration, args.max_duration)
with io.FileIO(args.output_path, "w") as file:
    for wav_path in tqdm(file_paths, total=len(file_paths)):
        transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
        sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
        file.write(sample.encode('utf-8'))
