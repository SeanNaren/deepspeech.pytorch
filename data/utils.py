import argparse
import io
import os

import subprocess

parser = argparse.ArgumentParser(description='Creates training and testing manifests')
parser.add_argument('--root_path', default='an4_dataset', help='Path to the dataset')

"""
We need to add progress bars like will did in gen audio. Just copy that code.
We also need a call (basically the same in the dataloader, a find that gives us the total number of wav files)
"""


def create_manifest(data_path, tag, ordered=True):
    manifest_path = '%s_manifest.csv' % tag
    file_paths = []
    with os.popen('find %s -type f -name "*.wav"' % data_path) as pipe:
        for file_path in pipe:
            file_paths.append(file_path.strip())
    if ordered:
        print("Sorting files by length...")

        def func(element):
            output = subprocess.check_output(
                ['soxi -D %s' % element.strip()],
                shell=True
            )
            return float(output)

        file_paths.sort(key=func)
    with io.FileIO(manifest_path, "w") as file:
        for wav_path in file_paths:
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample)
