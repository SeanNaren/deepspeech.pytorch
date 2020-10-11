from __future__ import print_function

import fnmatch
import io
import os
from multiprocessing import Pool
from typing import Optional

import sox
from tqdm import tqdm


def create_manifest(
        data_path: str,
        output_name: str,
        manifest_path: str,
        num_workers: int,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = order_and_prune_files(
        file_paths=file_paths,
        min_duration=min_duration,
        max_duration=max_duration,
        num_workers=num_workers
    )
    os.makedirs(manifest_path, exist_ok=True)
    with io.FileIO(os.path.join(manifest_path, output_name), "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')


def _duration_file_path(path):
    return path, sox.file_info.duration(path)


def order_and_prune_files(
        file_paths,
        min_duration,
        max_duration,
        num_workers):
    print("Gathering durations...")
    with Pool(processes=num_workers) as p:
        duration_file_paths = list(tqdm(p.imap(_duration_file_path, file_paths), total=len(file_paths)))
    print("Sorting manifests...")
    if min_duration and max_duration:
        print("Pruning manifests between %d and %d seconds" % (min_duration, max_duration))
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               min_duration <= duration <= max_duration]

    return [x[0] for x in duration_file_paths]  # Remove durations
