from __future__ import print_function

import json
import os
from multiprocessing import Pool
from pathlib import Path
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
        file_extension: str = "wav"):
    data_path = os.path.abspath(data_path)
    file_paths = list(Path(data_path).rglob(f"*.{file_extension}"))
    file_paths = order_and_prune_files(
        file_paths=file_paths,
        min_duration=min_duration,
        max_duration=max_duration,
        num_workers=num_workers
    )

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)

    manifest = {
        'root_path': data_path,
        'samples': []
    }
    for wav_path in tqdm(file_paths, total=len(file_paths)):
        wav_path = wav_path.relative_to(data_path)
        transcript_path = wav_path.parent.with_name("txt") / wav_path.with_suffix(".txt").name
        manifest['samples'].append({
            'wav_path': wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })

    output_path.write_text(json.dumps(manifest), encoding='utf8')


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

    total_duration = sum([x[1] for x in duration_file_paths])
    print(f"Total duration of split: {total_duration:.4f}s")
    return [x[0] for x in duration_file_paths]  # Remove durations
