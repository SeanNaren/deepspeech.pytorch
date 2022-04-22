import argparse
import os
import tarfile

import wget

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest


def download_an4(target_dir: str,
                 manifest_dir: str,
                 min_duration: float,
                 max_duration: float,
                 num_workers: int):
    raw_tar_path = 'an4.tar.gz'
    if not os.path.exists(raw_tar_path):
        wget.download('https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4.tar.gz')
    tar = tarfile.open('an4.tar.gz')
    os.makedirs(target_dir, exist_ok=True)
    tar.extractall(target_dir)
    train_path = target_dir + '/train/'
    val_path = target_dir + '/val/'
    test_path = target_dir + '/test/'

    print('Creating manifests...')
    create_manifest(data_path=train_path,
                    output_name='an4_train_manifest.json',
                    manifest_path=manifest_dir,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    num_workers=num_workers)
    create_manifest(data_path=val_path,
                    output_name='an4_val_manifest.json',
                    manifest_path=manifest_dir,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    num_workers=num_workers)
    create_manifest(data_path=test_path,
                    output_name='an4_test_manifest.json',
                    manifest_path=manifest_dir,
                    num_workers=num_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes and downloads an4.')
    parser = add_data_opts(parser)
    parser.add_argument('--target-dir', default='an4_dataset/', help='Path to save dataset')
    args = parser.parse_args()
    assert args.sample_rate == 16000, "AN4 only supports sample rate of 16000 currently."
    download_an4(
        target_dir=args.target_dir,
        manifest_dir=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )
