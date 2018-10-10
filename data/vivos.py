import argparse
import os
import io
import shutil
import tarfile
import wget

from utils import create_manifest

parser = argparse.ArgumentParser(description='Processes vivos dataset.')
parser.add_argument('--target-dir', default='vivos_dataset/', help='Path to save dataset')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes training samples longer than the max duration (given in seconds, default 15)')
args = parser.parse_args()

def main():
    train_path = args.target_dir + '/train/'
    test_path = args.target_dir + '/test/'
    test2_path = args.target_dir + '/test2/'
    print ('\n', 'Creating manifests...')
    create_manifest(train_path, 'vivos_train_manifest.csv', args.min_duration, args.max_duration)
    create_manifest(test_path, 'vivos_test_manifest.csv')
    create_manifest(test2_path, 'vivos_test2_manifest.csv')


if __name__ == '__main__':
    main()
