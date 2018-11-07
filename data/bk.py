import argparse
import os
import io
import shutil
import tarfile
import wget

from utils import create_manifest, create_manifest_for_bk_dataset

parser = argparse.ArgumentParser(description='Processes bk dataset.')
parser.add_argument('--target-dir', default='/media/zinzin/CA92B91D92B90F47/bk_dataset', help='Path to save dataset')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes training samples longer than the max duration (given in seconds, default 15)')
args = parser.parse_args()

def main():
    train_path = args.target_dir + '/train/'
    val_path = args.target_dir + '/val/'
    test_path = args.target_dir + '/test/'
    print ('\n', 'Creating manifests...')
    create_manifest_for_bk_dataset(train_path, 'bk_train_manifest.csv', args.min_duration, args.max_duration)
    create_manifest_for_bk_dataset(val_path, 'bk_val_manifest.csv')
    create_manifest_for_bk_dataset(test_path, 'bk_test_manifest.csv')


if __name__ == '__main__':
    main()
