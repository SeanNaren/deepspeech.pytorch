import argparse
import io
import os

parser = argparse.ArgumentParser(description='Creates training and testing manifests')
parser.add_argument('--root_path', default='dataset', help='Path to the dataset')


def create_manifest(data_path, tag):
    manifest_path = '%s_manifest.csv' % tag
    with os.popen('find %s -type f -name "*.wav"' % data_path) as pipe:
        with io.FileIO(manifest_path, "w") as file:
            for wav_path in pipe:
                wav_path = wav_path.strip()
                transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
                sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
                file.write(sample)


def main():
    args = parser.parse_args()
    train_path = args.root_path + '/train/'
    test_path = args.root_path + '/test/'
    create_manifest(train_path, 'train')
    create_manifest(test_path, 'test')


if __name__ == '__main__':
    main()
