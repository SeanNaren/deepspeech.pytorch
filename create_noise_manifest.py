import argparse
import io
import os

parser = argparse.ArgumentParser(description='Creates noise manifest')
parser.add_argument('--root_path', default='noise', help='Path to the noise dataset')


def main():
    args = parser.parse_args()
    manifest_path = 'noise_manifest.csv'
    with os.popen('find %s -type f -name "*.wav"' % args.root_path) as pipe:
        with io.FileIO(manifest_path, "w") as file:
            for wav_path in pipe:
                wav_path = wav_path.strip()
                sample = os.path.abspath(wav_path) + '\n'
                file.write(sample)


if __name__ == '__main__':
    main()
