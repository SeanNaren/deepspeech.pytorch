import argparse
import os
import io
import shutil
import tarfile
import wget

from utils import create_manifest

parser = argparse.ArgumentParser(description='Processes and downloads an4.')
parser.add_argument('--target-dir', default='an4_dataset/', help='Path to save dataset')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes training samples longer than the max duration (given in seconds, default 15)')
args = parser.parse_args()


def _format_data(root_path, data_tag, name, wav_folder):
    data_path = args.target_dir + data_tag + '/' + name + '/'
    new_transcript_path = data_path + '/txt/'
    new_wav_path = data_path + '/wav/'

    os.makedirs(new_transcript_path)
    os.makedirs(new_wav_path)

    wav_path = root_path + 'wav/'
    file_ids = root_path + 'etc/an4_%s.fileids' % data_tag
    transcripts = root_path + 'etc/an4_%s.transcription' % data_tag
    train_path = wav_path + wav_folder

    _convert_audio_to_wav(train_path)
    _format_files(file_ids, new_transcript_path, new_wav_path, transcripts, wav_path)


def _convert_audio_to_wav(train_path):
    with os.popen('find %s -type f -name "*.raw"' % train_path) as pipe:
        for line in pipe:
            raw_path = line.strip()
            new_path = line.replace('.raw', '.wav').strip()
            cmd = 'sox -t raw -r %d -b 16 -e signed-integer -B -c 1 \"%s\" \"%s\"' % (
                16000, raw_path, new_path)
            os.system(cmd)


def _format_files(file_ids, new_transcript_path, new_wav_path, transcripts, wav_path):
    with open(file_ids, 'r') as f:
        with open(transcripts, 'r') as t:
            paths = f.readlines()
            transcripts = t.readlines()
            for x in range(len(paths)):
                path = wav_path + paths[x].strip() + '.wav'
                filename = path.split('/')[-1]
                extracted_transcript = _process_transcript(transcripts, x)
                current_path = os.path.abspath(path)
                new_path = new_wav_path + filename
                text_path = new_transcript_path + filename.replace('.wav', '.txt')
                with io.FileIO(text_path, "w") as file:
                    file.write(extracted_transcript.encode('utf-8'))
                os.rename(current_path, new_path)


def _process_transcript(transcripts, x):
    extracted_transcript = transcripts[x].split('(')[0].strip("<s>").split('<')[0].strip().upper()
    return extracted_transcript


def main():
    root_path = 'an4/'
    name = 'an4'
    wget.download('http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz')
    tar = tarfile.open('an4_raw.bigendian.tar.gz')
    tar.extractall()
    os.makedirs(args.target_dir)
    _format_data(root_path, 'train', name, 'an4_clstk')
    _format_data(root_path, 'test', name, 'an4test_clstk')
    shutil.rmtree(root_path)
    os.remove('an4_raw.bigendian.tar.gz')
    train_path = args.target_dir + '/train/'
    test_path = args.target_dir + '/test/'
    print ('\n', 'Creating manifests...')
    create_manifest(train_path, 'an4_train_manifest.csv', args.min_duration, args.max_duration)
    create_manifest(test_path, 'an4_val_manifest.csv')


if __name__ == '__main__':
    main()
