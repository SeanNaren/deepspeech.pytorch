import argparse
import os
import io
import shutil

import subprocess

from data.utils import create_manifest

parser = argparse.ArgumentParser(description='Processes and downloads an4.')
parser.add_argument('--an4_path', default='an4_dataset/', help='Path to save dataset')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
args = parser.parse_args()


def _format_data(root_path, data_tag, name, wav_folder):
    data_path = args.an4_path + data_tag + '/' + name + '/'
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
                args.sample_rate, raw_path, new_path)
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
                    file.write(extracted_transcript)
                os.rename(current_path, new_path)


def _process_transcript(transcripts, x):
    extracted_transcript = transcripts[x].split('(')[0].strip("<s>").split('<')[0].strip().upper()
    return extracted_transcript


def main():
    root_path = 'an4/'
    name = 'an4'
    subprocess.call(['wget http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz'], shell=True)
    subprocess.call(['tar -xzvf an4_raw.bigendian.tar.gz'], stdout=open(os.devnull, 'wb'), shell=True)
    os.makedirs(args.an4_path)
    _format_data(root_path, 'train', name, 'an4_clstk')
    _format_data(root_path, 'test', name, 'an4test_clstk')
    shutil.rmtree(root_path)
    os.remove('an4_raw.bigendian.tar.gz')
    train_path = args.an4_path + '/train/'
    test_path = args.an4_path + '/test/'
    print ('Creating manifests...')
    create_manifest(train_path, 'train')
    create_manifest(test_path, 'val')


if __name__ == '__main__':
    main()
