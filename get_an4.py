import argparse
import os
import io
import shutil

parser = argparse.ArgumentParser(description='Processes and downloads an4.')
parser.add_argument('--an4_path', default='dataset/', help='Path to save dataset')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')


def format_data(data_tag, name, wav_folder):
    data_path = args.an4_path + data_tag + '/' + name + '/'
    new_transcript_path = data_path + '/txt/'
    new_wav_path = data_path + '/wav/'

    os.makedirs(new_transcript_path)
    os.makedirs(new_wav_path)

    wav_path = root_path + 'wav/'
    file_ids = root_path + 'etc/an4_%s.fileids' % data_tag
    transcripts = root_path + 'etc/an4_%s.transcription' % data_tag
    train_path = wav_path + wav_folder

    convert_audio_to_wav(train_path)
    format_files(file_ids, new_transcript_path, new_wav_path, transcripts, wav_path)


def convert_audio_to_wav(train_path):
    with os.popen('find %s -type f -name "*.raw"' % train_path) as pipe:
        for line in pipe:
            raw_path = line.strip()
            new_path = line.replace('.raw', '.wav').strip()
            cmd = 'sox -t raw -r %d -b 16 -e signed-integer -B -c 1 \"%s\" \"%s\"' % (
                args.sample_rate, raw_path, new_path)
            os.system(cmd)


def format_files(file_ids, new_transcript_path, new_wav_path, transcripts, wav_path):
    with open(file_ids, 'r') as f:
        with open(transcripts, 'r') as t:
            paths = f.readlines()
            transcripts = t.readlines()
            for x in range(len(paths)):
                path = wav_path + paths[x].strip() + '.wav'
                filename = path.split('/')[-1]
                extracted_transcript = process_transcript(transcripts, x)
                current_path = os.path.abspath(path)
                new_path = new_wav_path + filename
                text_path = new_transcript_path + filename.replace('.wav', '.txt')
                with io.FileIO(text_path, "w") as file:
                    file.write(extracted_transcript)
                os.rename(current_path, new_path)


def process_transcript(transcripts, x):
    extracted_transcript = transcripts[x].split('(')[0].strip("<s>").split('<')[0].strip().upper()
    return extracted_transcript


def main():
    global args, root_path
    args = parser.parse_args()
    root_path = 'an4/'
    name = 'an4'
    os.system('wget http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz')
    os.system('tar -xzvf an4_raw.bigendian.tar.gz')
    os.makedirs(args.an4_path)
    format_data('train', name, 'an4_clstk')
    format_data('test', name, 'an4test_clstk')
    shutil.rmtree(root_path)
    os.remove('an4_raw.bigendian.tar.gz')


if __name__ == '__main__':
    main()
