import argparse
import os
import io
import shutil
import tarfile

from sklearn.model_selection import train_test_split
import wget

from deepspeech_pytorch.data.data_opts import add_data_opts
from deepspeech_pytorch.data.utils import create_manifest


def _format_training_data(root_path,
                          val_fraction,
                          sample_rate,
                          target_dir):
    wav_path = root_path + 'wav/'
    file_ids_path = root_path + 'etc/an4_train.fileids'
    transcripts_path = root_path + 'etc/an4_train.transcription'
    root_wav_path = wav_path + 'an4_clstk'

    _convert_audio_to_wav(an4_audio_path=root_wav_path,
                          sample_rate=sample_rate)
    file_ids, transcripts = _retrieve_file_ids_and_transcripts(file_ids_path, transcripts_path)

    split_files = train_test_split(file_ids, transcripts, test_size=val_fraction)
    train_file_ids, val_file_ids, train_transcripts, val_transcripts = split_files

    _save_wav_transcripts(data_type='train',
                          file_ids=train_file_ids,
                          transcripts=train_transcripts,
                          wav_dir=wav_path,
                          target_dir=target_dir)
    _save_wav_transcripts(data_type='val',
                          file_ids=val_file_ids,
                          transcripts=val_transcripts,
                          wav_dir=wav_path,
                          target_dir=target_dir)


def _format_test_data(root_path,
                      sample_rate,
                      target_dir):
    wav_path = root_path + 'wav/'
    file_ids_path = root_path + 'etc/an4_test.fileids'
    transcripts_path = root_path + 'etc/an4_test.transcription'
    root_wav_path = wav_path + 'an4test_clstk'

    _convert_audio_to_wav(an4_audio_path=root_wav_path,
                          sample_rate=sample_rate)
    file_ids, transcripts = _retrieve_file_ids_and_transcripts(file_ids_path, transcripts_path)

    _save_wav_transcripts(data_type='test',
                          file_ids=file_ids,
                          transcripts=transcripts,
                          wav_dir=wav_path,
                          target_dir=target_dir)


def _save_wav_transcripts(data_type,
                          file_ids,
                          transcripts,
                          wav_dir,
                          target_dir):
    data_path = os.path.join(target_dir, data_type + '/an4/')
    new_transcript_dir = data_path + '/txt/'
    new_wav_dir = data_path + '/wav/'

    os.makedirs(new_transcript_dir)
    os.makedirs(new_wav_dir)

    _save_files(file_ids=file_ids,
                transcripts=transcripts,
                wav_dir=wav_dir,
                new_wav_dir=new_wav_dir,
                new_transcript_dir=new_transcript_dir)


def _convert_audio_to_wav(an4_audio_path, sample_rate):
    with os.popen('find %s -type f -name "*.raw"' % an4_audio_path) as pipe:
        for line in pipe:
            raw_path = line.strip()
            new_path = line.replace('.raw', '.wav').strip()
            cmd = 'sox -t raw -r %d -b 16 -e signed-integer -B -c 1 \"%s\" \"%s\"' % (
                sample_rate, raw_path, new_path)
            os.system(cmd)


def _save_files(file_ids, transcripts, wav_dir, new_wav_dir, new_transcript_dir):
    for file_id, transcript in zip(file_ids, transcripts):
        path = wav_dir + file_id.strip() + '.wav'
        filename = path.split('/')[-1]
        extracted_transcript = _process_transcript(transcript)
        new_path = new_wav_dir + filename
        text_path = new_transcript_dir + filename.replace('.wav', '.txt')
        with io.FileIO(text_path, "w") as file:
            file.write(extracted_transcript.encode('utf-8'))
        current_path = os.path.abspath(path)
        shutil.copy(current_path, new_path)
        os.remove(current_path)


def _retrieve_file_ids_and_transcripts(file_id_path, transcripts_path):
    with open(file_id_path, 'r') as f:
        file_ids = f.readlines()
    with open(transcripts_path, 'r') as t:
        transcripts = t.readlines()
    return file_ids, transcripts


def _process_transcript(transcript):
    """
    Removes tags found in AN4.
    """
    extracted_transcript = transcript.split('(')[0].strip("<s>").split('<')[0].strip().upper()
    return extracted_transcript


def download_an4(target_dir: str,
                 manifest_dir: str,
                 min_duration: float,
                 max_duration: float,
                 val_fraction: float,
                 sample_rate: int,
                 num_workers: int):
    root_path = 'an4/'
    raw_tar_path = 'an4_raw.bigendian.tar.gz'
    if not os.path.exists(raw_tar_path):
        wget.download('http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz')
    tar = tarfile.open('an4_raw.bigendian.tar.gz')
    tar.extractall()
    os.makedirs(target_dir, exist_ok=True)
    _format_training_data(root_path=root_path,
                          val_fraction=val_fraction,
                          sample_rate=sample_rate,
                          target_dir=target_dir)
    _format_test_data(root_path=root_path,
                      sample_rate=sample_rate,
                      target_dir=target_dir)
    shutil.rmtree(root_path)
    os.remove('an4_raw.bigendian.tar.gz')
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
    parser.add_argument('--val-fraction', default=0.1, type=float,
                        help='Number of files in the training set to use as validation.')
    args = parser.parse_args()
    download_an4(
        target_dir=args.target_dir,
        manifest_dir=args.manifest_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        val_fraction=args.val_fraction,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers
    )
