import os
from six.moves import urllib
import argparse
import re
import tempfile
import shutil
import subprocess
import tarfile
import io
from tqdm import tqdm

from utils import create_manifest

VOXFORGE_URL_16kHz = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'

parser = argparse.ArgumentParser(description='Processes and downloads VoxForge dataset.')
parser.add_argument("--target-dir", default='voxforge_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument('--sample-rate', default=16000,
                    type=int, help='Sample rate')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes training samples longer than the max duration (given in seconds, default 15)')
args = parser.parse_args()


def _get_recordings_dir(sample_dir, recording_name):
    wav_dir = os.path.join(sample_dir, recording_name, "wav")
    if os.path.exists(wav_dir):
        return "wav", wav_dir
    flac_dir = os.path.join(sample_dir, recording_name, "flac")
    if os.path.exists(flac_dir):
        return "flac", flac_dir
    raise Exception("wav or flac directory was not found for recording name: {}".format(recording_name))


def prepare_sample(recording_name, url, target_folder):
    """
    Downloads and extracts a sample from VoxForge and puts the wav and txt files into :target_folder.
    """
    wav_dir = os.path.join(target_folder, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    txt_dir = os.path.join(target_folder, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    # check if sample is processed
    filename_set = set(['_'.join(wav_file.split('_')[:-1]) for wav_file in os.listdir(wav_dir)])
    if recording_name in filename_set:
        return

    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    content = response.read()
    response.close()
    with tempfile.NamedTemporaryFile(suffix=".tgz", mode='wb') as target_tgz:
        target_tgz.write(content)
        target_tgz.flush()
        dirpath = tempfile.mkdtemp()

        tar = tarfile.open(target_tgz.name)
        tar.extractall(dirpath)
        tar.close()

        recordings_type, recordings_dir = _get_recordings_dir(dirpath, recording_name)
        tgz_prompt_file = os.path.join(dirpath, recording_name, "etc", "PROMPTS")

        if os.path.exists(recordings_dir) and os.path.exists(tgz_prompt_file):
            transcriptions = open(tgz_prompt_file).read().strip().split("\n")
            transcriptions = {t.split()[0]: " ".join(t.split()[1:]) for t in transcriptions}
            for wav_file in os.listdir(recordings_dir):
                recording_id = wav_file.split('.{}'.format(recordings_type))[0]
                transcription_key = recording_name + "/mfc/" + recording_id
                if transcription_key not in transcriptions:
                    continue
                utterance = transcriptions[transcription_key]

                target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(recording_name, recording_id))
                target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(recording_name, recording_id))
                with io.FileIO(target_txt_file, "w") as file:
                    file.write(utterance.encode('utf-8'))
                original_wav_file = os.path.join(recordings_dir, wav_file)
                subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(original_wav_file, str(args.sample_rate),
                                                                      target_wav_file)], shell=True)

        shutil.rmtree(dirpath)


if __name__ == '__main__':
    target_dir = args.target_dir
    sample_rate = args.sample_rate

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    request = urllib.request.Request(VOXFORGE_URL_16kHz)
    response = urllib.request.urlopen(request)
    content = response.read()
    all_files = re.findall("href\=\"(.*\.tgz)\"", content.decode("utf-8"))
    for f in tqdm(all_files, total=len(all_files)):
        prepare_sample(f.replace(".tgz", ""), VOXFORGE_URL_16kHz + f, target_dir)
    print('Creating manifests...')
    create_manifest(target_dir, 'voxforge_train_manifest.csv', args.min_duration, args.max_duration)
