import os
import urllib.request
import argparse
import re
import tempfile
import shutil
import subprocess

from utils import create_manifest

VOXFORGE_URL_48kHz = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/48kHz_16bit'
VOXFORGE_URL_16kHz = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'


parser = argparse.ArgumentParser(description='Processes and downloads VoxForge dataset.')
parser.add_argument( "--target_dir", type = str,
                     default = "voxforge", help = "Directory to store the dataset." )
parser.add_argument('--sample_rate', default=16000,
                    type=int, help='Sample rate')
args = parser.parse_args()


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

    request = urllib.request.Request(url)
    with urllib.request.urlopen(request) as response:
        content = response.read()
        with tempfile.NamedTemporaryFile( suffix = ".tgz" ) as target_tgz:
            target_tgz.write( content )
            dirpath = tempfile.mkdtemp()
            subprocess.call(["tar zxvf {} -C {}".format(target_tgz.name, dirpath)], shell=True)

            tgz_wav_dir = os.path.join( dirpath, recording_name, "wav" )
            tgz_prompt_dir = os.path.join( dirpath, recording_name, "etc", "PROMPTS" )

            transcriptions = open(tgz_prompt_dir).read().strip().split("\n")
            transcriptions = {t.split()[0] : " ".join(t.split()[1:]) for t in transcriptions if len(t) > 0}
            assert os.path.exists(tgz_wav_dir) and os.path.exists( tgz_prompt_dir ), \
                "wav or PROMPTS dir is not found in the archive "
            for wav_file in os.listdir( tgz_wav_dir ):
                recording_id = wav_file.split('.wav')[0]

                utterance = transcriptions[ recording_name + "/mfc/" + recording_id ]

                target_wav_file = os.path.join( wav_dir, "{}_{}.wav".format( recording_name, recording_id ) )
                target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(recording_name, recording_id))
                with open(target_txt_file, "w") as f:
                    f.write( utterance )
                shutil.copyfile( os.path.join(tgz_wav_dir, wav_file), target_wav_file  )

            shutil.rmtree(dirpath)

if __name__ == '__main__':
    target_dir = args.target_dir
    sample_rate = args.sample_rate

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    request = urllib.request.Request(VOXFORGE_URL_16kHz)
    with urllib.request.urlopen(request) as response:
        content = response.read()
        all_files = re.findall( "href\=\"(.*\.tgz)\"", content.decode("utf-8"))
        for f_idx,f in enumerate(all_files):
            print('Downloading {} / {} files'.format(f_idx, len(all_files)))
            prepare_sample(f.replace(".tgz", ""), VOXFORGE_URL_16kHz + '/' + f, target_dir)
    print('Creating manifests...')
    create_manifest(target_dir, 'train')