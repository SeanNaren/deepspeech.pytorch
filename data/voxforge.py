import os
import urllib.request
import argparse
import re
import tempfile
import shutil
import subprocess
import json

from utils import create_manifest

VOXFORGE_URL_16kHz = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'


parser = argparse.ArgumentParser(description='Processes and downloads VoxForge dataset.')
parser.add_argument( "--target_dir", type = str,
                     default = "voxforge", help = "Directory to store the dataset." )
parser.add_argument('--sample_rate', default=16000,
                    type=int, help='Sample rate')
parser.add_argument('--labels_path', default='./../labels.json', help='Contains all characters for prediction')

args = parser.parse_args()

def _process_transcript(transcript, labels):
    return "".join( [ c for c in transcript.strip().upper() if c in labels ] )

def _get_recordings_dir( sample_dir, recording_name ):
    wav_dir =  os.path.join( sample_dir, recording_name, "wav" )
    if os.path.exists(wav_dir):
        return "wav", wav_dir
    flac_dir = os.path.join( sample_dir, recording_name, "flac" )
    if os.path.exists( flac_dir ):
        return "flac", flac_dir
    raise Exception("wav or flac directory was not found for recording name: {}".format( recotding_name ))

def prepare_sample(recording_name, url, target_folder, labels):
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

            recordings_type, recordings_dir = _get_recordings_dir( dirpath, recording_name)
            tgz_prompt_file = os.path.join( dirpath, recording_name, "etc", "PROMPTS" )

            if os.path.exists( recordings_dir ) and os.path.exists( tgz_prompt_file ):
                transcriptions = open(tgz_prompt_file).read().strip().split("\n")
                transcriptions = {t.split()[0] : " ".join(t.split()[1:]) for t in transcriptions}
                for wav_file in os.listdir( recordings_dir ):
                   recording_id = wav_file.split('.{}'.format(recordings_type))[0]
                   transcription_key = recording_name + "/mfc/" + recording_id 
                   if transcription_key not in transcriptions:
                       continue
                   utterance = transcriptions[ transcription_key ]
                   utterance = _process_transcript(utterance, labels)

                   target_wav_file = os.path.join( wav_dir, "{}_{}.wav".format( recording_name, recording_id ) )
                   target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(recording_name, recording_id))
                   with open(target_txt_file, "w") as f:
                        f.write( utterance )
                   original_wav_file = os.path.join(recordings_dir, wav_file)
                   #shutil.copyfile( original_wav_file , target_wav_file  )
                   subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format( original_wav_file, str(args.sample_rate),
                                                                                target_wav_file )], shell = True)

            shutil.rmtree(dirpath)

if __name__ == '__main__':
    target_dir = args.target_dir
    sample_rate = args.sample_rate
    labels_path = args.labels_path

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    request = urllib.request.Request(VOXFORGE_URL_16kHz)
    with urllib.request.urlopen(request) as response:
        content = response.read()
        all_files = re.findall( "href\=\"(.*\.tgz)\"", content.decode("utf-8"))
        for f_idx,f in enumerate(all_files):
            print('Downloading {} / {} files'.format(f_idx, len(all_files)))
            prepare_sample(f.replace(".tgz", ""), VOXFORGE_URL_16kHz + '/' + f, target_dir, labels)
    print('Creating manifests...')
    create_manifest(target_dir, 'train')
